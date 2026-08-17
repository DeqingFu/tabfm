# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA test-time training (TTT) for the PyTorch TabFM regressor.

Wraps a plain, unmodified ``TabFMRegressor``. Per ensemble member: sample
independent random context/query splits of the training rows, process
``batch_size`` splits together per forward pass, average gradients over
``gradient_accumulation_steps`` such microbatches, do one optimizer update, and
repeat for ``steps`` updates while maintaining an EMA of the adapter weights.
The EMA is deployed. The wrapped regressor's ensemble weights (NNLS when
enabled) are left exactly as upstream fitted them, so they stay leakage-free
and ``steps=0`` is bit-exactly the plain ensemble. There is no early stopping
or checkpoint selection.

Two measured properties worth knowing before tuning this:

* The checkpoint runs in bfloat16, where batching changes the forward by ~8%
  relative (float32 agrees to 3.7e-6). So ``batch_size`` and
  ``gradient_accumulation_steps`` are gradient-identical in exact arithmetic
  but not interchangeable in practice -- swapping them moves results by about
  the size of the whole adaptation effect. Record which was used.
* Expect a modest average gain, not the size a single dataset suggests. On
  four datasets not used for tuning the mean gain is about -1.1% at
  n_estimators=1 and -0.4% at n_estimators=8, with two of four datasets
  regressing at the default learning rate.

Example:
  reg = TabFMRegressor(model, n_estimators=8, random_state=0)
  ttt = TestTimeTrainedRegressor(reg).fit(X_train, y_train)
  predictions = ttt.predict(X_test)
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize as opt

try:
  import torch

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

from tabfm.src.pytorch.model import (
    _COL_CHUNK_SIZE,
    _FFN_CHUNK_SIZE,
    _ROW_CHUNK_SIZE,
)
from tabfm.src.classifier_and_regressor import (
    _apply_categorical_permutation,
    _check_regressor_output_dim,
    _pad_cat_mask,
    _pad_features,
    _predict_step_pytorch,
)

# Matches classifier_and_regressor's default so TTT sampling is reproducible
# when the wrapped regressor leaves random_state unset.
_DEFAULT_RANDOM_STATE = 42

_TARGET_LAYER_ALIASES = {
    "cell": "cell_embedder.in_linear",
    "cellembedder": "cell_embedder",
    "cell_embedder": "cell_embedder",
    "icl": "icl_predictor",
    "iclpredictor": "icl_predictor",
    "icl_predictor": "icl_predictor",
}

_CONFIG_ALIASES = {
    "lr": "learning_rate",
    "ema": "ema_decay",
    "checkpointing": "gradient_checkpointing",
    "bf16": "cast_model_to_bfloat16",
}

_POSITIVE_INT_FIELDS = (
    "lora_rank", "batch_size", "gradient_accumulation_steps",
    "min_test_rows",
)
_OPTIONAL_POSITIVE_INT_FIELDS = (
    "cell_embedder_row_chunk_size", "col_embedder_chunk_size",
    "row_interactor_chunk_size", "ffn_chunk_size", "max_train_rows",
    "max_test_rows",
)


def _coerce_target_layers(value: Any) -> Tuple[str, ...]:
  """Normalizes user-friendly target layer names."""
  if value is None:
    return TabFMTestTimeTraining.target_layers
  raw = (
      [p.strip() for p in value.split(",") if p.strip()]
      if isinstance(value, str)
      else list(value)
  )
  layers = []
  for layer in raw:
    key = "".join(c for c in str(layer).lower() if c.isalnum() or c == "_")
    layers.append(_TARGET_LAYER_ALIASES.get(key, str(layer).strip()))
  return tuple(layers)


def _is_int(value: Any) -> bool:
  return not isinstance(value, bool) and isinstance(value, (int, np.integer))


@dataclass
class TabFMTestTimeTraining:
  """Specification for LoRA test-time training.

  Leave the row caps at None unless memory requires otherwise: at prediction
  time each member sees the full training set in context, and adapters
  trained against a smaller capped context lose their gains under that
  mismatch. ``cast_model_to_bfloat16`` requires CUDA, is a no-op for params
  already in bfloat16, and mutates the supplied model's base weights in
  place; set it to False on CPU.

  The chunk sizes default to None, meaning the model's own defaults. Smaller
  values were measured not to lower the backward pass's peak memory at all
  while serializing the computation (row_interactor at 64 splits it into ~63
  chunks and costs ~15% wall clock); ``gradient_checkpointing`` is the knob
  that actually trades memory for time.
  """

  lora_rank: int = 8
  target_layers: Tuple[str, ...] = ("cell_embedder.in_linear",)
  steps: int = 64
  learning_rate: float = 1e-2
  optimizer: str = "adamw"
  # Set explicitly rather than left to the optimizer: torch defaults differ
  # (AdamW 0.01, Muon 0.1), which would silently make the two incomparable.
  weight_decay: float = 0.01
  learning_rate_schedule: str = "cosine"
  # Floor for the cosine schedule. Annealing all the way to zero spends the
  # last steps not moving; a small floor keeps them useful.
  min_learning_rate: float = 1e-5
  batch_size: int = 4
  gradient_accumulation_steps: int = 1
  ema_decay: float = 0.9
  test_fraction: float = 0.2
  max_train_rows: Optional[int] = None
  max_test_rows: Optional[int] = None
  min_test_rows: int = 1
  cell_embedder_row_chunk_size: Optional[int] = None
  col_embedder_chunk_size: Optional[int] = None
  row_interactor_chunk_size: Optional[int] = None
  ffn_chunk_size: Optional[int] = None
  gradient_checkpointing: Union[bool, str] = "none"
  cast_model_to_bfloat16: bool = True
  seed: Optional[int] = None

  @classmethod
  def from_value(cls, value):
    """Builds a spec from None/False, True, a dict, or an instance."""
    if value is None or value is False:
      return None
    if value is True:
      return cls()
    if isinstance(value, cls):
      # Snapshot so later caller mutations cannot change the deployed spec.
      return replace(value)
    if not isinstance(value, dict):
      raise TypeError(
          "test_time_training must be a dict, TabFMTestTimeTraining, True,"
          f" or None; got {type(value)!r}."
      )
    normalized = {}
    for key, val in value.items():
      key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
      key = _CONFIG_ALIASES.get(key, key)
      if key == "target_layers":
        val = _coerce_target_layers(val)
      normalized[key] = val
    return cls(**normalized)

  def __post_init__(self):
    self.target_layers = _coerce_target_layers(self.target_layers)
    if not self.target_layers:
      raise ValueError("target_layers must contain at least one layer.")
    for name in _POSITIVE_INT_FIELDS:
      if not _is_int(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if not _is_int(self.steps) or self.steps < 0:
      raise ValueError("steps must be a non-negative integer.")
    for name in _OPTIONAL_POSITIVE_INT_FIELDS:
      value = getattr(self, name)
      if value is not None and (not _is_int(value) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None.")
    if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
      raise ValueError("learning_rate must be positive and finite.")
    if self.optimizer not in ("adamw", "muon"):
      raise ValueError(
          f"optimizer must be 'adamw' or 'muon', got {self.optimizer!r}."
      )
    if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
      raise ValueError("weight_decay must be non-negative and finite.")
    # Only the cosine schedule reads min_learning_rate, so only it constrains
    # the value; a constant schedule must stay usable at any learning rate.
    if self.learning_rate_schedule == "cosine" and (
        not math.isfinite(self.min_learning_rate)
        or not 0 <= self.min_learning_rate <= self.learning_rate
    ):
      raise ValueError(
          "min_learning_rate must be finite and in [0, learning_rate] when "
          "learning_rate_schedule is 'cosine'."
      )
    if self.learning_rate_schedule not in ("constant", "cosine"):
      raise ValueError(
          "learning_rate_schedule must be 'constant' or 'cosine', got "
          f"{self.learning_rate_schedule!r}."
      )
    if not math.isfinite(self.ema_decay) or not 0.0 < self.ema_decay < 1.0:
      raise ValueError("ema_decay must be in (0, 1).")
    if (
        not math.isfinite(self.test_fraction)
        or not 0.0 < self.test_fraction < 1.0
    ):
      raise ValueError("test_fraction must be in (0, 1).")
    if not isinstance(self.cast_model_to_bfloat16, bool):
      raise TypeError("cast_model_to_bfloat16 must be a bool.")
    if isinstance(self.gradient_checkpointing, bool):
      self.gradient_checkpointing = (
          "all" if self.gradient_checkpointing else "none"
      )
    self.gradient_checkpointing = str(self.gradient_checkpointing).lower()
    if self.gradient_checkpointing not in ("all", "icl", "none"):
      raise ValueError("gradient_checkpointing must be 'all', 'icl', 'none'.")
    if self.seed is not None and not _is_int(self.seed):
      raise TypeError("seed must be an integer or None.")


if HAS_TORCH:

  class _LoRALinear(torch.nn.Module):
    """LoRA adapter around a frozen Linear layer (scaling fixed at 1)."""

    def __init__(self, base, rank, adapter_dtype=None, initialize=True):
      super().__init__()
      self.base = base
      self.rank = rank
      for param in self.base.parameters():
        param.requires_grad = False
      kwargs = dict(
          device=base.weight.device, dtype=adapter_dtype or base.weight.dtype
      )
      self.lora_A = torch.nn.Parameter(
          torch.empty(rank, base.in_features, **kwargs)
      )
      self.lora_B = torch.nn.Parameter(
          torch.empty(base.out_features, rank, **kwargs)
      )
      if initialize:
        self.reset_lora_parameters()
      else:
        # Wrappers are installed right before a seeded reset (fit) or a
        # saved-state load (inference); zero init avoids consuming the
        # caller's ambient Torch RNG here.
        torch.nn.init.zeros_(self.lora_A)
        torch.nn.init.zeros_(self.lora_B)

    def reset_lora_parameters(self):
      # Standard LoRA init: A random, B zero -> starts as an exact no-op.
      torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
      torch.nn.init.zeros_(self.lora_B)

    @property
    def in_features(self):
      return self.base.in_features

    @property
    def out_features(self):
      return self.base.out_features

    @property
    def weight(self):
      return self.base.weight

    @property
    def bias(self):
      return self.base.bias

    def forward(self, x):
      out = self.base(x)
      lora = torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A)
      lora = torch.nn.functional.linear(lora, self.lora_B)
      return out + lora.to(out.dtype)

else:
  _LoRALinear = None


def _get_submodule(model, name):
  try:
    return model.get_submodule(name)
  except AttributeError:
    module = model
    for part in name.split("."):
      module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_submodule(model, name, value):
  if "." not in name:
    setattr(model, name, value)
    return
  parent_name, child_name = name.rsplit(".", 1)
  parent = _get_submodule(model, parent_name)
  if child_name.isdigit():
    parent[int(child_name)] = value
  else:
    setattr(parent, child_name, value)


def _ensure_lora_adapters(model, config, adapter_dtype=None):
  """Wraps target Linear layers with LoRA; returns the wrapped names."""
  if not HAS_TORCH or _LoRALinear is None:
    raise ImportError("PyTorch is required for test-time training.")
  wrapped, seen = [], set()
  for target_layer in config.target_layers:
    try:
      target = _get_submodule(model, target_layer)
    except (AttributeError, KeyError, IndexError) as exc:
      raise ValueError(f"Unknown TTT target layer: {target_layer!r}.") from exc
    if isinstance(target, (torch.nn.Linear, _LoRALinear)):
      candidates = [(target_layer, target)]
    else:
      adapter_paths = {
          n for n, c in target.named_modules()
          if n and isinstance(c, _LoRALinear)
      }
      candidates = [
          (f"{target_layer}.{n}" if n else target_layer, c)
          for n, c in target.named_modules()
          if isinstance(c, _LoRALinear)
          or (
              isinstance(c, torch.nn.Linear)
              and not any(n.startswith(f"{p}.") for p in adapter_paths)
          )
      ]
    for full_name, child in candidates:
      if full_name in seen:
        continue
      seen.add(full_name)
      if not isinstance(child, _LoRALinear):
        _set_submodule(
            model,
            full_name,
            _LoRALinear(
                child, config.lora_rank, adapter_dtype, initialize=False
            ),
        )
      wrapped.append(full_name)
  if not wrapped:
    raise ValueError(
        f"No Linear layers found under target_layers={config.target_layers!r}."
    )
  return wrapped


def _make_optimizer(config, params):
  """Builds the adapter optimizer named by the config.

  Muon orthogonalizes the momentum before stepping, so its updates have a
  fixed scale and its learning rate does not carry over from AdamW's -- it
  needs its own tuning. It also assumes the matrix it steps is the weight
  being learned, which a LoRA pair only approximates: orthogonalizing A and B
  separately does not orthogonalize their product.
  """
  if config.optimizer == "adamw":
    return torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
  if config.optimizer == "muon":
    return torch.optim.Muon(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
  raise ValueError(
      f"Unknown optimizer {config.optimizer!r}; expected 'adamw' or 'muon'."
  )


def _reset_lora_adapters(model, seed=None):
  adapters = [m for m in model.modules() if isinstance(m, _LoRALinear)]
  if seed is None:
    for module in adapters:
      module.reset_lora_parameters()
    return
  device = next(model.parameters()).device
  cuda_index = None
  if getattr(device, "type", None) == "cuda":
    cuda_index = (
        device.index if device.index is not None
        else torch.cuda.current_device()
    )
  # Adapter init is TTT's only stochastic Torch op; fork the RNG so the seed
  # is reproducible without perturbing the caller's generator state.
  with torch.random.fork_rng(devices=[cuda_index] if cuda_index is not None else []):
    torch.random.default_generator.manual_seed(int(seed))
    if cuda_index is not None:
      torch.cuda.default_generators[cuda_index].manual_seed(int(seed))
    for module in adapters:
      module.reset_lora_parameters()


def _remove_lora_adapters(model):
  """Removes all LoRA wrappers, restoring their base modules."""
  if not HAS_TORCH or _LoRALinear is None:
    return

  def _unwrap(module):
    for child_name, child in list(module.named_children()):
      if isinstance(child, _LoRALinear):
        base = child.base
        while isinstance(base, _LoRALinear):
          base = base.base
        module._modules[child_name] = base
      else:
        _unwrap(child)

  _unwrap(model)


def _lora_named_parameters(model):
  params = {}
  for module_name, module in model.named_modules():
    if _LoRALinear is not None and isinstance(module, _LoRALinear):
      prefix = f"{module_name}." if module_name else ""
      params[f"{prefix}lora_A"] = module.lora_A
      params[f"{prefix}lora_B"] = module.lora_B
  return params


def _freeze_non_lora_parameters(model):
  lora_ids = {id(p) for p in _lora_named_parameters(model).values()}
  for param in model.parameters():
    param.requires_grad = id(param) in lora_ids


def _requires_grad_state(model):
  return {n: p.requires_grad for n, p in model.named_parameters()}


def _restore_requires_grad_state(model, state):
  for name, param in model.named_parameters():
    if name in state:
      param.requires_grad = state[name]


def _lora_state_dict(model):
  state = {
      n: p.detach().cpu().clone()
      for n, p in _lora_named_parameters(model).items()
  }
  bad = [n for n, v in state.items() if not bool(torch.isfinite(v).all())]
  if bad:
    raise FloatingPointError(
        "Non-finite LoRA parameters: " + ", ".join(sorted(bad))
    )
  return state


def _load_lora_state_dict(model, state):
  """Strictly and atomically loads one complete LoRA adapter state.

  Exact-key matching prevents a missing tensor from inheriting the previous
  member's value and prevents malformed states from touching base weights.
  """
  if not isinstance(state, dict):
    raise TypeError(f"LoRA state must be a dict; got {type(state)!r}.")
  params = _lora_named_parameters(model)
  missing = sorted(set(params) - set(state))
  unexpected = sorted(set(state) - set(params))
  if missing or unexpected:
    raise ValueError(
        f"LoRA state keys mismatch (missing={missing!r},"
        f" unexpected={unexpected!r})."
    )
  converted = {}
  for name, param in params.items():
    value = state[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
        param.shape
    ):
      raise ValueError(f"Bad LoRA state tensor for {name!r}.")
    value = value.to(device=param.device, dtype=param.dtype)
    if not bool(torch.isfinite(value).all()):
      raise ValueError(f"LoRA state tensor {name!r} is non-finite.")
    converted[name] = value
  with torch.no_grad():
    for name, param in params.items():
      param.copy_(converted[name])


def _configure_model_for_ttt(model, config):
  """Applies chunking and gradient checkpointing for TTT backward passes.

  A None chunk size means "the model's own default", not "leave whatever a
  previous fit installed" -- the model object is shared, so the settings must
  be fully determined by this config.
  """
  pairs = [
      ("cell_embedder", "row_chunk_size",
       config.cell_embedder_row_chunk_size, _ROW_CHUNK_SIZE),
      ("row_interactor", "row_chunk_size",
       config.row_interactor_chunk_size, _ROW_CHUNK_SIZE),
      ("row_interactor_2", "row_chunk_size",
       config.row_interactor_chunk_size, _ROW_CHUNK_SIZE),
  ]
  for attr_owner, attr, value, fallback in pairs:
    module = getattr(model, attr_owner, None)
    if module is not None and hasattr(module, attr):
      setattr(module, attr, fallback if value is None else value)
  for attr, value, fallback in (
      ("col_chunk_size", config.col_embedder_chunk_size, _COL_CHUNK_SIZE),
      ("ffn_chunk_size", config.ffn_chunk_size, _FFN_CHUNK_SIZE),
  ):
    for module in model.modules():
      if hasattr(module, attr):
        setattr(module, attr, fallback if value is None else value)

  mode = config.gradient_checkpointing
  checkpointable = [
      m for m in model.modules() if hasattr(m, "gradient_checkpointing")
  ]
  for module in checkpointable:
    module.gradient_checkpointing = mode == "all"
  if mode == "icl":
    try:
      icl = _get_submodule(model, "icl_predictor.tf_icl")
      if hasattr(icl, "gradient_checkpointing"):
        icl.gradient_checkpointing = True
    except (AttributeError, KeyError, IndexError):
      pass


def _cast_base_model_to_bfloat16(model):
  """Casts frozen base params to bf16; RMSNorm/per-dim-scale/LoRA stay fp32."""
  skip = {id(p) for p in _lora_named_parameters(model).values()}
  for module in model.modules():
    if module.__class__.__name__ == "RMSNorm":
      skip.update(id(p) for p in module.parameters(recurse=False))
    pds = getattr(module, "per_dim_scale", None)
    if isinstance(pds, torch.nn.Parameter):
      skip.add(id(pds))
  for param in model.parameters():
    if (
        param.is_floating_point()
        and id(param) not in skip
        and param.dtype != torch.bfloat16
    ):
      param.data = param.data.to(torch.bfloat16)


def _forward_tensor(model, X, y, train_size, ds, cat_mask):
  """Grad-enabled twin of upstream _predict_step_pytorch (returns a tensor)."""
  device = next(model.parameters()).device
  X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
  y_t = torch.from_numpy(y).to(device)
  if y_t.dtype == torch.float64:
    y_t = y_t.to(torch.float32)
  batch = X.shape[0]
  train_size_t = torch.full((batch,), train_size, dtype=torch.long,
                            device=device)
  d_t = (
      torch.from_numpy(ds).to(device)
      if ds is not None
      else torch.full((batch,), X.shape[-1], dtype=torch.long, device=device)
  )
  cat_mask_t = (
      torch.from_numpy(cat_mask).to(device) if cat_mask is not None else None
  )
  return model(X_t, y_t, train_size_t, cat_mask=cat_mask_t, d=d_t)


class TestTimeTrainedRegressor:
  """Test-time-training wrapper around a plain TabFMRegressor.

  ``fit`` fits the wrapped regressor exactly as upstream defines it, then
  trains one LoRA adapter per ensemble member. ``predict`` runs the
  regressor's ensemble views through the adapted model and combines member
  predictions with the regressor's own rule. The wrapped regressor keeps
  producing unadapted baseline predictions through its own ``predict``.
  """

  __test__ = False  # not a pytest class despite the Test* name

  def __init__(self, regressor, config=True):
    self.regressor = regressor
    resolved = TabFMTestTimeTraining.from_value(config)
    if resolved is None:
      raise ValueError("TestTimeTrainedRegressor requires a configuration.")
    self.config = resolved

  def fit(self, X, y):
    """Fits the wrapped regressor, then trains the TTT adapters."""
    if getattr(self.regressor, "cache_context", False):
      raise ValueError(
          "cache_context=True cannot be combined with test-time training:"
          " adapter states must participate in each forward pass."
    )
    self._clear_fitted_state()
    # Fit the plain estimator exactly as upstream defines it, then adapt. Its
    # OOF predictions and NNLS weights stay untouched: reg.predict() remains a
    # real baseline, and the ensemble weights the adapted predictions are
    # combined with are upstream's own leakage-free out-of-fold weights.
    self.regressor.fit(X, y)
    try:
      self._fit_adapters()
      # Refit the ensemble weights on the adapted members. They are kept on
      # the wrapper, so reg.predict() stays the unadapted baseline. This
      # reuses the deployed adapters, which have seen every training row, so
      # the out-of-fold predictions feeding NNLS are mildly optimistic --
      # accepted deliberately: the alternative (fold-local adapters) costs
      # one adapter round per fold, and the weights matter enough that stale
      # ones are worse than slightly leaky ones.
      if self.test_time_training_.steps and getattr(
          self.regressor, "enable_nnls", False
      ):
        self.test_time_training_ensemble_weights_ = self._refit_nnls_weights(y)
    except Exception:
      self._clear_fitted_state()
      raise
    return self

  def predict(self, X):
    """Predicts with the adapted ensemble."""
    if not hasattr(self, "test_time_training_adapter_states_"):
      raise RuntimeError("This TestTimeTrainedRegressor is not fitted yet.")
    # Besides avoiding unnecessary work, delegation guarantees bit-exact
    # baseline behavior even if a zero LoRA branch or different member batching
    # would otherwise perturb floating-point execution.
    if self.test_time_training_.steps == 0:
      return self.regressor.predict(X)
    scaled = self._predict_scaled_with_adapters(X)
    weights = getattr(self, "test_time_training_ensemble_weights_", None)
    if weights is None:
      return self.regressor._combine_predictions(scaled)
    return np.dot(
        weights,
        np.stack([self.regressor._inverse_transform_y(row) for row in scaled]),
    )

  # -- fitted-ensemble views -------------------------------------------------
  def _flat_ensemble_configs(self):
    return [
        (norm, cfg)
        for norm, cfgs in
        self.regressor.ensemble_generator_.ensemble_configs_.items()
        for cfg in cfgs
    ]

  def _flat_ttt_configs(self):
    """Members' views projected onto original input features (engineered
    crosses/SVD columns hurt adaptation; they stay in inference views only)."""
    n_original = self.regressor.ensemble_generator_.n_original_features_
    flat = []
    for norm, (pattern, shift, cat_perm, rows) in self._flat_ensemble_configs():
      pattern = np.asarray(pattern, dtype=np.int64)
      pattern = pattern[pattern < n_original].copy()
      if pattern.size == 0:
        raise ValueError("TTT requires at least one original input feature.")
      flat.append((norm, (pattern, shift, cat_perm, rows)))
    return flat

  def _active_ttt_indices(self, member_idx):
    rows = self.test_time_training_flat_configs_[member_idx][1][3]
    if rows is None:
      generator = self.regressor.ensemble_generator_
      return np.arange(len(generator.y_), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)

  # -- sampling and loss -------------------------------------------------------
  def _sample_ttt_train_query(self, train_pool, rng, config):
    train_pool = np.asarray(train_pool, dtype=np.int64)
    if train_pool.size <= 1:
      return train_pool, np.array([], dtype=np.int64)
    shuffled = rng.permutation(train_pool)
    query_size = max(
        config.min_test_rows,
        int(round(train_pool.size * config.test_fraction)),
    )
    if config.max_test_rows is not None:
      query_size = min(query_size, config.max_test_rows)
    query_size = min(train_pool.size - 1, query_size)
    query_idx, context_idx = shuffled[:query_size], shuffled[query_size:]
    if (
        config.max_train_rows is not None
        and context_idx.size > config.max_train_rows
    ):
      context_idx = rng.choice(
          context_idx, size=config.max_train_rows, replace=False
      )
    return context_idx, query_idx

  def _member_ttt_arrays(self, member_idx, context_idx, query_idx):
    """Builds one member's (X, y, y_query, cat_mask, d) arrays for a split."""
    norm, (pattern, _, cat_perm, _) = self.test_time_training_flat_configs_[
        member_idx
    ]
    generator = self.regressor.ensemble_generator_
    preprocessor = generator.preprocessors_[norm]
    context_idx = np.asarray(context_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    if cat_perm:
      X_full = np.concatenate(
          [generator.X_[context_idx], generator.X_[query_idx]], axis=0
      ).copy()
      _apply_categorical_permutation(X_full, cat_perm)
      X_variant = preprocessor.transform(X_full)
    else:
      X_transformed = getattr(preprocessor, "X_transformed_", None)
      if X_transformed is None:
        X_transformed = preprocessor.transform(generator.X_)
      X_variant = np.concatenate(
          [X_transformed[context_idx], X_transformed[query_idx]], axis=0
      )

    X_variant = _pad_features(
        X_variant[:, pattern], self.test_time_training_max_features_
    )
    y_context = generator.y_[context_idx]
    y_full = np.concatenate([
        y_context,
        np.full(query_idx.shape[0], -100.0, dtype=y_context.dtype),
    ])
    mask = np.zeros(generator.n_features_in_, dtype=np.bool_)
    if hasattr(generator, "cat_features_"):
      mask[generator.cat_features_] = True
    cat_mask = _pad_cat_mask(
        mask[pattern], self.test_time_training_max_features_
    )
    return (
        X_variant[None, ...].astype(np.float32),
        y_full[None, ...],
        generator.y_[query_idx].astype(np.float32),
        cat_mask[None, ...],
        np.array([len(pattern)], dtype=np.int32),
    )

  def _member_ttt_batch_arrays(self, member_idx, splits):
    """Stacks one member's per-split arrays into a single batch.

    For a fixed member, context/query lengths depend only on the active row
    count and the config -- not on which rows a draw picked -- so every split
    has equal length and stacks with no padding.
    """
    per_split = [
        self._member_ttt_arrays(member_idx, context_idx, query_idx)
        for context_idx, query_idx in splits
    ]
    X_batch = np.concatenate([p[0] for p in per_split], axis=0)
    y_batch = np.concatenate([p[1] for p in per_split], axis=0)
    y_query_batch = np.stack([p[2] for p in per_split], axis=0)
    cat_mask_batch = np.concatenate([p[3] for p in per_split], axis=0)
    d_batch = np.concatenate([p[4] for p in per_split], axis=0)
    train_size = len(np.asarray(splits[0][0]))
    return X_batch, y_batch, y_query_batch, cat_mask_batch, d_batch, train_size

  def _ttt_loss_for_member_batched(self, member_idx, splits):
    """MSE over the query predictions, batched across independent splits.

    One forward pass over ``len(splits)`` splits at once instead of one
    sequential forward pass per split -- the frozen backbone otherwise runs
    hundreds of small ops per split at batch size 1, which starves the GPU.
    """
    X, y, y_query, cat_mask, d, train_size = self._member_ttt_batch_arrays(
        member_idx, splits
    )
    out = _forward_tensor(self.regressor.model, X, y, train_size, d, cat_mask)
    pred = out[:, train_size:, :].squeeze(-1)
    target = torch.from_numpy(y_query).to(pred.device, pred.dtype)
    return torch.nn.functional.mse_loss(pred.float(), target.float())

  # -- adapter fitting ---------------------------------------------------------
  def _clear_fitted_state(self):
    for attr in (
        "test_time_training_",
        "test_time_training_flat_configs_",
        "test_time_training_max_features_",
        "test_time_training_wrapped_layers_",
        "test_time_training_adapter_states_",
        "test_time_training_ensemble_weights_",
    ):
      if hasattr(self, attr):
        delattr(self, attr)
    model = getattr(self.regressor, "model", None)
    if HAS_TORCH and isinstance(model, torch.nn.Module):
      # A loader-cached model may carry another wrapper's adapters.
      _remove_lora_adapters(model)

  def _fit_adapters(self):
    """Trains adapters; always restores the shared base model afterward."""
    model = self.regressor.model
    if not HAS_TORCH or not isinstance(model, torch.nn.Module):
      raise ValueError("TTT is implemented for PyTorch models only.")
    was_training = model.training
    grad_state = _requires_grad_state(model)
    try:
      self._fit_adapters_impl()
    except Exception:
      # A failed fit must not leave metadata that lets predict silently fall
      # back to unadapted outputs.
      self._clear_fitted_state()
      raise
    finally:
      _remove_lora_adapters(model)
      _restore_requires_grad_state(model, grad_state)
      model.train(was_training)

  def _fit_adapters_impl(self):
    model = self.regressor.model
    _remove_lora_adapters(model)

    self.test_time_training_ = replace(self.config)
    config = self.test_time_training_
    self.test_time_training_flat_configs_ = self._flat_ttt_configs()
    self.test_time_training_max_features_ = max(
        len(cfg[0]) for _, cfg in self.test_time_training_flat_configs_
    )

    # Install targets before any destructive dtype change: a bad target must
    # fail without leaving the model partially cast.
    self.test_time_training_wrapped_layers_ = _ensure_lora_adapters(
        model, config, adapter_dtype=torch.float32
    )
    _freeze_non_lora_parameters(model)
    lora_params = _lora_named_parameters(model)

    # Zero steps is a strict no-op for the fitted base estimator. In
    # particular, do not cast its weights or mutate its chunk/checkpoint knobs.
    if config.steps:
      if config.cast_model_to_bfloat16:
        if getattr(next(model.parameters()).device, "type", None) != "cuda":
          raise ValueError(
              "cast_model_to_bfloat16=True requires a CUDA PyTorch model."
          )
        _cast_base_model_to_bfloat16(model)
      _configure_model_for_ttt(model, config)

    n_members = len(self.test_time_training_flat_configs_)
    self._warn_if_rows_undersampled()
    print(
        f"TTT: {len(self.test_time_training_wrapped_layers_)} LoRA layers"
        f" under {config.target_layers!r},"
        f" {sum(p.numel() for p in lora_params.values()):,} trainable params,"
        f" {config.steps} steps x batch {config.batch_size}"
        f" x accum {config.gradient_accumulation_steps}"
        f" ({config.batch_size * config.gradient_accumulation_steps}"
        " splits/update),"
        f" lr {config.learning_rate:g}, ema {config.ema_decay:g}",
        flush=True,
    )

    base_seed = (
        config.seed
        if config.seed is not None
        else self.regressor.random_state
        if self.regressor.random_state is not None
        else _DEFAULT_RANDOM_STATE
    )
    was_training = model.training
    model.train()
    states = []
    for member_idx in range(n_members):
      states.append(
          self._train_single_adapter(
              config, member_idx, int(base_seed) + member_idx, lora_params
          )
      )
      print(f"TTT member {member_idx + 1}/{n_members} done", flush=True)
    self.test_time_training_adapter_states_ = states

    # Adapter tensors live on the wrapper; the shared model goes back to its
    # base architecture between calls.
    _remove_lora_adapters(model)
    model.train(was_training)

  def _warn_if_rows_undersampled(self):
    """Warns when the row cap is too small for every row to be reachable.

    Each optimizer step draws ``batch_size`` splits and there are ``steps``
    of them (times ``gradient_accumulation_steps``), so the training run
    touches at most ``draws * max_train_rows`` rows in total. Below the
    training-set size, some rows can never enter any context.
    """
    config = self.test_time_training_
    cap = config.max_train_rows
    if not cap or not config.steps:
      return
    n_rows = len(self.regressor.ensemble_generator_.y_)
    draws = (
        config.steps * config.batch_size * config.gradient_accumulation_steps
    )
    reachable = draws * cap
    if reachable < n_rows:
      print(
          f"TTT warning: max_train_rows={cap} x {draws} draws reaches at most"
          f" {reachable} of {n_rows} training rows; raise max_train_rows to at"
          f" least {-(-n_rows // draws)} so every row can be sampled.",
          flush=True,
      )

  def _train_single_adapter(self, config, member_idx, seed, lora_params):
    """One member: sample -> microbatch -> accumulate -> step -> EMA."""
    model = self.regressor.model
    rng = np.random.default_rng(seed)
    train_pool = np.asarray(
        self._active_ttt_indices(member_idx), dtype=np.int64
    )
    _reset_lora_adapters(model, seed=seed)
    if train_pool.size <= 1:
      return _lora_state_dict(model)
    optimizer = _make_optimizer(config, list(lora_params.values()))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.steps),
            eta_min=config.min_learning_rate,
        )
        if config.learning_rate_schedule == "cosine"
        else None
    )
    ema = {
        n: torch.zeros_like(p, dtype=torch.float32)
        for n, p in lora_params.items()
    }
    n_updates = 0

    for step in range(config.steps):
      optimizer.zero_grad(set_to_none=True)
      detached_loss_sum = None
      for _ in range(config.gradient_accumulation_steps):
        splits = [
            self._sample_ttt_train_query(train_pool, rng, config)
            for _ in range(config.batch_size)
        ]
        loss = self._ttt_loss_for_member_batched(member_idx, splits)
        detached_loss_sum = (
            loss.detach()
            if detached_loss_sum is None
            else detached_loss_sum + loss.detach()
        )
        (loss / config.gradient_accumulation_steps).backward()
      # Synchronize once per optimizer update, not once per microbatch.
      if not bool(torch.isfinite(detached_loss_sum).item()):
        raise FloatingPointError(
            f"Non-finite TTT loss for member {member_idx}, step {step + 1}."
        )
      optimizer.step()
      if scheduler is not None:
        scheduler.step()
      n_updates += 1
      with torch.no_grad():
        for name, param in lora_params.items():
          ema[name].mul_(config.ema_decay).add_(
              param.detach().float(), alpha=1.0 - config.ema_decay
          )

    if n_updates:
      # Bias correction (as in Adam): without it the zero-initialized EMA
      # still carries decay**n_updates of its starting value, which shrinks
      # the deployed adapter -- badly so for a short step budget.
      scale = 1.0 / (1.0 - config.ema_decay**n_updates)
      _load_lora_state_dict(
          model, {n: (v * scale).cpu() for n, v in ema.items()}
      )
    return _lora_state_dict(model)

  # -- adapted forward / NNLS ----------------------------------------------------
  def _adapted_batch_forward(self, Xs, ys, cat_masks=None, ds=None):
    """Upstream _batch_forward's contract, but per member with its adapter."""
    model = self.regressor.model
    states = self.test_time_training_adapter_states_
    if len(states) != Xs.shape[0]:
      raise ValueError(
          f"TTT adapter states ({len(states)}) do not match the ensemble"
          f" ({Xs.shape[0]})."
      )
    was_training = model.training
    grad_state = _requires_grad_state(model)
    model.eval()
    outputs = []
    try:
      _ensure_lora_adapters(
          model, self.test_time_training_, adapter_dtype=torch.float32
      )
      for member_idx, state in enumerate(states):
        _load_lora_state_dict(model, state)
        X_b = Xs[member_idx : member_idx + 1]
        y_b = ys[member_idx : member_idx + 1]
        cat_b = (
            cat_masks[member_idx : member_idx + 1]
            if cat_masks is not None
            else None
        )
        ds_b = ds[member_idx : member_idx + 1] if ds is not None else None
        seq_len, train_size = X_b.shape[1], y_b.shape[1]
        if train_size < seq_len:
          y_b = np.pad(
              y_b, ((0, 0), (0, seq_len - train_size)), constant_values=-100.0
          )
        out = _predict_step_pytorch(model, X_b, y_b, train_size, ds_b, cat_b)
        outputs.append(out[:, train_size:seq_len, :])
    finally:
      _remove_lora_adapters(model)
      _restore_requires_grad_state(model, grad_state)
      model.train(was_training)
    return np.concatenate(outputs, axis=0)

  def _refit_nnls_weights(self, y):
    """Refits NNLS weights from out-of-fold predictions of the adapted members.

    Same procedure and candidate count the plain ensemble uses, so the only
    difference from it is that the members carry adapters -- the comparison
    stays apples to apples. Routes upstream's own fold/holdout machinery
    through the adapter-aware forward pass rather than reimplementing it.

    The out-of-fold predictions come from the deployed adapters, which have
    seen every training row, so they are mildly optimistic. Accepted
    deliberately: fold-local adapters would cost one adapter round per fold.
    """
    reg = self.regressor
    # Temporarily shadow the bound method; deleted in finally so nothing
    # unpicklable is left on the estimator.
    reg._batch_forward = self._adapted_batch_forward
    try:
      y_oof_scaled, val_idx = reg._compute_oof_preds_scaled(
          cv=reg.num_folds_for_cv
      )
    finally:
      del reg._batch_forward

    y_orig = np.asarray(y, dtype=float).ravel()
    if val_idx is not None:
      y_oof_scaled = y_oof_scaled[:, val_idx]
      y_orig = y_orig[val_idx]
    n_members = y_oof_scaled.shape[0]
    y_oof = np.stack(
        [reg._inverse_transform_y(row) for row in y_oof_scaled], axis=0
    )
    weights, _ = opt.nnls(y_oof.T, y_orig)
    total = float(np.sum(weights))
    weights = weights / total if total > 0 else np.ones(n_members) / n_members
    uniform = np.ones(n_members) / n_members
    return reg.nnls_beta * weights + (1.0 - reg.nnls_beta) * uniform

  def _predict_scaled_with_adapters(self, X):
    """Returns (n_members, n_test) scaled predictions with adapters loaded."""
    reg = self.regressor
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
      raise ValueError(
          "The provided input X is one-dimensional. Reshape your data."
      )
    X_transformed = reg.X_encoder_.transform(X)
    data = reg.ensemble_generator_.transform(X_transformed)
    Xs, ys, cat_masks, ds, _ = reg.ensemble_generator_.prepare_ensemble_tensors(
        data
    )
    output = self._adapted_batch_forward(Xs, ys, cat_masks, ds)
    _check_regressor_output_dim(output.shape[-1])
    return output.squeeze(-1)


def fit_and_predict(regressor, X_train, y_train, X_test, config=True):
  """Fits the regressor, trains TTT adapters, and predicts X_test.

  The regressor is fitted in place, so its own unadapted ``predict`` remains
  available for baseline comparisons.
  """
  return (
      TestTimeTrainedRegressor(regressor, config)
      .fit(X_train, y_train)
      .predict(X_test)
  )
