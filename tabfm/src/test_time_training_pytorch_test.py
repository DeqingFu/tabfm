# Copyright 2026 Google LLC
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

"""Tests for the test-time-training wrapper."""

import copy
import pickle
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from tabfm.src import test_time_training as ttt_lib
from tabfm.src.classifier_and_regressor import TabFMRegressor
from tabfm.src.pytorch import model as pytorch_model
from tabfm.src.test_time_training import (
    TabFMTestTimeTraining,
    TestTimeTrainedRegressor,
    _LoRALinear,
    fit_and_predict,
)


def _tiny_model():
  return pytorch_model.TabFM(
      embed_dim=8,
      max_classes=1,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=8,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=2,
      icl_num_blocks=1,
      icl_nhead=2,
      ff_factor=2,
      feature_group_size=2,
      is_classifier=False,
  )


def _cpu_config(**overrides):
  base = dict(lora_rank=2, steps=1, cast_model_to_bfloat16=False)
  base.update(overrides)
  return TabFMTestTimeTraining(**base)


class TestTimeTrainingPyTorchTest(unittest.TestCase):

  def test_lora_adapter_initializes_as_noop(self):
    base = torch.nn.Linear(4, 3, bias=False)
    adapter = _LoRALinear(base, rank=2)
    x = torch.randn(5, 4)
    self.assertTrue(torch.allclose(adapter(x), base(x)))
    self.assertEqual(int(torch.count_nonzero(adapter.lora_B)), 0)
    self.assertGreater(int(torch.count_nonzero(adapter.lora_A)), 0)
    self.assertIs(adapter.weight, base.weight)
    self.assertIs(adapter.bias, base.bias)

  def test_fit_predict_and_model_configuration(self):
    np.random.seed(42)
    model = _tiny_model()
    reg = TabFMRegressor(
        model=model, n_estimators=2, batch_size=1, random_state=42
    )
    X, y = np.random.rand(12, 3), np.random.rand(12)
    ttt = TestTimeTrainedRegressor(reg, _cpu_config()).fit(X, y)

    self.assertEqual(len(ttt.test_time_training_adapter_states_), 2)
    self.assertTrue(ttt.test_time_training_wrapped_layers_)
    # Chunk sizes default to the model's own values, and an explicit config
    # value overrides them for the shared model.
    self.assertEqual(model.cell_embedder.row_chunk_size, ttt_lib._ROW_CHUNK_SIZE)
    self.assertEqual(model.col_embedder.col_chunk_size, ttt_lib._COL_CHUNK_SIZE)
    self.assertEqual(
        model.row_interactor.row_chunk_size, ttt_lib._ROW_CHUNK_SIZE
    )
    TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=1, norm_methods=["none"]),
        _cpu_config(steps=1, row_interactor_chunk_size=64),
    ).fit(X, y)
    self.assertEqual(model.row_interactor.row_chunk_size, 64)
    # ...and a later None-valued config restores the model default rather
    # than silently keeping the previous run's value.
    TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=1, norm_methods=["none"]),
        _cpu_config(steps=1),
    ).fit(X, y)
    self.assertEqual(
        model.row_interactor.row_chunk_size, ttt_lib._ROW_CHUNK_SIZE
    )
    # Checkpointing is off by default and switched on by the config.
    self.assertFalse(model.icl_predictor.tf_icl.gradient_checkpointing)
    TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=1, norm_methods=["none"]),
        _cpu_config(steps=1, gradient_checkpointing="all"),
    ).fit(X, y)
    self.assertTrue(model.icl_predictor.tf_icl.gradient_checkpointing)
    self.assertEqual(ttt.predict(X[:3]).shape, (3,))
    # The wrapped regressor stays usable as an unadapted baseline.
    self.assertEqual(reg.predict(X[:3]).shape, (3,))

  def test_fit_and_predict_convenience(self):
    np.random.seed(42)
    reg = TabFMRegressor(
        model=_tiny_model(), n_estimators=2, norm_methods=["none"],
        random_state=42,
    )
    X, y = np.random.rand(12, 3), np.random.rand(12)
    preds = fit_and_predict(reg, X, y, X[:3], _cpu_config(steps=0))
    self.assertEqual(preds.shape, (3,))

  def test_rejects_context_cache(self):
    reg = TabFMRegressor(model=_tiny_model(), n_estimators=2,
                         cache_context=True)
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=0))
    with self.assertRaisesRegex(ValueError, "cache_context"):
      ttt.fit(np.random.rand(8, 3), np.random.rand(8))

  def test_config_validation_and_aliases(self):
    with self.assertRaises(ValueError):
      TabFMTestTimeTraining(batch_size=0)
    with self.assertRaises(ValueError):
      TabFMTestTimeTraining(gradient_accumulation_steps=0)
    with self.assertRaises(ValueError):
      TabFMTestTimeTraining(ema_decay=1.0)
    with self.assertRaises(ValueError):
      TabFMTestTimeTraining(steps=-1)
    with self.assertRaises(ValueError):
      TabFMTestTimeTraining(test_fraction=0.0)
    config = TabFMTestTimeTraining.from_value(
        {
            "lr": 1e-3,
            "batch_size": 2,
            "gradient_accumulation_steps": 3,
            "ema": 0.8,
            "target_layers": "cell",
        }
    )
    self.assertEqual(config.learning_rate, 1e-3)
    self.assertEqual(config.batch_size, 2)
    self.assertEqual(config.gradient_accumulation_steps, 3)
    self.assertEqual(config.ema_decay, 0.8)
    self.assertEqual(config.target_layers, ("cell_embedder.in_linear",))
    self.assertIsNone(TabFMTestTimeTraining.from_value(None))
    self.assertIsNone(TabFMTestTimeTraining.from_value(False))
    self.assertIsInstance(
        TabFMTestTimeTraining.from_value(True), TabFMTestTimeTraining
    )

  def test_batching_and_gradient_accumulation_use_independent_splits(self):
    np.random.seed(42)
    model = _tiny_model()
    reg = TabFMRegressor(
        model=model, n_estimators=1, norm_methods=["none"], random_state=42
    )
    X, y = np.random.rand(12, 3), np.random.rand(12)
    ttt = TestTimeTrainedRegressor(
        reg,
        _cpu_config(steps=2, batch_size=2, gradient_accumulation_steps=3,
                    learning_rate=1e-2, test_fraction=0.25),
    )
    calls = []

    def fake_loss(member_idx, splits):
      del member_idx
      calls.append((
          torch.is_grad_enabled(),
          tuple(
              (tuple(np.asarray(c).tolist()), tuple(np.asarray(q).tolist()))
              for c, q in splits
          ),
      ))
      lora_b = next(
          m.lora_B for m in model.modules() if isinstance(m, _LoRALinear)
      )
      return lora_b.sum()

    step_grads = []

    def record_step(unused_optimizer, closure=None):
      del unused_optimizer, closure
      lora_b = next(
          m.lora_B for m in model.modules() if isinstance(m, _LoRALinear)
      )
      step_grads.append(lora_b.grad.detach().clone())

    with mock.patch.object(
        ttt, "_ttt_loss_for_member_batched", side_effect=fake_loss
    ), mock.patch.object(
        torch.optim.AdamW, "step", autospec=True, side_effect=record_step
    ) as optimizer_step:
      ttt.fit(X, y)

    # Each optimizer step has three sequential microbatches, and every
    # microbatch forwards two independently sampled splits together.
    self.assertEqual(len(calls), 2 * 3)
    self.assertEqual(optimizer_step.call_count, 2)
    self.assertTrue(
        all(torch.equal(grad, torch.ones_like(grad)) for grad in step_grads)
    )
    self.assertTrue(all(grad for grad, _ in calls))
    self.assertTrue(all(len(splits) == 2 for _, splits in calls))
    self.assertGreater(len({s for _, splits in calls for s in splits}), 1)

  def test_ema_deploys_averaged_weights_and_zero_steps_is_noop(self):
    np.random.seed(42)
    torch.manual_seed(42)
    model = _tiny_model().eval()
    baseline_model = copy.deepcopy(model)
    X, y = np.random.rand(16, 4), np.random.rand(16)
    X_test = np.random.rand(3, 4)
    kwargs = dict(n_estimators=2, norm_methods=["none"], random_state=42)

    # steps=0 keeps the exact no-op adapters: predictions match the plain
    # regressor bit-for-bit on CPU fp32.
    baseline = TabFMRegressor(model=baseline_model, **kwargs).fit(X, y)
    ttt0 = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, **kwargs), _cpu_config(steps=0)
    ).fit(X, y)
    np.testing.assert_allclose(
        ttt0.predict(X_test), baseline.predict(X_test), rtol=1e-5, atol=1e-6
    )

    # With steps>0 the deployed state is the EMA: it moved off init but is
    # not the final step's weights.
    ttt = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, **kwargs),
        _cpu_config(steps=3, learning_rate=1e-2, ema_decay=0.5),
    )
    train_b = []
    orig_loss = TestTimeTrainedRegressor._ttt_loss_for_member_batched

    def recording_loss(self, member_idx, splits):
      lora_b = next(
          m.lora_B for m in model.modules() if isinstance(m, _LoRALinear)
      )
      train_b.append(lora_b.detach().clone())
      return orig_loss(self, member_idx, splits)

    with mock.patch.object(
        TestTimeTrainedRegressor, "_ttt_loss_for_member_batched", recording_loss
    ):
      ttt.fit(X, y)
    deployed_b = next(
        v for n, v in ttt.test_time_training_adapter_states_[0].items()
        if n.endswith("lora_B")
    )
    self.assertGreater(float(deployed_b.abs().sum()), 0.0)
    self.assertFalse(torch.allclose(deployed_b, train_b[-1].cpu(), atol=1e-9))

  def test_failure_leaves_model_clean_and_wrapper_unfitted(self):
    model = _tiny_model().eval()
    grad_state = {n: p.requires_grad for n, p in model.named_parameters()}
    reg = TabFMRegressor(model=model, n_estimators=2, norm_methods=["none"])
    ttt = TestTimeTrainedRegressor(reg, _cpu_config())
    with mock.patch.object(
        ttt, "_ttt_loss_for_member_batched", side_effect=RuntimeError("forced")
    ):
      with self.assertRaisesRegex(RuntimeError, "forced"):
        ttt.fit(np.random.rand(8, 3), np.random.rand(8))

    self.assertFalse(model.training)
    self.assertFalse(
        any(isinstance(m, _LoRALinear) for m in model.modules())
    )
    self.assertEqual(
        {n: p.requires_grad for n, p in model.named_parameters()}, grad_state
    )
    self.assertFalse(hasattr(ttt, "test_time_training_adapter_states_"))
    with self.assertRaises(RuntimeError):
      ttt.predict(np.random.rand(2, 3))
    # The regressor was fitted before adapters failed: still a valid baseline.
    self.assertEqual(reg.predict(np.random.rand(2, 3)).shape, (2,))

  def test_invalid_target_does_not_mutate_base_model(self):
    np.random.seed(42)
    model = _tiny_model().eval()
    base_state = {
        n: v.detach().clone() for n, v in model.state_dict().items()
    }
    ttt = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=1, norm_methods=["none"]),
        TabFMTestTimeTraining(
            target_layers=("missing.layer",), steps=0,
            cast_model_to_bfloat16=True,
        ),
    )
    with self.assertRaisesRegex(ValueError, "Unknown TTT target layer"):
      ttt.fit(np.random.rand(8, 3), np.random.rand(8))
    self.assertEqual(set(model.state_dict()), set(base_state))
    for name, value in model.state_dict().items():
      self.assertEqual(value.dtype, base_state[name].dtype)
      torch.testing.assert_close(value, base_state[name])

  def test_adapter_state_loading_is_strict_and_atomic(self):
    np.random.seed(42)
    model = _tiny_model().eval()
    reg = TabFMRegressor(
        model=model, n_estimators=2, norm_methods=["none"], batch_size=1,
        random_state=42,
    )
    X, y = np.random.rand(10, 3), np.random.rand(10)
    # steps>0 is required: zero-step predict delegates straight to the plain
    # regressor, so it would never load an adapter state at all.
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=1)).fit(X, y)
    states = [
        {n: v.clone() for n, v in s.items()}
        for s in ttt.test_time_training_adapter_states_
    ]
    base_weight = model.cell_embedder.in_linear.weight
    base_before = base_weight.detach().clone()

    missing = dict(states[1])
    missing.pop(next(iter(missing)))
    ttt.test_time_training_adapter_states_ = [states[0], missing]
    with self.assertRaisesRegex(ValueError, "missing="):
      ttt.predict(X[:2])

    poisoned = dict(states[0])
    poisoned["cell_embedder.in_linear.base.weight"] = torch.zeros_like(
        base_before
    )
    ttt.test_time_training_adapter_states_ = [poisoned, states[1]]
    with self.assertRaisesRegex(ValueError, "unexpected="):
      ttt.predict(X[:2])

    torch.testing.assert_close(base_weight, base_before)
    self.assertFalse(
        any(isinstance(m, _LoRALinear) for m in model.modules())
    )

  def test_predict_loads_member_adapter_states(self):
    np.random.seed(42)
    reg = TabFMRegressor(
        model=_tiny_model(), n_estimators=2, batch_size=1, random_state=42
    )
    X, y = np.random.rand(12, 3), np.random.rand(12)
    ttt = TestTimeTrainedRegressor(
        reg, _cpu_config(learning_rate=1e-2, test_fraction=0.25)
    ).fit(X, y)
    baseline = ttt.predict(X[:3])

    state0 = {
        n: v.clone()
        for n, v in ttt.test_time_training_adapter_states_[0].items()
    }
    state1 = {
        n: v.clone()
        for n, v in ttt.test_time_training_adapter_states_[1].items()
    }
    b_key = next(n for n in state0 if n.endswith("lora_B"))
    state1[b_key].add_(5.0)
    ttt.test_time_training_adapter_states_ = [state0, state1]
    perturbed = ttt.predict(X[:3])
    ttt.test_time_training_adapter_states_ = [state0, state0]
    repeated = ttt.predict(X[:3])

    self.assertFalse(np.allclose(baseline, perturbed))
    self.assertFalse(np.allclose(repeated, perturbed))

  def test_ttt_trains_on_raw_features_infers_on_engineered_views(self):
    np.random.seed(42)
    reg = TabFMRegressor.ensemble(
        model=_tiny_model(), n_estimators=2, norm_methods=["none"],
        num_folds_for_cv=2, batch_size=1, random_state=42,
    )
    ttt = TestTimeTrainedRegressor(
        reg,
        _cpu_config(
            steps=1, learning_rate=1e-2, test_fraction=0.25,
            gradient_checkpointing="none",
        ),
    )
    X, y = np.random.rand(12, 3), np.random.rand(12)
    widths = {"train": [], "infer": []}
    orig_forward = ttt_lib._forward_tensor
    orig_predict = ttt_lib._predict_step_pytorch

    def rec_forward(model_arg, Xb, yb, ts, ds, cm):
      widths["train"].append(int(np.asarray(ds)[0]))
      return orig_forward(model_arg, Xb, yb, ts, ds, cm)

    def rec_predict(model_arg, Xb, yb, ts, ds, cm):
      widths["infer"].append(int(np.asarray(ds)[0]))
      return orig_predict(model_arg, Xb, yb, ts, ds, cm)

    with (
        mock.patch.object(ttt_lib, "_forward_tensor", rec_forward),
        mock.patch.object(ttt_lib, "_predict_step_pytorch", rec_predict),
    ):
      ttt.fit(X, y)
      # The out-of-fold passes that refit NNLS already run adapted inference
      # on the full engineered views; keep them separate from predict()'s.
      refit_widths = list(widths["infer"])
      self.assertTrue(refit_widths)
      widths["infer"].clear()
      preds = ttt.predict(X[:3])

    n_original = reg.ensemble_generator_.n_original_features_
    self.assertGreater(reg.ensemble_generator_.n_features_in_, n_original)
    # Adapter training sees only original features...
    self.assertTrue(all(w <= n_original for w in widths["train"]))
    # ...while adapted inference runs each member's full engineered view.
    engineered = [len(cfg[0]) for _, cfg in ttt._flat_ensemble_configs()]
    self.assertEqual(widths["infer"], engineered)
    self.assertTrue(any(w > n_original for w in widths["infer"]))
    self.assertEqual(preds.shape, (3,))

  def test_member_views_preserve_identity(self):
    rng = np.random.default_rng(123)
    n_rows = 30
    X = pd.DataFrame({
        "cat": np.resize(np.array(["a", "b", "c"], dtype=object), n_rows),
        **{f"x{i}": rng.normal(size=n_rows) for i in range(5)},
    })
    y = rng.normal(size=n_rows)
    reg = TabFMRegressor.ensemble(
        model=_tiny_model(), n_estimators=8, norm_methods=["none", "power"],
        enable_nnls=False, permute_categorical=True, max_num_features=4,
        max_num_rows=12, random_state=17,
    )
    ttt = TestTimeTrainedRegressor(
        reg, _cpu_config(steps=0, gradient_checkpointing="none")
    ).fit(X, y)

    generator = reg.ensemble_generator_
    n_original = generator.n_original_features_
    full = ttt._flat_ensemble_configs()
    projected = ttt.test_time_training_flat_configs_
    self.assertEqual(len(projected), reg.n_estimators)
    self.assertTrue(any(np.any(np.asarray(c[0]) >= n_original) for _, c in full))
    for (p_norm, p_cfg), (f_norm, f_cfg) in zip(projected, full):
      self.assertEqual(p_norm, f_norm)
      f_pattern = np.asarray(f_cfg[0])
      np.testing.assert_array_equal(
          p_cfg[0], f_pattern[f_pattern < n_original]
      )
      self.assertEqual(p_cfg[1], f_cfg[1])
      self.assertEqual(p_cfg[2], f_cfg[2])

    member_idx = 0
    context_idx = np.array([0, 1, 2], dtype=np.int64)
    query_idx = np.array([3, 4], dtype=np.int64)
    X_b, _, _, _, d = ttt._member_ttt_arrays(member_idx, context_idx, query_idx)
    norm, (pattern, _, cat_perm, _) = projected[member_idx]
    expected = generator.X_[np.concatenate([context_idx, query_idx])].copy()
    ttt_lib._apply_categorical_permutation(expected, cat_perm)
    expected = generator.preprocessors_[norm].transform(expected)[:, pattern]
    expected = ttt_lib._pad_features(
        expected, ttt.test_time_training_max_features_
    )
    np.testing.assert_allclose(X_b[0], expected)
    self.assertEqual(int(d[0]), len(pattern))

  def test_seed_controls_adapters_and_rng_does_not_leak(self):
    model_a = _tiny_model()
    model_b = pickle.loads(pickle.dumps(model_a))
    X, y = np.random.rand(8, 3), np.random.rand(8)
    config = _cpu_config(steps=0, seed=77)

    torch.manual_seed(1)
    rng_before = torch.get_rng_state().clone()
    ttt_a = TestTimeTrainedRegressor(
        TabFMRegressor(model=model_a, n_estimators=2, norm_methods=["none"]),
        config,
    ).fit(X, y)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)

    torch.manual_seed(2)
    ttt_b = TestTimeTrainedRegressor(
        TabFMRegressor(model=model_b, n_estimators=2, norm_methods=["none"]),
        config,
    ).fit(X, y)
    for sa, sb in zip(
        ttt_a.test_time_training_adapter_states_,
        ttt_b.test_time_training_adapter_states_,
    ):
      self.assertEqual(set(sa), set(sb))
      for name in sa:
        torch.testing.assert_close(sa[name], sb[name])

    rng_before = torch.get_rng_state().clone()
    ttt_a.predict(X[:2])
    torch.testing.assert_close(torch.get_rng_state(), rng_before)

  def test_caller_config_snapshot(self):
    np.random.seed(42)
    config = _cpu_config(steps=0)
    ttt = TestTimeTrainedRegressor(
        TabFMRegressor(model=_tiny_model(), n_estimators=1,
                       norm_methods=["none"]),
        config,
    ).fit(np.random.rand(8, 3), np.random.rand(8))
    X_test = np.random.rand(2, 3)
    baseline = ttt.predict(X_test)
    self.assertIsNot(ttt.test_time_training_, config)
    config.lora_rank = 7
    np.testing.assert_allclose(ttt.predict(X_test), baseline)
    self.assertEqual(ttt.test_time_training_.lora_rank, 2)

  def test_fp32_adapters_over_bfloat16_base(self):
    np.random.seed(42)
    model = _tiny_model().to(torch.bfloat16)
    ttt = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=1, norm_methods=["none"],
                       random_state=42),
        _cpu_config(steps=0),
    ).fit(np.random.rand(8, 3), np.random.rand(8))
    self.assertEqual(
        model.cell_embedder.in_linear.weight.dtype, torch.bfloat16
    )
    self.assertTrue(
        all(
            v.dtype == torch.float32
            for s in ttt.test_time_training_adapter_states_
            for v in s.values()
        )
    )

  def test_reused_model_stays_clean_across_wrappers(self):
    model = _tiny_model()
    X, y = np.random.rand(8, 3), np.random.rand(8)
    grad_state = {n: p.requires_grad for n, p in model.named_parameters()}

    first = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=2, norm_methods=["none"]),
        _cpu_config(steps=0),
    ).fit(X, y)
    self.assertFalse(any(isinstance(m, _LoRALinear) for m in model.modules()))
    self.assertEqual(
        {n: p.requires_grad for n, p in model.named_parameters()}, grad_state
    )

    plain = TabFMRegressor(model=model, n_estimators=2, norm_methods=["none"])
    plain.fit(X, y)
    self.assertEqual(plain.predict(X[:2]).shape, (2,))

    first_preds = first.predict(X[:2])
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_allclose(
        restored.predict(X[:2]), first_preds, rtol=1e-5, atol=1e-6
    )

    second = TestTimeTrainedRegressor(
        TabFMRegressor(model=model, n_estimators=2, norm_methods=["none"]),
        _cpu_config(lora_rank=3, steps=0),
    ).fit(X, y)
    self.assertFalse(any(isinstance(m, _LoRALinear) for m in model.modules()))
    a_key = next(
        n for n in second.test_time_training_adapter_states_[0]
        if n.endswith("lora_A")
    )
    self.assertEqual(
        second.test_time_training_adapter_states_[0][a_key].shape[0], 3
    )


  def test_zero_steps_is_exactly_the_plain_ensemble(self):
    """steps=0 must reproduce the plain ensemble exactly, NNLS included."""
    np.random.seed(42)
    X = np.random.rand(20, 4)
    y = np.linspace(0.0, 1.0, len(X))
    X_test = np.random.rand(5, 4)
    model = _tiny_model()
    kwargs = dict(
        n_estimators=3, norm_methods=["none"], num_folds_for_cv=2,
        random_state=42,
    )
    baseline = TabFMRegressor.ensemble(
        model=copy.deepcopy(model), **kwargs
    ).fit(X, y)
    reg = TabFMRegressor.ensemble(model=model, **kwargs)
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=0)).fit(X, y)

    np.testing.assert_allclose(
        reg.ensemble_weights_, baseline.ensemble_weights_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        reg.y_oof_scaled_, baseline.y_oof_scaled_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        ttt.predict(X_test), baseline.predict(X_test), rtol=0.0, atol=0.0
    )

  def test_nnls_weights_are_refit_on_adapted_members(self):
    """TTT refits ensemble weights, but leaves the regressor's own untouched."""
    np.random.seed(42)
    X = np.random.rand(16, 4)
    y = np.linspace(0.0, 1.0, len(X))
    model = _tiny_model()
    kwargs = dict(
        n_estimators=2, norm_methods=["none"], num_folds_for_cv=2,
        random_state=42,
    )
    baseline = TabFMRegressor.ensemble(
        model=copy.deepcopy(model), **kwargs
    ).fit(X, y)

    reg = TabFMRegressor.ensemble(model=model, **kwargs)
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=1)).fit(X, y)

    # The regressor keeps the weights its own unadapted fit produced, so
    # reg.predict() remains a clean baseline...
    np.testing.assert_allclose(
        reg.ensemble_weights_, baseline.ensemble_weights_, rtol=1e-6
    )
    self.assertNotIn("_batch_forward", reg.__dict__)
    # ...while the wrapper carries its own weights, refit over every member's
    # adapted and unadapted variant plus the raw-feature sibling's members.
    # Both the plain ensemble and the raw-feature ensemble stay reachable, so
    # weighting can fall back to either.
    # One weight per member -- the same candidate count the plain ensemble
    # uses, so the only difference from it is the adapters themselves.
    w = ttt.test_time_training_ensemble_weights_
    self.assertEqual(w.shape, (2,))
    self.assertAlmostEqual(float(w.sum()), 1.0)
    self.assertTrue(np.all(w >= 0))
    expected = np.dot(
        w,
        np.stack([
            reg._inverse_transform_y(row)
            for row in ttt._predict_scaled_with_adapters(X[:3])
        ]),
    )
    np.testing.assert_allclose(ttt.predict(X[:3]), expected, rtol=1e-6)

  def test_row_subsampling_uses_the_holdout_path(self):
    """Capped context (max_num_rows) must work with the adapted NNLS refit.

    Upstream switches from cross-validation to a reserved holdout when rows
    are subsampled, and the refit routes through that same machinery. Large
    datasets rely on this path, so cover it here rather than discovering it
    hours into a benchmark run.
    """
    rng = np.random.default_rng(0)
    n = 6000
    X, y = rng.random((n, 4)), rng.random(n)
    reg = TabFMRegressor.ensemble(
        model=_tiny_model(), n_estimators=2, batch_size=2, random_state=42,
        max_num_rows=5000,
    )
    ttt = TestTimeTrainedRegressor(
        reg, _cpu_config(steps=1, batch_size=1)
    ).fit(X, y)

    self.assertIsNotNone(reg.ensemble_generator_.holdout_indices)
    for member_idx in range(2):
      self.assertEqual(len(ttt._active_ttt_indices(member_idx)), 5000)
    w = ttt.test_time_training_ensemble_weights_
    self.assertEqual(w.shape, (2,))
    self.assertAlmostEqual(float(w.sum()), 1.0)
    self.assertEqual(ttt.predict(X[:3]).shape, (3,))

  def test_nnls_refit_is_skipped_at_zero_steps(self):
    np.random.seed(42)
    X = np.random.rand(16, 4)
    y = np.linspace(0.0, 1.0, len(X))
    reg = TabFMRegressor.ensemble(
        model=_tiny_model(), n_estimators=2, norm_methods=["none"],
        num_folds_for_cv=2, random_state=42,
    )
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=0)).fit(X, y)
    self.assertFalse(hasattr(ttt, "test_time_training_ensemble_weights_"))

  def test_ttt_view_equals_inference_view_on_original_features(self):
    """TTT must see exactly the inference data, minus the engineered columns."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({
        "cat": np.resize(np.array(["a", "b", "c"], dtype=object), n),
        **{f"x{i}": rng.normal(size=n) for i in range(5)},
    })
    y = rng.normal(size=n)
    reg = TabFMRegressor.ensemble(
        model=_tiny_model(), n_estimators=6, norm_methods=["none", "power"],
        permute_categorical=True, random_state=7, enable_nnls=False,
    )
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=0)).fit(X, y)

    # The upstream inference path, untouched.
    data = reg.ensemble_generator_.transform(reg.X_encoder_.transform(X))
    Xs, _, cat_masks, _, _ = (
        reg.ensemble_generator_.prepare_ensemble_tensors(data)
    )
    n_orig = reg.ensemble_generator_.n_original_features_
    context_idx = np.array([3, 11, 25, 7], dtype=np.int64)
    query_idx = np.array([1, 19], dtype=np.int64)

    for member_idx, (_, cfg) in enumerate(ttt._flat_ensemble_configs()):
      pattern = np.asarray(cfg[0])
      keep = np.where(pattern < n_orig)[0]
      X_ttt, y_ttt, _, cat_ttt, d_ttt = ttt._member_ttt_arrays(
          member_idx, context_idx, query_idx
      )
      width = int(d_ttt[0])
      self.assertEqual(width, len(keep))
      for j, row in enumerate(context_idx):
        np.testing.assert_allclose(
            X_ttt[0, j, :width], Xs[member_idx, row, keep], atol=1e-6
        )
      np.testing.assert_array_equal(
          cat_ttt[0, :width], cat_masks[member_idx, keep]
      )
      np.testing.assert_allclose(
          y_ttt[0, : len(context_idx)],
          reg.ensemble_generator_.y_[context_idx],
      )

  def test_nnls_refit_is_skipped_without_nnls(self):
    np.random.seed(42)
    reg = TabFMRegressor(
        model=_tiny_model(), n_estimators=2, norm_methods=["none"],
        random_state=42,
    )
    self.assertFalse(reg.enable_nnls)
    ttt = TestTimeTrainedRegressor(reg, _cpu_config(steps=0))
    ttt.fit(np.random.rand(12, 3), np.random.rand(12))
    self.assertFalse(hasattr(reg, "ensemble_weights_"))
    self.assertEqual(ttt.predict(np.random.rand(3, 3)).shape, (3,))


if __name__ == "__main__":
  unittest.main()
