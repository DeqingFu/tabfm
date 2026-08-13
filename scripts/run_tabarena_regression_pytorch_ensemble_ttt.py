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

"""Runs PyTorch TabFM and TTT variants on TabArena regressions.

The user-facing selector is dataset-centric: pass TabArena regression dataset
names or OpenML dataset ids with --dataset. The script resolves the matching
TabArena split definition from the configured OpenML suite, runs repeat-0 /
fold-0 by default, and writes incremental CSV results under ttt_results/.
Ensemble TTT keeps the plain regressor's NNLS weights: they are fitted from
its own leakage-free out-of-fold predictions and are not refit after adaptation.
This
requires one adapter-training round per NNLS fold plus the final round.

Example:
  PYTHONPATH=. python scripts/run_tabarena_regression_pytorch_ensemble_ttt.py
  PYTHONPATH=. python scripts/run_tabarena_regression_pytorch_ensemble_ttt.py \
      --dataset airfoil_self_noise
  PYTHONPATH=. python scripts/run_tabarena_regression_pytorch_ensemble_ttt.py \
      --dataset 46904 --method ensemble_ttt
  PYTHONPATH=. python scripts/run_tabarena_regression_pytorch_ensemble_ttt.py \
      --dataset 46904 --method default_ttt
"""

from __future__ import annotations

import argparse
import hashlib
import csv
from dataclasses import dataclass
import gc
from contextlib import contextmanager
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import tabfm


DEFAULT_TABARENA_OPENML_SUITE = "tabarena-v0.1"

TABARENA_REGRESSION_DATASETS: tuple[str, ...] = (
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "concrete_compressive_strength",
    "diamonds",
    "Food_Delivery_Time",
    "healthcare_insurance_expenses",
    "houses",
    "miami_housing",
    "physiochemical_protein",
    "QSAR-TID-11",
    "QSAR_fish_toxicity",
    "superconductivity",
    "wine_quality",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "status",
    "dataset_name",
    "dataset_id",
    "method",
    "openml_suite",
    "repeat",
    "fold",
    "n_train",
    "n_test",
    "n_features",
    "n_estimators",
    "ensemble_enable_nnls",
    "batch_size",
    "rmse",
    "mae",
    "r2",
    "fit_seconds",
    "predict_seconds",
    "total_seconds",
    "ttt_steps",
    "ttt_batch_size",
    "ttt_gradient_accumulation_steps",
    "ttt_ema_decay",
    "ttt_learning_rate",
    "ttt_lora_rank",
    "ttt_target_layers",
    "ttt_cell_embedder_row_chunk_size",
    "ttt_col_embedder_chunk_size",
    "ttt_row_interactor_chunk_size",
    "ttt_ffn_chunk_size",
    "ttt_gradient_checkpointing",
    "ttt_cast_model_to_bfloat16",
    "ttt_max_train_rows",
    "ttt_max_test_rows",
    "error",
)


@dataclass(frozen=True)
class DatasetSpec:
  name: str
  dataset_id: int
  task_id: int


class _TeeStream:

  def __init__(self, *streams):
    self._streams = streams

  def write(self, data):
    for stream in self._streams:
      stream.write(data)
    return len(data)

  def flush(self):
    for stream in self._streams:
      stream.flush()

  def isatty(self):
    return any(
        getattr(stream, "isatty", lambda: False)() for stream in self._streams
    )

  def fileno(self):
    return self._streams[0].fileno()


@contextmanager
def _tee_output(log_file: Path | None):
  if log_file is None:
    yield
    return

  log_file.parent.mkdir(parents=True, exist_ok=True)
  stdout = sys.stdout
  stderr = sys.stderr
  with log_file.open("a", encoding="utf-8", buffering=1) as handle:
    sys.stdout = _TeeStream(stdout, handle)
    sys.stderr = _TeeStream(stderr, handle)
    try:
      yield
    finally:
      sys.stdout = stdout
      sys.stderr = stderr


def _log(message: str) -> None:
  print(f"[tabarena] {message}", flush=True)


def _optional_positive_int(value: str) -> int | None:
  normalized = value.strip().lower()
  if normalized in ("none", "null", "all", "unlimited"):
    return None
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"expected a positive integer or one of none/all; got {value!r}"
    ) from exc
  if parsed <= 0:
    raise argparse.ArgumentTypeError(
        f"expected a positive integer or none/all; got {value!r}"
    )
  return parsed


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument(
      "--dataset",
      action="append",
      default=[],
      help=(
          "TabArena regression dataset name or OpenML dataset id. May be"
          " repeated. Defaults to all TabArena regression datasets."
      ),
  )
  parser.add_argument(
      "--method",
      action="append",
      choices=("default", "default_ttt", "ensemble", "ensemble_ttt"),
      default=[],
      help=(
          "Method to run. May be repeated. Defaults to ensemble and"
          " ensemble_ttt."
      ),
  )
  parser.add_argument(
      "--results-dir",
      type=Path,
      default=Path("ttt_results"),
  )
  parser.add_argument(
      "--log-file",
      action="store_true",
      help=(
          "Write stdout/stderr to a timestamped log file under results-dir/"
          "logs/ and tee output to it."
      ),
  )
  parser.add_argument(
      "--openml-cache-dir",
      type=Path,
      default=Path.home() / ".cache" / "openml",
  )
  parser.add_argument("--openml-suite", default=DEFAULT_TABARENA_OPENML_SUITE)
  parser.add_argument("--repeat", type=int, default=0)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--n-estimators", type=int, default=32)
  parser.add_argument(
      "--no-ensemble-nnls",
      action="store_true",
      help=(
          "For ensemble_ttt only, use uniform averaging instead of NNLS "
          "weighting. The "
          "ensemble baseline keeps standard NNLS."
      ),
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--no-amp", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--fail-fast", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--no-save-predictions", action="store_true")

  parser.add_argument("--ttt-lora-rank", type=int, default=8)
  parser.add_argument(
      "--ttt-target-layers",
      default="cell_embedder.in_linear",
      help=(
          "Comma-separated TTT target layers. Defaults to"
          " cell_embedder.in_linear."
      ),
  )
  parser.add_argument("--ttt-steps", type=int, default=8)
  parser.add_argument(
      "--ttt-batch-size",
      type=int,
      default=4,
      help=(
          "Independent context/query splits forwarded together per "
          "microbatch. Larger values improve GPU utilization at proportional "
          "activation memory."
      ),
  )
  parser.add_argument(
      "--ttt-gradient-accumulation-steps",
      type=int,
      default=1,
      help=(
          "Sequential TTT microbatches accumulated before each optimizer "
          "step. Increase this, while reducing --ttt-batch-size, to preserve "
          "the effective split batch on datasets that do not fit in memory."
      ),
  )
  parser.add_argument("--ttt-ema-decay", type=float, default=0.9)
  parser.add_argument("--ttt-learning-rate", type=float, default=1e-3)
  parser.add_argument("--ttt-test-fraction", type=float, default=0.2)
  parser.add_argument("--ttt-cell-row-chunk-size", type=int, default=128)
  parser.add_argument("--ttt-col-chunk-size", type=int, default=4)
  parser.add_argument("--ttt-row-chunk-size", type=int, default=64)
  parser.add_argument("--ttt-ffn-chunk-size", type=int, default=4096)
  checkpointing_group = parser.add_mutually_exclusive_group()
  checkpointing_group.add_argument(
      "--ttt-gradient-checkpointing",
      choices=("none", "icl", "all"),
      default="none",
      help=(
          "Gradient checkpointing mode for TTT. none is fastest; icl "
          "checkpoints only the ICL transformer for a speed/memory balance; "
          "all saves the most memory. Defaults to none."
      ),
  )
  checkpointing_group.add_argument(
      "--ttt-checkpointing",
      dest="ttt_gradient_checkpointing",
      action="store_const",
      const="all",
      help=(
          "Deprecated compatibility alias for --ttt-gradient-checkpointing all."
      ),
  )
  parser.add_argument(
      "--no-ttt-bfloat16-model",
      action="store_true",
      help=(
          "Keep the frozen base model weights in fp32 during TTT. By default "
          "TTT casts memory-heavy base parameters to bf16 while keeping LoRA "
          "parameters and optimizer state in fp32."
      ),
  )
  parser.add_argument(
      "--ttt-use-all-rows",
      action="store_true",
      help=(
          "Disable TTT row caps. Equivalent to setting all TTT max-row "
          "arguments to none. Row caps are already disabled by default; this "
          "flag overrides any explicit max-row arguments."
      ),
  )
  # Row caps default to None (no cap): capping the TTT context (e.g. at 512
  # rows) trains the adapters against a smaller-context model
  # than the one deployed at prediction time, where each member sees the full
  # training set in context. That regime mismatch was measured to erase or
  # invert the adapters' validation gains at test time. Use the caps only to
  # avoid out-of-memory on very large training sets.
  parser.add_argument(
      "--ttt-max-train-rows", type=_optional_positive_int, default=None
  )
  parser.add_argument(
      "--ttt-max-test-rows", type=_optional_positive_int, default=None
  )
  return parser.parse_args()


def _normalize_name(name: str) -> str:
  return name.lower().replace("_", "-")


def _safe_name(name: str) -> str:
  keep = []
  for ch in name.lower():
    keep.append(ch if ch.isalnum() else "_")
  return "".join(keep).strip("_")


def _canonical_ttt_target_layers(value) -> tuple[str, ...]:
  config = tabfm.TabFMTestTimeTraining.from_value({"target_layers": value})
  return (
      config.target_layers
      if config is not None
      else (tabfm.TabFMTestTimeTraining.target_layers)
  )


def _is_ttt_method(method: str) -> bool:
  return method.endswith("_ttt")


def _effective_n_estimators(method: str, args: argparse.Namespace) -> int:
  return args.n_estimators


def _ensemble_enable_nnls(method: str, args: argparse.Namespace) -> bool | None:
  if not method.startswith("ensemble"):
    return None
  if method == "ensemble_ttt" and args.no_ensemble_nnls:
    return False
  return True


def _ttt_row_cap(args: argparse.Namespace, name: str) -> int | None:
  if args.ttt_use_all_rows:
    return None
  return getattr(args, name)


def _ttt_gradient_checkpointing_mode(args: argparse.Namespace) -> str:
  return args.ttt_gradient_checkpointing


def _format_ttt_row_cap(value: int | None) -> str | int:
  return "all" if value is None else value


def _run_slug(
    methods: Sequence[str],
    datasets: Sequence[DatasetSpec],
    args: argparse.Namespace,
) -> str:
  signature = {
      "methods": list(methods),
      "datasets": [spec.dataset_id for spec in datasets],
      "repeat": args.repeat,
      "fold": args.fold,
      "n_estimators": {
          method: _effective_n_estimators(method, args) for method in methods
      },
      "batch_size": args.batch_size,
      "seed": args.seed,
      "no_amp": args.no_amp,
      "ensemble_enable_nnls": {
          method: _ensemble_enable_nnls(method, args) for method in methods
      },
      "openml_suite": args.openml_suite,
      "ttt": {
          "lora_rank": args.ttt_lora_rank,
          "target_layers": list(
              _canonical_ttt_target_layers(args.ttt_target_layers)
          ),
          "steps": args.ttt_steps,
          "batch_size": args.ttt_batch_size,
          "gradient_accumulation_steps": (
              args.ttt_gradient_accumulation_steps
          ),
          "ema_decay": args.ttt_ema_decay,
          "learning_rate": args.ttt_learning_rate,
          "test_fraction": args.ttt_test_fraction,
          "cell_embedder_row_chunk_size": args.ttt_cell_row_chunk_size,
          "col_embedder_chunk_size": args.ttt_col_chunk_size,
          "row_interactor_chunk_size": args.ttt_row_chunk_size,
          "ffn_chunk_size": args.ttt_ffn_chunk_size,
          "gradient_checkpointing": _ttt_gradient_checkpointing_mode(args),
          "cast_model_to_bfloat16": not args.no_ttt_bfloat16_model,
          "max_train_rows": _ttt_row_cap(args, "ttt_max_train_rows"),
          "max_test_rows": _ttt_row_cap(args, "ttt_max_test_rows"),
      },
  }
  payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
  digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
  method_part = "-".join(methods)
  return f"{method_part}_r{args.repeat}f{args.fold}_{digest}"


def _is_regression_task(task) -> bool:
  return "regression" in str(getattr(task, "task_type", "")).lower()


def _configure_openml_cache(cache_dir: Path | None, log: bool = True) -> None:
  if cache_dir is None:
    return
  import openml  # pylint: disable=g-import-not-at-top

  cache_dir.mkdir(parents=True, exist_ok=True)
  openml.config.set_root_cache_directory(str(cache_dir))
  if log:
    _log(f"OpenML cache set to {cache_dir}")


def _resolve_datasets(
    requested_datasets: Sequence[str],
    suite_name: str,
) -> tuple[DatasetSpec, ...]:
  """Resolves dataset names/ids to TabArena split tasks."""
  import openml  # pylint: disable=g-import-not-at-top

  requested_dataset_ids = {
      int(item) for item in requested_datasets if item.isdigit()
  }
  requested_names = {
      _normalize_name(item) for item in requested_datasets if not item.isdigit()
  }

  canonical_names = {
      _normalize_name(name): name for name in TABARENA_REGRESSION_DATASETS
  }
  _log(
      "Resolving TabArena regression datasets from OpenML suite "
      f"{suite_name!r}..."
  )
  if not requested_datasets:
    target_names = set(canonical_names)
  elif requested_names:
    unknown = sorted(set(requested_names) - set(canonical_names))
    if unknown:
      raise ValueError(
          "Unknown TabArena regression dataset(s): " + ", ".join(unknown)
      )
    target_names = requested_names
  else:
    target_names = set()

  resolved_by_name: dict[str, DatasetSpec] = {}
  resolved_by_dataset_id: dict[int, DatasetSpec] = {}
  suite = openml.study.get_suite(suite_name)
  _log(
      f"OpenML suite returned {len(suite.tasks)} tasks; matching regression "
      "datasets..."
  )
  for task_id in suite.tasks:
    if len(resolved_by_name) == len(
        target_names
    ) and requested_dataset_ids <= set(resolved_by_dataset_id):
      break

    task = openml.tasks.get_task(task_id)
    if not _is_regression_task(task):
      continue
    dataset = task.get_dataset()
    dataset_key = _normalize_name(dataset.name)
    if dataset_key not in canonical_names:
      continue

    resolved = DatasetSpec(
        name=canonical_names[dataset_key],
        dataset_id=int(dataset.id),
        task_id=int(task.id),
    )
    resolved_by_dataset_id[resolved.dataset_id] = resolved
    if dataset_key in target_names and dataset_key not in resolved_by_name:
      resolved_by_name[dataset_key] = resolved

  missing = sorted(target_names - set(resolved_by_name))
  if missing:
    raise ValueError(
        "Could not resolve TabArena regression dataset(s) from OpenML suite "
        f"{suite_name!r}: {', '.join(missing)}"
    )

  selected: list[DatasetSpec] = []
  for name in TABARENA_REGRESSION_DATASETS:
    name_key = _normalize_name(name)
    if name_key in target_names:
      selected.append(resolved_by_name[name_key])

  for dataset_id in sorted(requested_dataset_ids):
    if dataset_id not in resolved_by_dataset_id:
      raise ValueError(
          f"OpenML dataset id {dataset_id} is not a TabArena regression"
          f" dataset in suite {suite_name!r}."
      )
    selected.append(resolved_by_dataset_id[dataset_id])

  deduped = []
  seen = set()
  for spec in selected:
    if spec.dataset_id not in seen:
      deduped.append(spec)
      seen.add(spec.dataset_id)
  _log(
      "Resolved datasets: "
      + ", ".join(
          f"{spec.name} (dataset_id={spec.dataset_id})" for spec in deduped
      )
  )
  return tuple(deduped)


def _load_split(spec: DatasetSpec, repeat: int, fold: int):
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(spec.task_id)
  dataset = task.get_dataset()
  target_name = task.target_name or dataset.default_target_attribute
  x, y, _, _ = dataset.get_data(
      target=target_name,
      dataset_format="dataframe",
  )
  train_idx, test_idx = task.get_train_test_split_indices(
      fold=fold,
      repeat=repeat,
  )
  return (
      x.iloc[train_idx].copy(),
      y.iloc[train_idx],
      x.iloc[test_idx].copy(),
      y.iloc[test_idx],
      np.asarray(test_idx),
  )


def _build_ttt_config(args: argparse.Namespace) -> tabfm.TabFMTestTimeTraining:
  return tabfm.TabFMTestTimeTraining(
      lora_rank=args.ttt_lora_rank,
      target_layers=_canonical_ttt_target_layers(args.ttt_target_layers),
      steps=args.ttt_steps,
      batch_size=args.ttt_batch_size,
      gradient_accumulation_steps=args.ttt_gradient_accumulation_steps,
      ema_decay=args.ttt_ema_decay,
      learning_rate=args.ttt_learning_rate,
      test_fraction=args.ttt_test_fraction,
      cell_embedder_row_chunk_size=args.ttt_cell_row_chunk_size,
      col_embedder_chunk_size=args.ttt_col_chunk_size,
      row_interactor_chunk_size=args.ttt_row_chunk_size,
      ffn_chunk_size=args.ttt_ffn_chunk_size,
      gradient_checkpointing=_ttt_gradient_checkpointing_mode(args),
      cast_model_to_bfloat16=not args.no_ttt_bfloat16_model,
      max_train_rows=_ttt_row_cap(args, "ttt_max_train_rows"),
      max_test_rows=_ttt_row_cap(args, "ttt_max_test_rows"),
      seed=args.seed,
  )


def _build_regressor(model, method: str, args: argparse.Namespace):
  """Builds the estimator for a method.

  Baseline methods return a plain TabFMRegressor. TTT methods return a
  TestTimeTrainedRegressor wrapping a plain regressor, so all test-time
  training stays additive to the upstream pipeline.
  """
  kwargs = dict(
      n_estimators=args.n_estimators,
      batch_size=args.batch_size,
      random_state=args.seed,
      use_amp=not args.no_amp,
  )
  if method == "ensemble_ttt" and args.no_ensemble_nnls:
    kwargs["enable_nnls"] = False
  if method.startswith("default"):
    regressor = tabfm.TabFMRegressor(model=model, **kwargs)
  elif method.startswith("ensemble"):
    regressor = tabfm.TabFMRegressor.ensemble(model=model, **kwargs)
  else:
    raise ValueError(f"Unknown method: {method!r}")
  if _is_ttt_method(method):
    return tabfm.TestTimeTrainedRegressor(regressor, _build_ttt_config(args))
  return regressor


def _log_ttt_fit_plan(regressor, method, n_rows):
  """Logs how many member adapters a TTT run will fit."""
  del n_rows
  if not _is_ttt_method(method):
    return
  base = regressor.regressor
  if regressor.config.steps == 0:
    _log(f"[{method}] TTT steps=0: no adapter optimization.")
    return
  weights = (
      "the plain regressor's NNLS weights"
      if base.enable_nnls
      else "uniform averaging"
  )
  _log(
      f"[{method}] one adapter round over {int(base.n_estimators)} members;"
      f" ensemble combination uses {weights}."
  )


def _load_model(args: argparse.Namespace):
  _log(f"Loading PyTorch TabFM regression model on {args.device}...")
  return tabfm.tabfm_v1_0_0_pytorch.load(
      model_type="regression",
      device=args.device,
      use_cache=False,
  )


def _metrics(y_true, pred: np.ndarray) -> dict[str, float]:
  from sklearn.metrics import (  # pylint: disable=g-import-not-at-top
      mean_absolute_error,
      mean_squared_error,
      r2_score,
  )

  y_true = np.asarray(y_true, dtype=float).ravel()
  pred = np.asarray(pred, dtype=float).ravel()
  return {
      "rmse": mean_squared_error(y_true, pred) ** 0.5,
      "mae": mean_absolute_error(y_true, pred),
      "r2": r2_score(y_true, pred),
  }


def _existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
  if not path.exists():
    return set()
  existing = pd.read_csv(path)
  keys = set()
  for _, row in existing.iterrows():
    if row.get("status") == "ok":
      keys.add((
          str(row["dataset_id"]),
          str(row["method"]),
          str(row["repeat"]),
          str(row["fold"]),
      ))
  return keys


def _append_summary(path: Path, row: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  write_header = not path.exists()
  with path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
    if write_header:
      writer.writeheader()
    writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _write_predictions(
    run_dir: Path,
    spec: DatasetSpec,
    method: str,
    repeat: int,
    fold: int,
    test_indices: np.ndarray,
    y_true,
    pred: np.ndarray,
) -> None:
  pred_dir = run_dir / "predictions"
  pred_dir.mkdir(parents=True, exist_ok=True)
  out_path = (
      pred_dir
      / f"{spec.dataset_id}_{_safe_name(spec.name)}_{method}_r{repeat}f{fold}.csv"
  )
  pd.DataFrame({
      "row_index": test_indices,
      "y_true": np.asarray(y_true, dtype=float).ravel(),
      "y_pred": np.asarray(pred, dtype=float).ravel(),
  }).to_csv(out_path, index=False)


def _base_result_row(
    args: argparse.Namespace,
    spec: DatasetSpec,
    method: str,
    x_train=None,
    x_test=None,
) -> dict[str, object]:
  return {
      "dataset_name": spec.name,
      "dataset_id": spec.dataset_id,
      "method": method,
      "openml_suite": args.openml_suite,
      "repeat": args.repeat,
      "fold": args.fold,
      "n_train": "" if x_train is None else len(x_train),
      "n_test": "" if x_test is None else len(x_test),
      "n_features": "" if x_train is None else x_train.shape[1],
      "n_estimators": _effective_n_estimators(method, args),
      "ensemble_enable_nnls": (
          _ensemble_enable_nnls(method, args)
          if method.startswith("ensemble")
          else ""
      ),
      "batch_size": args.batch_size,
      "ttt_steps": args.ttt_steps if _is_ttt_method(method) else "",
      "ttt_batch_size": args.ttt_batch_size if _is_ttt_method(method) else "",
      "ttt_gradient_accumulation_steps": (
          args.ttt_gradient_accumulation_steps
          if _is_ttt_method(method)
          else ""
      ),
      "ttt_ema_decay": args.ttt_ema_decay if _is_ttt_method(method) else "",
      "ttt_learning_rate": (
          args.ttt_learning_rate if _is_ttt_method(method) else ""
      ),
      "ttt_lora_rank": args.ttt_lora_rank if _is_ttt_method(method) else "",
      "ttt_target_layers": (
          ",".join(_canonical_ttt_target_layers(args.ttt_target_layers))
          if _is_ttt_method(method)
          else ""
      ),
      "ttt_cell_embedder_row_chunk_size": (
          args.ttt_cell_row_chunk_size if _is_ttt_method(method) else ""
      ),
      "ttt_col_embedder_chunk_size": (
          args.ttt_col_chunk_size if _is_ttt_method(method) else ""
      ),
      "ttt_row_interactor_chunk_size": (
          args.ttt_row_chunk_size if _is_ttt_method(method) else ""
      ),
      "ttt_ffn_chunk_size": (
          args.ttt_ffn_chunk_size if _is_ttt_method(method) else ""
      ),
      "ttt_gradient_checkpointing": (
          _ttt_gradient_checkpointing_mode(args)
          if _is_ttt_method(method)
          else ""
      ),
      "ttt_cast_model_to_bfloat16": (
          not args.no_ttt_bfloat16_model if _is_ttt_method(method) else ""
      ),
      "ttt_max_train_rows": (
          _format_ttt_row_cap(_ttt_row_cap(args, "ttt_max_train_rows"))
          if _is_ttt_method(method)
          else ""
      ),
      "ttt_max_test_rows": (
          _format_ttt_row_cap(_ttt_row_cap(args, "ttt_max_test_rows"))
          if _is_ttt_method(method)
          else ""
      ),
  }


def _run_one(
    model,
    spec: DatasetSpec,
    method: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, object]:
  x_train, y_train, x_test, y_test, test_indices = _load_split(
      spec,
      repeat=args.repeat,
      fold=args.fold,
  )
  row = _base_result_row(args, spec, method, x_train=x_train, x_test=x_test)

  reg = _build_regressor(model, method, args)
  _log_ttt_fit_plan(reg, method, len(x_train))
  start = time.perf_counter()
  fit_start = start
  _log(
      f"[{method}] fitting {spec.name}: n_train={len(x_train)},"
      f" n_test={len(x_test)}, n_features={x_train.shape[1]}"
  )
  reg.fit(x_train, y_train)
  fit_seconds = time.perf_counter() - fit_start

  predict_start = time.perf_counter()
  _log(f"[{method}] predicting {spec.name}")
  pred = np.asarray(reg.predict(x_test), dtype=float).ravel()
  predict_seconds = time.perf_counter() - predict_start
  total_seconds = time.perf_counter() - start
  _log(
      f"[{method}] completed {spec.name}: fit={fit_seconds:.1f}s, "
      f"predict={predict_seconds:.1f}s"
  )

  row.update({
      "status": "ok",
      "fit_seconds": fit_seconds,
      "predict_seconds": predict_seconds,
      "total_seconds": total_seconds,
      **_metrics(y_test, pred),
  })
  if not args.no_save_predictions:
    _write_predictions(
        run_dir,
        spec,
        method,
        args.repeat,
        args.fold,
        test_indices,
        y_test,
        pred,
    )
  return row


def _cleanup_cuda() -> None:
  gc.collect()
  try:
    import torch  # pylint: disable=g-import-not-at-top

    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  except ImportError:
    pass


def _print_dry_run(
    datasets: Iterable[DatasetSpec],
    methods: Sequence[str],
    args: argparse.Namespace,
) -> None:
  _log(f"OpenML suite: {args.openml_suite}")
  _log(f"repeat={args.repeat}, fold={args.fold}")
  _log(f"methods: {', '.join(methods)}")
  for spec in datasets:
    _log(f"{spec.name}: dataset_id={spec.dataset_id}")


def main() -> None:
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

  args = _parse_args()
  _configure_openml_cache(args.openml_cache_dir, log=False)
  args.results_dir.mkdir(parents=True, exist_ok=True)
  log_file = None
  methods = tuple(args.method or ("ensemble", "ensemble_ttt"))
  datasets = _resolve_datasets(args.dataset, args.openml_suite)
  run_slug = _run_slug(methods, datasets, args)
  run_dir = args.results_dir / "runs" / run_slug
  summary_path = run_dir / "summary.csv"
  errors_dir = run_dir / "errors"
  if args.log_file:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    log_file = args.results_dir / "logs" / f"{run_slug}__{timestamp}.log"

  with _tee_output(log_file):
    _log(
        "Starting TabArena regression benchmark "
        f"with results_dir={args.results_dir}"
    )
    _log(f"Run directory: {run_dir}")
    if log_file is not None:
      _log(f"Logging to {log_file}")
    _configure_openml_cache(args.openml_cache_dir)
    _log(f"Requested methods: {', '.join(methods)}")

    if args.dry_run:
      _print_dry_run(datasets, methods, args)
      return

    if args.overwrite and run_dir.exists():
      _log(f"Overwriting existing run directory at {run_dir}")
      shutil.rmtree(run_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    completed = set() if args.overwrite else _existing_keys(summary_path)

    for method in methods:
      model = _load_model(args)
      try:
        for idx, spec in enumerate(datasets, start=1):
          key = (
              str(spec.dataset_id),
              method,
              str(args.repeat),
              str(args.fold),
          )
          if key in completed:
            _log(f"Skipping {spec.name} / {method}: already in summary.csv")
            continue

          _log(
              f"[{method}] ({idx}/{len(datasets)}) running {spec.name} "
              f"(dataset_id={spec.dataset_id})"
          )
          try:
            _log(f"[{method}] loading split for {spec.name}")
            row = _run_one(model, spec, method, args, run_dir)
            _log(
                f"  rmse={row['rmse']:.6g}, mae={row['mae']:.6g},"
                f" r2={row['r2']:.6g}, seconds={row['total_seconds']:.1f}"
            )
          except Exception as exc:  # pylint: disable=broad-except
            row = _base_result_row(args, spec, method)
            row.update({
                "status": "error",
                "error": repr(exc),
            })
            err_path = (
                errors_dir
                / f"{spec.dataset_id}_{_safe_name(spec.name)}_{method}_r{args.repeat}f{args.fold}.txt"
            )
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"  ERROR: {exc!r}. Traceback saved to {err_path}")
            if args.fail_fast:
              _append_summary(summary_path, row)
              raise

          _append_summary(summary_path, row)
          _cleanup_cuda()
      finally:
        del model
        _cleanup_cuda()

    _log(f"Wrote {summary_path}")


if __name__ == "__main__":
  main()
