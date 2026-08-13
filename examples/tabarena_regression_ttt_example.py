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

"""Runs the PyTorch TabFM ensemble + LoRA TTT on TabArena regressions.

This resolves TabArena split definitions by dataset name or dataset id and only
runs repeat-0 / fold-0 by default, matching TabArena-Lite's split
convention. It is also useful as a lightweight downloader because OpenML caches
each task locally as it is loaded.

The ensemble preset keeps the plain regressor's NNLS weights, which come from
its own leakage-free out-of-fold fit; TTT only trains one adapter per member.

Example:
  python examples/tabarena_regression_ttt_example.py --download-only
  python examples/tabarena_regression_ttt_example.py --n-estimators=32
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Sequence

import numpy as np
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


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument(
      "--dataset",
      action="append",
      default=[],
      help=(
          "Dataset name or OpenML dataset id to run. May be repeated. Dataset"
          " ids are resolved through the current OpenML suite. Defaults to all"
          " TabArena regression datasets."
      ),
  )
  parser.add_argument("--repeat", type=int, default=0)
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--openml-suite", default=DEFAULT_TABARENA_OPENML_SUITE)
  parser.add_argument("--download-only", action="store_true")
  parser.add_argument("--openml-cache-dir", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--n-estimators", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--no-amp", action="store_true")
  parser.add_argument("--lora-rank", type=int, default=8)
  parser.add_argument(
      "--ttt-target-layers",
      default="cell_embedder.in_linear",
      help=(
          "Comma-separated TTT target layers. Defaults to"
          " cell_embedder.in_linear."
      ),
  )
  parser.add_argument("--ttt-steps", type=int, default=50)
  parser.add_argument(
      "--ttt-batch-size",
      type=int,
      default=4,
      help="Independent context/query splits forwarded per TTT microbatch.",
  )
  parser.add_argument(
      "--ttt-gradient-accumulation-steps",
      type=int,
      default=1,
      help="TTT microbatches accumulated before each optimizer update.",
  )
  parser.add_argument("--ttt-ema-decay", type=float, default=0.9)
  parser.add_argument("--ttt-learning-rate", type=float, default=1e-4)
  parser.add_argument("--test-fraction", type=float, default=0.2)
  parser.add_argument("--cell-row-chunk-size", type=int, default=128)
  parser.add_argument("--col-chunk-size", type=int, default=4)
  parser.add_argument("--row-chunk-size", type=int, default=64)
  parser.add_argument("--ffn-chunk-size", type=int, default=4096)
  parser.add_argument(
      "--ttt-gradient-checkpointing",
      choices=("none", "icl", "all"),
      default="none",
      help=(
          "Gradient checkpointing mode. 'icl' balances memory and speed; "
          "'all' saves the most memory; 'none' is fastest."
      ),
  )
  parser.add_argument(
      "--no-ttt-bfloat16-model",
      action="store_true",
      help="Keep the frozen base model in fp32 during TTT (required on CPU).",
  )
  parser.add_argument(
      "--max-train-rows",
      type=int,
      default=None,
      help="Optional TTT training-row cap; defaults to all rows.",
  )
  parser.add_argument(
      "--max-test-rows",
      type=int,
      default=None,
      help="Optional TTT query-row cap; defaults to all available rows.",
  )
  return parser.parse_args()


def _normalize_name(name: str) -> str:
  return name.lower().replace("_", "-")


def _is_regression_task(task) -> bool:
  return "regression" in str(getattr(task, "task_type", "")).lower()


def _resolve_tabarena_regression_tasks(
    requested_datasets: Sequence[str],
    suite_name: str,
) -> tuple[tuple[str, int, int], ...]:
  """Resolves TabArena regression dataset names/ids to split definitions."""
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

  resolved_by_name: dict[str, tuple[str, int, int]] = {}
  resolved_by_dataset_id: dict[int, tuple[str, int, int]] = {}
  suite = openml.study.get_suite(suite_name)
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

    resolved = (canonical_names[dataset_key], int(task.id), int(dataset.id))
    resolved_by_dataset_id[int(dataset.id)] = resolved
    if dataset_key in target_names and dataset_key not in resolved_by_name:
      resolved_by_name[dataset_key] = resolved

  missing = sorted(target_names - set(resolved_by_name))
  if missing:
    raise ValueError(
        "Could not resolve TabArena regression dataset(s) from OpenML suite "
        f"{suite_name!r}: {', '.join(missing)}"
    )

  selected = []
  for name in TABARENA_REGRESSION_DATASETS:
    name_key = _normalize_name(name)
    if name_key in target_names:
      selected.append(resolved_by_name[name_key])

  for item_id in sorted(requested_dataset_ids):
    if item_id in resolved_by_dataset_id:
      selected.append(resolved_by_dataset_id[item_id])
    else:
      raise ValueError(
          f"OpenML dataset id {item_id} is not a TabArena regression dataset"
          f" in suite {suite_name!r}."
      )

  return tuple(selected)


def _configure_openml_cache(cache_dir: Path | None) -> None:
  if cache_dir is None:
    return
  import openml  # pylint: disable=g-import-not-at-top

  cache_dir.mkdir(parents=True, exist_ok=True)
  openml.config.set_root_cache_directory(str(cache_dir))


def _load_split(task_id: int, repeat: int, fold: int):
  """Returns train/test arrays for one OpenML task split."""
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
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
  )


def _evaluate_rmse(y_true, pred: np.ndarray) -> float:
  from sklearn.metrics import mean_squared_error  # pylint: disable=g-import-not-at-top

  y_true = np.asarray(y_true, dtype=float).ravel()
  pred = np.asarray(pred, dtype=float).ravel()
  return mean_squared_error(y_true, pred) ** 0.5


def _canonical_ttt_target_layers(value) -> tuple[str, ...]:
  config = tabfm.TabFMTestTimeTraining.from_value({"target_layers": value})
  return (
      config.target_layers
      if config is not None
      else (tabfm.TabFMTestTimeTraining.target_layers)
  )


def _build_test_time_training(args: argparse.Namespace):
  return tabfm.TabFMTestTimeTraining(
      lora_rank=args.lora_rank,
      target_layers=_canonical_ttt_target_layers(args.ttt_target_layers),
      steps=args.ttt_steps,
      batch_size=args.ttt_batch_size,
      gradient_accumulation_steps=args.ttt_gradient_accumulation_steps,
      ema_decay=args.ttt_ema_decay,
      learning_rate=args.ttt_learning_rate,
      test_fraction=args.test_fraction,
      cell_embedder_row_chunk_size=args.cell_row_chunk_size,
      col_embedder_chunk_size=args.col_chunk_size,
      row_interactor_chunk_size=args.row_chunk_size,
      ffn_chunk_size=args.ffn_chunk_size,
      gradient_checkpointing=args.ttt_gradient_checkpointing,
      cast_model_to_bfloat16=not args.no_ttt_bfloat16_model,
      max_train_rows=args.max_train_rows,
      max_test_rows=args.max_test_rows,
      seed=args.seed,
  )


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset_name",
            "dataset_id",
            "repeat",
            "fold",
            "n_train",
            "n_test",
            "rmse",
            "fit_predict_seconds",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  args = _parse_args()
  _configure_openml_cache(args.openml_cache_dir)
  tasks = _resolve_tabarena_regression_tasks(args.dataset, args.openml_suite)

  model = None
  ttt = None
  if not args.download_only:
    model = tabfm.tabfm_v1_0_0_pytorch.load(
        model_type="regression",
        device=args.device,
        use_cache=False,
    )
    ttt = _build_test_time_training(args)

  rows = []
  for dataset_name, task_id, dataset_id in tasks:
    x_train, y_train, x_test, y_test = _load_split(
        task_id=task_id,
        repeat=args.repeat,
        fold=args.fold,
    )
    print(
        f"{dataset_name}: dataset_id={dataset_id}, train={len(x_train)},"
        f" test={len(x_test)}"
    )
    if args.download_only:
      continue

    reg = tabfm.TestTimeTrainedRegressor(
        tabfm.TabFMRegressor.ensemble(
            model=model,
            n_estimators=args.n_estimators,
            batch_size=args.batch_size,
            random_state=args.seed,
            use_amp=not args.no_amp,
        ),
        ttt,
    )
    print(f"  TTT: {args.n_estimators} member adapter fits")
    start = time.perf_counter()
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    elapsed = time.perf_counter() - start
    rmse = _evaluate_rmse(y_test, pred)
    rows.append({
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "repeat": args.repeat,
        "fold": args.fold,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "rmse": rmse,
        "fit_predict_seconds": elapsed,
    })
    print(f"  rmse={rmse:.6g}, seconds={elapsed:.1f}")

  if args.output is not None and rows:
    _write_results(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
