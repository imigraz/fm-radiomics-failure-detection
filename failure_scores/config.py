"""
Benchmark configuration: one YAML file -> one immutable ``Benchmark`` that is
passed explicitly to every stage (no module-level state).

YAML keys (``benchmarks/*.yaml``):

    name                   output subdirectory name
    reference_site         site the model was trained on; the reference set
    query_sites            sites to score; their order fixes the bootstrap's
                           random stream (keep it to reproduce results)
    dataset_dirs           site -> nnU-Net dataset folder under ``data_root``
    pred_dirs              site -> prediction folder name inside each
                           ``<pred_root>/fold_<k>/``
    task_name_to_tag       nnU-Net task name (as in the fold JSONs) -> site
    fold_json_name         per-fold validation JSON (reference fold membership)
    bilateral_split_sites  sites whose volumes hold a left and a right structure,
                           e.g. both hippocampi (optional)
    bilateral_split_axis   0 = x = left-right (optional, default 0)
    atlas_flip_r_half      Atlas-RCA: mirror R halves to L before registering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class Benchmark:
    name: str
    reference_site: str
    query_sites: Tuple[str, ...]
    dataset_dirs: Dict[str, Path]
    pred_dirs: Dict[str, str]
    task_name_to_tag: Dict[str, str]
    fold_json_name: str
    pred_root: Path
    out_dir: Path
    bilateral_split_sites: Tuple[str, ...] = ()
    bilateral_split_axis: int = 0
    atlas_flip_r_half: bool = False
    extra: Dict = field(default_factory=dict)

    @property
    def sites(self) -> List[str]:
        return [self.reference_site, *self.query_sites]

    def bilateral_axis(self, site: str) -> Optional[int]:
        """Split axis (YAML convention) if ``site`` is split into L/R halves."""
        return self.bilateral_split_axis if site in self.bilateral_split_sites else None

    @property
    def mv_dir(self) -> Path:
        return self.out_dir / "mv_masks"


_KEYS = {"name", "reference_site", "query_sites", "dataset_dirs", "pred_dirs",
         "task_name_to_tag", "fold_json_name", "bilateral_split_sites",
         "bilateral_split_axis", "atlas_flip_r_half"}


def load(path, data_root="datasets", pred_root="predictions", out_root="outputs",
         out_dir=None) -> Benchmark:
    with open(path) as f:
        spec = yaml.safe_load(f)
    missing = {"name", "reference_site", "query_sites", "dataset_dirs", "pred_dirs",
               "task_name_to_tag", "fold_json_name"} - set(spec)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    sites = [spec["reference_site"], *spec["query_sites"]]
    for key in ("dataset_dirs", "pred_dirs"):
        absent = [s for s in sites if s not in spec[key]]
        if absent:
            raise ValueError(f"{path}: {key} has no entry for {absent}")
    return Benchmark(
        name=spec["name"],
        reference_site=spec["reference_site"],
        query_sites=tuple(spec["query_sites"]),
        dataset_dirs={s: Path(data_root) / d for s, d in spec["dataset_dirs"].items()},
        pred_dirs=dict(spec["pred_dirs"]),
        task_name_to_tag=dict(spec["task_name_to_tag"]),
        fold_json_name=spec["fold_json_name"],
        pred_root=Path(pred_root),
        out_dir=Path(out_dir) if out_dir else Path(out_root) / spec["name"],
        bilateral_split_sites=tuple(spec.get("bilateral_split_sites") or ()),
        bilateral_split_axis=int(spec.get("bilateral_split_axis", 0)),
        atlas_flip_r_half=bool(spec.get("atlas_flip_r_half", False)),
        extra={k: v for k, v in spec.items() if k not in _KEYS},
    )
