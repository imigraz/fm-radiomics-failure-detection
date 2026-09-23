"""
FM stage: embed every case of every site, cache the embeddings, score them.

Cache: ``<out_dir>/fm_cache/{backbone}_{mode}_{site}.npz`` with ``ids`` and
``emb`` (float32, one row per case). A site is embedded only if one of its
mode files is missing.

Scores: ``<out_dir>/scores/fm_{backbone}.csv``, long format
``case_id, site, method, score`` with method ``{backbone}_md_{mode}``. The
reference is fit on the L2-normalised reference embeddings (ground-truth
masks); reference cases are scored leave-one-out, query cases against
the full fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..io import iter_site_cases
from ..scoring import MahalanobisScorer
from . import MODES

Embeddings = Dict[str, Dict[str, Tuple[List[str], np.ndarray]]]   # mode -> site -> (ids, emb)

def cache_path(out_dir: Path, backbone: str, mode: str, site: str) -> Path:
    return out_dir / "fm_cache" / f"{backbone}_{mode}_{site}.npz"


def load_cache(out_dir: Path, backbone: str, sites: Sequence[str]) -> Embeddings:
    out: Embeddings = {m: {} for m in MODES}
    for mode in MODES:
        for site in sites:
            z = np.load(cache_path(out_dir, backbone, mode, site), allow_pickle=True)
            out[mode][site] = (z["ids"].tolist(), z["emb"])
    return out


def embed_sites(
    backbone: str,
    out_dir: Path,
    site_dirs: Dict[str, Path],
    reference_site: str,
    mv_dir: Path,
    bilateral_sites: Sequence[str] = (),
    bilateral_axis: int = 0,
) -> Embeddings:
    """Embed (or load from cache) all sites in ``site_dirs``."""
    sites = list(site_dirs)
    todo = [s for s in sites
            if not all(cache_path(out_dir, backbone, m, s).exists() for m in MODES)]
    if todo:
        if backbone == "3dino":
            from . import dino3d
            model, device = dino3d.load_model()
            embed = lambda img, mask: dino3d.embed_case(img, mask, model, device)
        else:
            from . import vit2d
            processor, model, device = vit2d.load_model(backbone)
            embed = lambda img, mask: vit2d.embed_case(img, mask, processor, model, device)

        for site in todo:
            ids, embs = [], {m: [] for m in MODES}
            cases = iter_site_cases(
                site, site_dirs[site], is_reference=(site == reference_site), mv_dir=mv_dir,
                bilateral_axis=bilateral_axis if site in bilateral_sites else None)
            for case_id, img, mask in cases:
                e = embed(img, mask)
                if e is None:
                    print(f"      skip {site}/{case_id}: empty mask (at 1 mm)")
                    continue
                ids.append(case_id)
                for m in MODES:
                    embs[m].append(e[m])
            if not ids:
                if site == reference_site:
                    raise ValueError(f"{backbone}: no reference case of {site} could be embedded")
                print(f"    WARNING {backbone}/{site}: no case with a mask; site not scored")
            for m in MODES:
                p = cache_path(out_dir, backbone, m, site)
                p.parent.mkdir(parents=True, exist_ok=True)
                emb = np.vstack(embs[m]) if ids else np.zeros((0, 0))
                np.savez_compressed(p, ids=np.array(ids, dtype=object),
                                    emb=emb.astype(np.float32))
            print(f"    {backbone}/{site}: {len(ids)} cases embedded")
    return load_cache(out_dir, backbone, sites)


def score_embeddings(embeddings: Embeddings, backbone: str, reference_site: str,
                     query_sites: Sequence[str]) -> pd.DataFrame:
    """Mahalanobis++ failure scores for every mode, long format."""
    rows = []
    for mode in MODES:
        method = f"{backbone}_md_{mode}"
        ref_ids, X_ref = embeddings[mode][reference_site]
        scorer = MahalanobisScorer("l2").fit(X_ref)
        rows.append(pd.DataFrame({"case_id": ref_ids, "site": reference_site,
                                  "method": method, "score": scorer.loo_scores(X_ref)}))
        for site in query_sites:
            ids, X = embeddings[mode][site]
            if not ids:
                continue
            rows.append(pd.DataFrame({"case_id": ids, "site": site, "method": method,
                                      "score": scorer.score(X).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def run(backbone: str, out_dir: Path, site_dirs: Dict[str, Path], reference_site: str,
        query_sites: Sequence[str], mv_dir: Path, bilateral_sites: Sequence[str] = (),
        bilateral_axis: int = 0) -> pd.DataFrame:
    """Embed, score and write ``scores/fm_{backbone}.csv``."""
    sites = [reference_site, *query_sites]
    emb = embed_sites(backbone, out_dir, {s: site_dirs[s] for s in sites}, reference_site,
                      mv_dir, bilateral_sites, bilateral_axis)
    scores = score_embeddings(emb, backbone, reference_site, query_sites)
    path = out_dir / "scores" / f"fm_{backbone}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    print(f"  saved {path}")
    return scores
