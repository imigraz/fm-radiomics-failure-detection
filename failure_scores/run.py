"""
Pipeline CLI.

    python -m failure_scores.run --benchmark benchmarks/prostate_msd.yaml \\
        --stage {extract,fm,baselines,evaluate,all}

Stages (outputs in ``outputs/<name>/`` unless ``--out_dir`` is given):

  extract    majority-vote masks, Dice (risk), shape features, Rad-MD-Mask
             -> mv_masks/, cases.csv, shape_raw.csv, shape_features.csv,
                scores/shape.csv
  fm         DINOv3 / RadDINO / 3DINO embeddings and -MD scores (GPU)
             -> fm_cache/, scores/fm_<backbone>.csv
  baselines  Ens-Mpd, Rad-L2 / Rad-MD-Img, Atlas-RCA
             -> scores/{ens_mpd,frd,atlas_rca}.csv, frd_cache/, atlas_rca_cache/
  evaluate   E-AURC table and paired tests -> evaluation/

Atlas-RCA registers every query case to every reference case (hours of ANTs
SyNRA); registrations are cached per case in ``atlas_rca_cache/`` and never
deleted. 3DINO needs its own environment (numpy<2) and is not run by
default; see the README.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import evaluation, masks, shape
from .config import Benchmark, load
from .io import find_images, site_case_ids

STAGES = ("extract", "fm", "baselines", "evaluate")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved {path}")


def stage_extract(cfg: Benchmark) -> None:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    folds = masks.discover_folds(cfg.pred_root)
    held_out = masks.reference_fold_membership(
        folds, cfg.fold_json_name, cfg.task_name_to_tag, cfg.reference_site)
    ref_dir = cfg.dataset_dirs[cfg.reference_site]
    if not masks.has_labels(ref_dir):
        raise FileNotFoundError(f"reference site {cfg.reference_site} needs labels in "
                                f"{ref_dir / 'labelsTr'} (or labels/)")
    if not held_out:
        raise FileNotFoundError(
            f"no reference case of {cfg.reference_site} in the fold JSONs "
            f"{cfg.pred_root}/fold_*/{cfg.fold_json_name}; check fold_json_name and "
            f"task_name_to_tag in the benchmark YAML")
    extractor = shape.build_extractor()
    cases, feats = [], []
    for site in cfg.sites:
        is_ref = site == cfg.reference_site
        if not is_ref:
            n = masks.build_mv_masks(site, folds, cfg.pred_dirs[site], cfg.mv_dir)
            print(f"  [{site}] {n} majority-vote masks written")
        f = shape.extract_site(site, cfg.dataset_dirs[site], is_ref, cfg.mv_dir,
                               extractor, cfg.bilateral_axis(site))
        if len(f):
            feats.append(f)
        if is_ref:
            # Reference: cases with a held-out fold, a valid Dice and shape features.
            dice = masks.site_dice(site, cfg.dataset_dirs[site], folds, cfg.pred_dirs[site],
                                   True, held_out, cfg.bilateral_axis(site))
            ref = dice.merge(f[["case_id", "site"]], on=["case_id", "site"], how="inner") \
                if len(f) else dice.iloc[:0]
            if ref.empty:
                raise ValueError(f"reference site {site}: no case with a label, a fold-JSON "
                                 f"entry, a prediction of its held-out fold and a non-empty mask")
            cases.append(ref)
            print(f"  [{site}] Dice {len(dice)} cases, shape features {len(f)} cases")
        else:
            cases.append(_query_cases(cfg, site, f, folds))
    cases = pd.concat([c for c in cases if len(c)], ignore_index=True)
    feats = pd.concat(feats, ignore_index=True)
    _save(cases, cfg.out_dir / "cases.csv")
    _save(feats, cfg.out_dir / "shape_raw.csv")
    selected, _ = shape.run(feats.merge(cases[["case_id", "site"]]), cfg.reference_site,
                            cfg.query_sites, cfg.out_dir)
    print(f"  Rad-MD-Mask: {len(selected)} shape features: "
          f"{[s.replace('original_shape_', '') for s in selected]}")


def _query_cases(cfg: Benchmark, site: str, feats: pd.DataFrame, folds) -> pd.DataFrame:
    """
    ``case_id, site, dice`` of every case of a query site with an image
    (sites in YAML order, case ids sorted). ``dice`` (the risk) is set only for
    the cases that are evaluated: labelled, with a valid Dice and a non-empty
    majority-vote mask; the other cases are scored where possible, not evaluated.
    """
    ids = site_case_ids(cfg.dataset_dirs[site], cfg.bilateral_axis(site))
    if not ids:
        print(f"  WARNING [{site}] no images in {cfg.dataset_dirs[site]}; site skipped")
        return pd.DataFrame(columns=["case_id", "site", "dice"])
    with_mask = set(feats["case_id"]) if len(feats) else set()
    no_mask = [c for c in ids if c not in with_mask]
    if not with_mask:
        print(f"  WARNING [{site}] no predictions found (fold_*/{cfg.pred_dirs[site]}); "
              f"its {len(ids)} cases get only image-based scores")
    elif no_mask:
        print(f"  WARNING [{site}] {len(no_mask)} cases have no shape features: the "
              f"majority-vote mask is missing, empty or too small (single voxel or line); "
              f"not evaluated: {', '.join(no_mask)}")
    dice = {}
    if masks.has_labels(cfg.dataset_dirs[site]):
        d = masks.site_dice(site, cfg.dataset_dirs[site], folds, cfg.pred_dirs[site],
                            False, None, cfg.bilateral_axis(site))
        dice = {c: v for c, v in zip(d["case_id"], d["dice"]) if c in with_mask}
        if d.empty:
            print(f"  WARNING [{site}] labels found, but none gives a Dice: label files must be "
                  f"named like the images without _0000 and have a prediction")
        print(f"  [{site}] Dice {len(d)} cases, shape features {len(with_mask)} cases, "
              f"{len(dice)} of {len(ids)} cases evaluated")
    else:
        print(f"  [{site}] no labels: {len(ids)} cases scored, not evaluated")
    return pd.DataFrame({"case_id": ids, "site": site,
                         "dice": [dice.get(c, np.nan) for c in ids]})


def stage_fm(cfg: Benchmark, backbones: Sequence[str]) -> None:
    from .fm import extract
    for backbone in backbones:
        print(f"── FM: {backbone}")
        extract.run(backbone, cfg.out_dir, cfg.dataset_dirs, cfg.reference_site,
                    cfg.query_sites, cfg.mv_dir, cfg.bilateral_split_sites,
                    cfg.bilateral_split_axis)


def _frd_features(cfg: Benchmark, site: str):
    from .baselines import frd
    path = cfg.out_dir / "frd_cache" / f"{site}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["features"], z["case_ids"].tolist()
    X, ids = frd.extract_site(find_images(cfg.dataset_dirs[site]),
                              cfg.out_dir / "frd_cache" / f"{site}_1mm",
                              cfg.bilateral_axis(site))
    np.savez_compressed(path, features=X, case_ids=np.array(ids, dtype=object))
    return X, ids


def stage_baselines(cfg: Benchmark) -> None:
    from .baselines import atlas_rca, ens_mpd, frd
    folds = masks.discover_folds(cfg.pred_root)
    cases = evaluation.load_cases(cfg.out_dir)
    # Query sites without images were skipped by the extract stage.
    sites = [s for s in cfg.query_sites if (cases["site"] == s).any()]
    for s in cfg.query_sites:
        if s not in sites:
            print(f"  WARNING [{s}] no cases; site skipped")

    print("── Ens-Mpd")
    _save(_concat([ens_mpd.score_site(s, cases[cases.site == s]["case_id"].tolist(), folds,
                                      cfg.pred_dirs[s], cfg.bilateral_axis(s))
                   for s in sites]), cfg.out_dir / "scores" / "ens_mpd.csv")

    print("── Rad-L2 / Rad-MD-Img (FRD features)")
    feats = {s: _frd_features(cfg, s) for s in [cfg.reference_site, *sites]}
    _save(frd.score(feats, cfg.reference_site, sites), cfg.out_dir / "scores" / "frd.csv")

    print("── Atlas-RCA")
    held_out = masks.reference_fold_membership(
        folds, cfg.fold_json_name, cfg.task_name_to_tag, cfg.reference_site)
    refs = atlas_rca.reference_cases(cfg.dataset_dirs[cfg.reference_site], held_out)
    parts = []
    for site in sites:
        qc = atlas_rca.query_cases(site, cfg.dataset_dirs[site], cfg.mv_dir,
                                   cfg.out_dir / "atlas_rca_tmp", cfg.bilateral_axis(site))
        parts.append(atlas_rca.score_site(site, qc, refs, cfg.out_dir / "atlas_rca_cache",
                                          cfg.atlas_flip_r_half))
    _save(_concat(parts), cfg.out_dir / "scores" / "atlas_rca.csv")


def _concat(parts) -> pd.DataFrame:
    return pd.concat(parts) if parts else pd.DataFrame(columns=["case_id", "site", "method", "score"])


def stage_evaluate(cfg: Benchmark) -> None:
    cases, scores = evaluation.load_cases(cfg.out_dir), evaluation.load_scores(cfg.out_dir)
    methods, pairs = evaluation.available_methods(scores)
    if not methods:
        raise FileNotFoundError(f"no failure scores in {cfg.out_dir / 'scores'}; "
                                f"run the extract stage first")
    _save(evaluation.scores_table(cases, scores, cfg.query_sites),
          cfg.out_dir / "failure_scores.csv")
    sites, skipped = evaluation.labelled_sites(cases, cfg.query_sites)
    for reason, names in skipped.items():
        print(f"  not evaluated ({reason}): {', '.join(names)}")
    if not sites:
        print("  no query site to evaluate: failure scores written, evaluation skipped")
        return
    evaluated = evaluation.evaluated_cases(cases)
    methods, dropped = evaluation.complete_methods(evaluated, scores, sites, methods)
    for m, missing in dropped.items():
        print(f"  WARNING {m} not evaluated: no score for {len(missing)} evaluated "
              f"case(s): {', '.join(missing)}")
    if not methods:
        print("  no method scores every evaluated case: evaluation skipped")
        return
    pairs = [(m, r) for m, r in pairs if m in methods and r in methods]
    print(f"  evaluating {len(methods)} methods on {', '.join(sites)}: {', '.join(methods)}")
    result = evaluation.evaluate(evaluated, scores, sites, methods, pairs)
    evaluation.print_tables(result)
    _save(evaluation.eaurc_table(result), cfg.out_dir / "evaluation" / "eaurc_table.csv")
    _save(evaluation.paired_tests_table(result), cfg.out_dir / "evaluation" / "paired_tests.csv")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", required=True, help="benchmark YAML")
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--out_dir", default=None, help="default: outputs/<name>")
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--pred_root", default="predictions")
    # 3DINO is opt-in: it needs its own environment (numpy<2, README).
    ap.add_argument("--fm_models", nargs="+", default=["dinov3", "raddino"],
                    choices=["dinov3", "raddino", "3dino"])
    args = ap.parse_args(argv)

    cfg = load(args.benchmark, data_root=args.data_root, pred_root=args.pred_root,
               out_dir=args.out_dir)
    stages = STAGES if args.stage == "all" else (args.stage,)
    print(f"benchmark {cfg.name}: reference {cfg.reference_site}, "
          f"query {', '.join(cfg.query_sites)}; stages {', '.join(stages)}; out {cfg.out_dir}")
    if "extract" in stages:
        stage_extract(cfg)
    if "fm" in stages:
        stage_fm(cfg, args.fm_models)
    if "baselines" in stages:
        stage_baselines(cfg)
    if "evaluate" in stages:
        stage_evaluate(cfg)


if __name__ == "__main__":
    main()
