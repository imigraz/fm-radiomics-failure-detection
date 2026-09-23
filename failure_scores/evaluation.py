"""
Evaluation of per-case failure scores with E-AURC.

Every method gives one failure score per case, oriented so that a higher score
means a higher predicted risk. Per-case risk is ``1 - Dice``. For each query
site we rank cases by score and compute

    AURC    mean selective risk over all coverages k/n
    AURC*   AURC of the oracle ranking (by true risk)
    E-AURC  AURC - AURC*        (0 = oracle, lower is better)

Within-site E-AURC is the equal-weight mean of the per-site E-AURC. Pooled
E-AURC ranks all query cases together and is computed once.

Uncertainty comes from one site-stratified, method-paired percentile
bootstrap: per replicate, case indices are drawn once per site and reused for
every method, so paired differences (method - ref) are valid.

Reproducibility of the paper numbers depends on the bootstrap's RNG stream,
which is determined by the case set, the case order within each site, the site
order (the benchmark YAML ``query_sites`` order), ``seed=42`` and
``n_bootstrap=2000``. The RNG is consumed only by sites, never by methods, so
the method set does not change the CIs.

Usage:
    python -m failure_scores.evaluation --benchmark benchmarks/prostate_msd.yaml \\
        --scores_dir outputs/prostate_msd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Paper methods
# ─────────────────────────────────────────────────────────────────────────────

# method key -> label used in the paper
METHODS: Dict[str, str] = {
    "ens_mpd":        "Ens-Mpd",
    "atlas_rca":      "Atlas-RCA",
    "rad_l2":         "Rad-L2",
    "rad_md_img":     "Rad-MD-Img",
    "rad_md_mask":    "Rad-MD-Mask",
    "dinov3_md_img":  "DINOv3-MD-Img",
    "dinov3_md_ov":   "DINOv3-MD-Ov",
    "dinov3_md_w":    "DINOv3-MD-W",
    "dinov3_md_ovw":  "DINOv3-MD-OvW",
    "raddino_md_img": "RadDINO-MD-Img",
    "raddino_md_ov":  "RadDINO-MD-Ov",
    "raddino_md_w":   "RadDINO-MD-W",
    "raddino_md_ovw": "RadDINO-MD-OvW",
    "3dino_md_img":   "3DINO-MD-Img",
    "3dino_md_ov":    "3DINO-MD-Ov",
    "3dino_md_w":     "3DINO-MD-W",
    "3dino_md_ovw":   "3DINO-MD-OvW",
}

# Paired tests reported in the paper, (method, ref): Δ = method - ref, so a
# negative Δ means `method` is better.
PAIRED_TESTS: List[Tuple[str, str]] = [
    ("dinov3_md_ovw", "rad_md_mask"),
    ("rad_md_mask",   "atlas_rca"),
    ("rad_md_mask",   "ens_mpd"),
    ("dinov3_md_ovw", "atlas_rca"),
    ("dinov3_md_ovw", "ens_mpd"),
]

N_BOOTSTRAP = 2000
SEED = 42
CONFIDENCE_LEVEL = 0.95
MIN_FINITE_REPS = 100   # fewer finite bootstrap replicates -> NaN CI / p
MIN_SITE_CASES = 4      # fewer evaluated cases -> the site is not evaluated


# ─────────────────────────────────────────────────────────────────────────────
# Loading the per-case scores
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(scores_dir: Path) -> pd.DataFrame:
    """
    All cases, ``case_id, site, dice``, in the order written by the extract
    stage (``cases.csv``: sites in YAML order, case ids sorted). ``dice`` is
    empty for the query cases that are not evaluated (no label, no valid Dice,
    or no majority-vote mask).
    """
    # Read with pandas' default float parser, as for the paper:
    # float_precision="round_trip" changes the last bit of some Dice values
    # and thus of the bootstrap results.
    return pd.read_csv(scores_dir / "cases.csv")


def load_scores(scores_dir: Path) -> pd.DataFrame:
    """
    Collect the failure scores from the stage outputs in
    ``scores_dir`` into one long table ``case_id, site, method, score``.

    Stages write ``<scores_dir>/scores/*.csv`` in this format.
    """
    parts = [pd.read_csv(p) for p in sorted((scores_dir / "scores").glob("*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=["case_id", "site", "method", "score"])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def aurc(confids: np.ndarray, risks: np.ndarray) -> float:
    """
    AURC for one sample. ``confids`` are oriented higher = lower predicted
    risk (i.e. the negated failure score). Ties keep input order (stable
    mergesort, then reversed).
    """
    confids = np.asarray(confids, dtype=float)
    risks = np.asarray(risks, dtype=float)
    n = len(risks)
    order = np.argsort(confids, kind="mergesort")[::-1]   # most confident first
    selective_risk = np.cumsum(risks[order]) / np.arange(1, n + 1, dtype=float)
    return float(np.mean(selective_risk))


def _percentile_ci(boot: List[float], point: float,
                   lo_pct: float, hi_pct: float) -> Dict[str, float]:
    """Percentile CI from bootstrap replicates; NaN bounds if too few finite reps."""
    arr = np.asarray([v for v in boot if np.isfinite(v)], dtype=float)
    if len(arr) < MIN_FINITE_REPS:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {"point": point,
            "ci_lo": float(np.percentile(arr, lo_pct)),
            "ci_hi": float(np.percentile(arr, hi_pct))}


def _bootstrap_pvalue(boot_diffs: List[float]) -> Tuple[float, int]:
    """
    Two-sided percentile-bootstrap p-value for H0: paired difference = 0,
    p = 2 * (min(#{d* <= 0}, #{d* >= 0}) + 1) / (B + 1)  (Davison & Hinkley
    1997, §4.2). Consistent with the percentile CI from the same replicates,
    so "CI excludes 0" iff p < alpha. The smallest value is 2 / (B + 1).
    Returns (p, B) with B the number of finite replicates.
    """
    arr = np.asarray([v for v in boot_diffs if np.isfinite(v)], dtype=float)
    B = int(len(arr))
    if B < MIN_FINITE_REPS:
        return float("nan"), B
    n_le = int(np.sum(arr <= 0.0))
    n_ge = int(np.sum(arr >= 0.0))
    return float(min(1.0, 2.0 * (min(n_le, n_ge) + 1) / (B + 1))), B


def paired_bootstrap(
    site_arrays: Dict[str, Dict],
    methods: Sequence[str],
    pairs: Sequence[Tuple[str, str]] = (),
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Dict:
    """
    Site-stratified, method-paired bootstrap of within-site and pooled E-AURC.

    site_arrays : {site: {"risks": (n,), "confids": {method: (n,)}}}, in the
                  site order that defines the RNG stream. ``confids`` are
                  oriented higher = lower predicted risk.
    pairs       : (method, ref) pairs for paired differences, method - ref.

    Returns {"sites", "methods": {m: {E_AURC_within_mean[_ci_lo|_ci_hi],
    E_AURC_pooled[_ci_lo|_ci_hi], per_site: {site: {A, Astar}}}},
    "diffs": {(m, r): {same six fields, *_pval, n_boot}}}.
    """
    alpha = 1.0 - confidence_level
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    sites = list(site_arrays)
    for site in sites:
        n = len(site_arrays[site]["risks"])
        if n < MIN_SITE_CASES:
            raise ValueError(f"site '{site}' has n={n} < {MIN_SITE_CASES} cases")
    risks = {s: np.asarray(site_arrays[s]["risks"], dtype=float) for s in sites}
    confids = {m: {s: np.asarray(site_arrays[s]["confids"][m], dtype=float)
                   for s in sites} for m in methods}

    # ── Point estimates ─────────────────────────────────────────────────────
    per_site = {m: {s: {"A": aurc(confids[m][s], risks[s]),
                        "Astar": aurc(-risks[s], risks[s])} for s in sites}
                for m in methods}
    risks_pool = np.concatenate([risks[s] for s in sites])
    astar_pool = aurc(-risks_pool, risks_pool)
    point_within, point_pooled = {}, {}
    for m in methods:
        point_within[m] = float(np.mean(
            [per_site[m][s]["A"] - per_site[m][s]["Astar"] for s in sites]))
        confids_pool = np.concatenate([confids[m][s] for s in sites])
        point_pooled[m] = aurc(confids_pool, risks_pool) - astar_pool

    # ── Bootstrap: one index draw per site per replicate, shared by methods ─
    boot_within = {m: [] for m in methods}
    boot_pooled = {m: [] for m in methods}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        idx = {}
        for s in sites:
            n = len(risks[s])
            idx[s] = rng.integers(0, n, n)
        risks_b = {s: risks[s][idx[s]] for s in sites}
        astar_b = {s: aurc(-risks_b[s], risks_b[s]) for s in sites}
        risks_pool_b = np.concatenate([risks_b[s] for s in sites])
        astar_pool_b = aurc(-risks_pool_b, risks_pool_b)
        for m in methods:
            confids_b = {s: confids[m][s][idx[s]] for s in sites}
            boot_within[m].append(float(np.mean(
                [aurc(confids_b[s], risks_b[s]) - astar_b[s] for s in sites])))
            confids_pool_b = np.concatenate([confids_b[s] for s in sites])
            boot_pooled[m].append(aurc(confids_pool_b, risks_pool_b) - astar_pool_b)

    # ── Assemble ────────────────────────────────────────────────────────────
    out_methods = {}
    for m in methods:
        cw = _percentile_ci(boot_within[m], point_within[m], lo_pct, hi_pct)
        cp = _percentile_ci(boot_pooled[m], point_pooled[m], lo_pct, hi_pct)
        out_methods[m] = {
            "E_AURC_within_mean":       cw["point"],
            "E_AURC_within_mean_ci_lo": cw["ci_lo"],
            "E_AURC_within_mean_ci_hi": cw["ci_hi"],
            "E_AURC_pooled":            cp["point"],
            "E_AURC_pooled_ci_lo":      cp["ci_lo"],
            "E_AURC_pooled_ci_hi":      cp["ci_hi"],
            "per_site":                 per_site[m],
        }

    out_diffs = {}
    for m, r in pairs:
        d_within = [a - b for a, b in zip(boot_within[m], boot_within[r])]
        d_pooled = [a - b for a, b in zip(boot_pooled[m], boot_pooled[r])]
        cw = _percentile_ci(d_within, point_within[m] - point_within[r], lo_pct, hi_pct)
        cp = _percentile_ci(d_pooled, point_pooled[m] - point_pooled[r], lo_pct, hi_pct)
        p_within, n_boot = _bootstrap_pvalue(d_within)
        p_pooled, _ = _bootstrap_pvalue(d_pooled)
        out_diffs[(m, r)] = {
            "E_AURC_within_mean":       cw["point"],
            "E_AURC_within_mean_ci_lo": cw["ci_lo"],
            "E_AURC_within_mean_ci_hi": cw["ci_hi"],
            "E_AURC_within_mean_pval":  p_within,
            "E_AURC_pooled":            cp["point"],
            "E_AURC_pooled_ci_lo":      cp["ci_lo"],
            "E_AURC_pooled_ci_hi":      cp["ci_hi"],
            "E_AURC_pooled_pval":       p_pooled,
            "n_boot":                   n_boot,
        }

    return {"sites": sites, "methods": out_methods, "diffs": out_diffs}


def scores_table(cases: pd.DataFrame, scores: pd.DataFrame,
                 query_sites: Sequence[str]) -> pd.DataFrame:
    """
    One row per query case (``cases.csv`` order): ``case_id, site, dice`` and
    one column per method (higher = higher predicted failure risk).
    """
    methods = [m for m in METHODS if m in set(scores["method"])]
    wide = scores[scores["method"].isin(methods)].pivot(
        index=["case_id", "site"], columns="method", values="score").reset_index()
    q = cases[cases["site"].isin(query_sites)]
    return q.merge(wide, on=["case_id", "site"], how="left")[
        ["case_id", "site", "dice", *methods]]


def evaluated_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """The cases that are evaluated: those with a Dice (order kept)."""
    return cases[cases["dice"].notna()]


def labelled_sites(cases: pd.DataFrame,
                   query_sites: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Query sites (in order) with at least MIN_SITE_CASES evaluated cases, and
    the other query sites grouped by why they are not evaluated.
    """
    n = evaluated_cases(cases).groupby("site").size()
    sites, skipped = [], {}
    for s in query_sites:
        k = int(n.get(s, 0))
        if k >= MIN_SITE_CASES:
            sites.append(s)
        elif not (cases["site"] == s).any():
            skipped.setdefault("no cases", []).append(s)
        elif k == 0:
            skipped.setdefault("no labelled cases", []).append(s)
        else:
            skipped.setdefault(f"fewer than {MIN_SITE_CASES} labelled cases",
                               []).append(f"{s} ({k})")
    return sites, skipped


def available_methods(scores: pd.DataFrame) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Methods present in ``scores`` (in METHODS order) and the paired tests among them."""
    present = set(scores["method"])
    methods = [m for m in METHODS if m in present]
    pairs = [(m, r) for m, r in PAIRED_TESTS if m in present and r in present]
    return methods, pairs


def complete_methods(cases: pd.DataFrame, scores: pd.DataFrame, query_sites: Sequence[str],
                     methods: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    The methods that score every evaluated case of ``query_sites``, and the
    others with the cases they miss (e.g. Atlas-RCA without a valid
    registration). Leaving out a method, never a case, keeps the case set
    and thus the bootstrap's random stream unchanged.
    """
    q = cases[cases["site"].isin(query_sites)]
    keys = set(zip(q["case_id"], q["site"]))
    kept, dropped = [], {}
    for m in methods:
        s = scores[(scores["method"] == m) & scores["score"].notna()]
        missing = keys - set(zip(s["case_id"], s["site"]))
        if missing:
            dropped[m] = sorted(f"{site}/{c}" for c, site in missing)
        else:
            kept.append(m)
    return kept, dropped


def evaluate(
    cases: pd.DataFrame,
    scores: pd.DataFrame,
    query_sites: Sequence[str],
    methods: Sequence[str] = tuple(METHODS),
    pairs: Sequence[Tuple[str, str]] = tuple(PAIRED_TESTS),
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Dict:
    """
    E-AURC of every method on the query sites, with paired tests.

    cases  : ``case_id, site, dice``; the row order within a site is the case
             order used by the bootstrap.
    scores : long table ``case_id, site, method, score`` (higher = riskier).

    Every method must score every query case: a missing score raises instead
    of silently shrinking the case set, which would change the bootstrap.
    """
    wide = scores[scores["method"].isin(methods)].pivot(
        index=["case_id", "site"], columns="method", values="score").reset_index()
    q = cases[cases["site"].isin(query_sites)].merge(
        wide, on=["case_id", "site"], how="left", validate="one_to_one")

    missing = {m: int(q[m].isna().sum()) if m in q else len(q) for m in methods}
    missing = {m: n for m, n in missing.items() if n}
    if missing or q["dice"].isna().any():
        raise ValueError(f"incomplete scores on query cases: {missing}, "
                         f"dice NaN: {int(q['dice'].isna().sum())}")

    site_arrays = {}
    for site in query_sites:
        s = q[q["site"] == site]
        site_arrays[site] = {
            "risks": 1.0 - s["dice"].values.astype(float),
            # failure score -> confidence; -(-x) == x keeps negated scores exact
            "confids": {m: -s[m].values.astype(float) for m in methods},
        }
    result = paired_bootstrap(site_arrays, methods, pairs,
                              n_bootstrap=n_bootstrap, seed=seed)
    result["site_n"] = {site: len(site_arrays[site]["risks"]) for site in query_sites}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

_FIELDS = ["E_AURC_pooled", "E_AURC_pooled_ci_lo", "E_AURC_pooled_ci_hi",
           "E_AURC_within_mean", "E_AURC_within_mean_ci_lo", "E_AURC_within_mean_ci_hi"]


def eaurc_table(result: Dict) -> pd.DataFrame:
    """One row per method, full precision, in METHODS order."""
    rows = [{"method": m, "label": METHODS.get(m, m),
             **{k: md[k] for k in _FIELDS}}
            for m, md in result["methods"].items()]
    return pd.DataFrame(rows)


def paired_tests_table(result: Dict) -> pd.DataFrame:
    """One row per paired test (method - ref), full precision."""
    rows = [{"method": m, "ref": r, **d} for (m, r), d in result["diffs"].items()]
    return pd.DataFrame(rows)


def print_tables(result: Dict) -> None:
    """Print E-AURC x100 as in the paper."""
    def fmt(pt, lo, hi, nd=1):
        return f"{100 * pt:.{nd}f} [{100 * lo:.{nd}f}, {100 * hi:.{nd}f}]"

    n = sum(result["site_n"].values())
    print(f"\nE-AURC (x1e-2, lower is better), n={n}, sites: {', '.join(result['sites'])}")
    print(f"{'Method':<17}{'Pooled':>22}{'Within-site':>22}")
    for m, md in result["methods"].items():
        print(f"{METHODS.get(m, m):<17}"
              f"{fmt(*(md[k] for k in _FIELDS[:3])):>22}"
              f"{fmt(*(md[k] for k in _FIELDS[3:])):>22}")

    print("\nPaired tests, Δ = method - ref (x1e-2; negative: method better)")
    for (m, r), d in result["diffs"].items():
        for scope, key in (("within", "E_AURC_within_mean"), ("pooled", "E_AURC_pooled")):
            print(f"  {METHODS.get(m, m)} vs {METHODS.get(r, r)} [{scope}]: "
                  f"Δ={fmt(d[key], d[key + '_ci_lo'], d[key + '_ci_hi'], 2)}  "
                  f"p={d[key + '_pval']:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", required=True, help="benchmark YAML")
    ap.add_argument("--scores_dir", required=True,
                    help="directory with the stage outputs (per-case score CSVs)")
    ap.add_argument("--out_dir", default=None,
                    help="where to write eaurc_table.csv and paired_tests.csv "
                         "(default: <scores_dir>/evaluation)")
    args = ap.parse_args(argv)

    with open(args.benchmark) as f:
        query_sites = list(yaml.safe_load(f)["query_sites"])
    scores_dir = Path(args.scores_dir)
    cases, scores = load_cases(scores_dir), load_scores(scores_dir)
    methods, pairs = available_methods(scores)
    if not methods:
        raise SystemExit(f"no failure scores in {scores_dir / 'scores'}")
    sites, skipped = labelled_sites(cases, query_sites)
    for reason, names in skipped.items():
        print(f"not evaluated ({reason}): {', '.join(names)}")
    if not sites:
        raise SystemExit("no labelled query site to evaluate")
    evaluated = evaluated_cases(cases)
    methods, dropped = complete_methods(evaluated, scores, sites, methods)
    for m, missing in dropped.items():
        print(f"WARNING {m} not evaluated: no score for {len(missing)} evaluated "
              f"case(s): {', '.join(missing)}")
    if not methods:
        raise SystemExit("no method scores every evaluated case")
    pairs = [(m, r) for m, r in pairs if m in methods and r in methods]
    result = evaluate(evaluated, scores, sites, methods, pairs)
    print_tables(result)

    out_dir = Path(args.out_dir) if args.out_dir else scores_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    eaurc_table(result).to_csv(out_dir / "eaurc_table.csv", index=False)
    paired_tests_table(result).to_csv(out_dir / "paired_tests.csv", index=False)
    print(f"\nSaved {out_dir / 'eaurc_table.csv'} and {out_dir / 'paired_tests.csv'}")


if __name__ == "__main__":
    main()
