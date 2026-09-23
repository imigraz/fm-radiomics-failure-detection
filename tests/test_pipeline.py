"""
End-to-end CLI runs on small synthetic benchmarks. The FM backbones and the
ANTs registration are replaced by fakes; everything else runs for real.
"""

import json

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from failure_scores import run
from failure_scores.baselines import atlas_rca
from failure_scores.fm import vit2d

N_FOLDS = 5
SHAPE = (16, 32, 32)   # z, y, x


def _ellipsoid(radii, center, shape=SHAPE):
    zz, yy, xx = np.indices(shape)
    d = sum(((g - c) / r) ** 2 for g, c, r in zip((zz, yy, xx), center, radii))
    return (d <= 1).astype(np.uint8)


def _write(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))


def _make_site(root, task, cases, rng, labelled, pred_folds, bilateral=False, empty=()):
    """
    Images, labels (``labelled``: bool or predicate on the case id) and fold
    predictions of one site. Bilateral volumes hold two ellipsoids along x;
    cases in ``empty`` are predicted empty by every fold.
    """
    (root / "datasets" / task / "imagesTr").mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        (root / "predictions" / f"fold_{k}" / f"Preds_{task}").mkdir(parents=True, exist_ok=True)
    centers = [(8, 16, 8), (8, 16, 24)] if bilateral else [(8, 16, 16)]
    for case in cases:
        parts = [(rng.uniform([3, 5, 3], [5, 9, 6]) if bilateral
                  else rng.uniform([3, 6, 5], [6, 12, 11]), c) for c in centers]
        gt = np.zeros(SHAPE, np.uint8)
        for radii, c in parts:
            gt |= _ellipsoid(radii, c)
        img = (100 + 20 * rng.standard_normal(SHAPE) + 80 * gt).astype(np.float32)
        _write(img, root / "datasets" / task / "imagesTr" / f"{case}_0000.nii.gz")
        if labelled is True or (callable(labelled) and labelled(case)):
            _write(gt, root / "datasets" / task / "labelsTr" / f"{case}.nii.gz")
        for k in pred_folds(case):
            pred = np.zeros(SHAPE, np.uint8)
            if case not in empty:
                for radii, c in parts:
                    pred |= _ellipsoid(radii * rng.uniform(0.7, 1.2, 3), c)
            _write(pred, root / "predictions" / f"fold_{k}" / f"Preds_{task}" / f"{case}.nii.gz")


def _build(root, sites, extra_yaml=""):
    """Reference R (15 cases) plus query sites (tag, cases, labelled, bilateral, empty)."""
    rng = np.random.default_rng(0)
    ref = [f"R{i:02d}" for i in range(15)]
    held_out = {c: i % N_FOLDS for i, c in enumerate(ref)}
    _make_site(root, "Task001_R", ref, rng, True, lambda c: [held_out[c]])
    for k in range(N_FOLDS):
        cases = {c: {"mask_1": {"Dice": 0.9}} for c, f in held_out.items() if f == k}
        with open(root / "predictions" / f"fold_{k}" / "folds.json", "w") as f:
            json.dump({"epoch_1": {"Task001_R": cases}}, f)
    dataset_dirs, pred_dirs = {"R": "Task001_R"}, {"R": "Preds_Task001_R"}
    for i, (tag, cases, labelled, bilateral, empty) in enumerate(sites):
        task = f"Task{100 + i}_{tag}"
        _make_site(root, task, cases, rng, labelled, lambda c: range(N_FOLDS), bilateral, empty)
        dataset_dirs[tag], pred_dirs[tag] = task, f"Preds_{task}"
    cfg = root / "bench.yaml"
    cfg.write_text(
        f"name: synth\nreference_site: R\nquery_sites: {json.dumps([s[0] for s in sites])}\n"
        f"dataset_dirs: {json.dumps(dataset_dirs)}\npred_dirs: {json.dumps(pred_dirs)}\n"
        "fold_json_name: folds.json\ntask_name_to_tag: {Task001_R: R}\n" + extra_yaml)
    return cfg


def _run(root, cfg, stage):
    run.main(["--benchmark", str(cfg), "--stage", stage, "--out_dir", str(root / "out"),
              "--data_root", str(root / "datasets"), "--pred_root", str(root / "predictions")])


@pytest.fixture
def fakes(monkeypatch):
    """Deterministic fake 2D FM and a registration that returns a fixed mask."""
    def embed(img, mask, processor, model, device):
        if not mask.any():
            return None
        rng = np.random.default_rng(int(mask.sum()))
        base = np.array([img.mean(), img.std(), mask.sum()], dtype=np.float32)
        return {m: np.concatenate([base, rng.standard_normal(6).astype(np.float32)])
                for m in ("img", "ov", "w", "ovw")}

    def register(moving_img, moving_mask, fixed_img, transform="SyNRA"):
        # An empty moving mask warps to an empty mask, as with ANTs.
        arr = fixed_img.numpy()
        warped = (arr > np.percentile(arr, 80)) & bool(moving_mask.numpy().any())
        return fixed_img.new_image_like(warped.astype("float32"))

    monkeypatch.setattr(vit2d, "load_model", lambda backbone: (None, None, "cpu"))
    monkeypatch.setattr(vit2d, "embed_case", embed)
    monkeypatch.setattr(atlas_rca, "register_and_warp", register)


@pytest.fixture
def benchmark(tmp_path):
    cfg = _build(tmp_path, [("A", [f"A{i}" for i in range(6)], True, False, ()),
                            ("B", [f"B{i}" for i in range(5)], False, False, ())])
    return tmp_path, cfg


def test_labelled_and_unlabelled_query_sites(benchmark):
    root, cfg = benchmark
    _run(root, cfg, "extract")
    _run(root, cfg, "evaluate")
    out = root / "out"

    cases = pd.read_csv(out / "cases.csv")
    assert cases.groupby("site", sort=False).size().to_dict() == {"R": 15, "A": 6, "B": 5}
    assert cases.loc[cases.site == "A", "dice"].notna().all()
    assert cases.loc[cases.site == "B", "dice"].isna().all()

    table = pd.read_csv(out / "failure_scores.csv")
    assert list(table.columns) == ["case_id", "site", "dice", "rad_md_mask"]
    assert len(table) == 11 and table["rad_md_mask"].notna().all()

    # Evaluated on the labelled query site only.
    eaurc = pd.read_csv(out / "evaluation" / "eaurc_table.csv")
    assert eaurc["method"].tolist() == ["rad_md_mask"]
    assert np.isfinite(eaurc["E_AURC_pooled"]).all()


def test_no_labelled_query_site_skips_evaluation(benchmark):
    root, cfg = benchmark
    for p in (root / "datasets" / "Task100_A" / "labelsTr").iterdir():
        p.unlink()
    _run(root, cfg, "extract")
    _run(root, cfg, "evaluate")
    assert len(pd.read_csv(root / "out" / "failure_scores.csv")) == 11
    assert not (root / "out" / "evaluation").exists()


def test_all_stages_on_mixed_sites(tmp_path, fakes):
    """
    --stage all with the default FM models (no 3DINO) on a labelled site (A),
    an unlabelled site with an empty prediction (B), a partially labelled site
    (C), a bilateral unlabelled site (D), a labelled site with only 3 cases (S)
    and a site without cases (Z).
    """
    cfg = _build(tmp_path, [
        ("A", [f"A{i}" for i in range(6)], True, False, ()),
        ("B", [f"B{i}" for i in range(5)], False, False, ("B2",)),
        ("C", [f"C{i}" for i in range(7)], lambda c: c[-1] in "0246", False, ()),
        ("D", [f"D{i}" for i in range(3)], False, True, ()),
        ("S", [f"S{i}" for i in range(3)], True, False, ()),
        ("Z", [], False, False, ()),
    ], "bilateral_split_sites: [D]\nbilateral_split_axis: 0\n"
       "atlas_flip_r_half: true\n")
    _run(tmp_path, cfg, "all")
    out = tmp_path / "out"

    # Every query case with an image has a row; only evaluated cases have a Dice.
    table = pd.read_csv(out / "failure_scores.csv").set_index("case_id")
    assert table.groupby("site", sort=False).size().to_dict() == \
        {"A": 6, "B": 5, "C": 7, "D": 6, "S": 3}
    assert table.groupby("site", sort=False)["dice"].count().to_dict() == \
        {"A": 6, "B": 0, "C": 4, "D": 0, "S": 3}
    methods = ["ens_mpd", "atlas_rca", "rad_l2", "rad_md_img", "rad_md_mask",
               *[f"{b}_md_{m}" for b in ("dinov3", "raddino") for m in ("img", "ov", "w", "ovw")]]
    assert list(table.columns) == ["site", "dice", *methods]

    # Unlabelled cases of the partially labelled site are scored.
    assert table.loc[["C1", "C3", "C5"], methods].notna().all().all()
    # Bilateral halves get every score.
    assert sorted(table[table.site == "D"].index) == [f"D{i}_{s}" for i in range(3) for s in "LR"]
    assert table.loc[table.site == "D", methods].notna().all().all()
    # The empty prediction is kept: no mask-based score, but the image-based ones.
    mask_based = ["atlas_rca", "rad_md_mask", *[m for m in methods if "_md_" in m and "dino" in m]]
    assert table.loc["B2", mask_based].isna().all()
    assert table.loc["B2", ["rad_l2", "rad_md_img"]].notna().all()

    # Evaluated on A and C only (S has 3 labelled cases, B / D none, Z no cases).
    eaurc = pd.read_csv(out / "evaluation" / "eaurc_table.csv")
    assert eaurc["method"].tolist() == methods
    assert np.isfinite(eaurc[["E_AURC_pooled", "E_AURC_within_mean"]]).all().all()


def test_missing_fold_json_fails_before_writing(benchmark):
    root, cfg = benchmark
    for p in (root / "predictions").glob("fold_*/folds.json"):
        p.unlink()
    with pytest.raises(FileNotFoundError, match="fold JSONs"):
        _run(root, cfg, "extract")
    assert not (root / "out").exists()


def test_evaluate_without_scores_fails(benchmark):
    root, cfg = benchmark
    (root / "out").mkdir()
    pd.DataFrame({"case_id": ["A0"], "site": ["A"], "dice": [0.9]}).to_csv(
        root / "out" / "cases.csv", index=False)
    with pytest.raises(FileNotFoundError, match="no failure scores"):
        _run(root, cfg, "evaluate")


def test_images_and_labels_folders(benchmark):
    """``images/`` + ``labels/`` are accepted like ``imagesTr/`` + ``labelsTr/``."""
    root, cfg = benchmark
    for task in ("Task001_R", "Task100_A"):
        d = root / "datasets" / task
        (d / "imagesTr").rename(d / "images")
        (d / "labelsTr").rename(d / "labels")
    _run(root, cfg, "extract")
    cases = pd.read_csv(root / "out" / "cases.csv")
    assert cases.groupby("site", sort=False)["dice"].count().to_dict() == {"R": 15, "A": 6, "B": 0}
