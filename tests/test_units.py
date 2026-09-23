"""Unit tests that need no data."""

import warnings

import numpy as np
import pandas as pd
import pytest

import SimpleITK as sitk

from failure_scores import evaluation as ev, io, masks
from failure_scores.baselines import atlas_rca, ens_mpd
from failure_scores.fm import pooling, render, vit2d
from failure_scores.scoring import MahalanobisScorer


def test_aurc_oracle_and_ties():
    risks = np.array([0.1, 0.9, 0.3, 0.5])
    # Oracle: most confident = lowest risk.
    oracle = ev.aurc(-risks, risks)
    expected = np.mean(np.cumsum(np.sort(risks)) / np.arange(1, 5))
    assert oracle == pytest.approx(expected, abs=0)
    # All tied: stable mergesort reversed -> last case ranked first.
    tied = ev.aurc(np.zeros(4), risks)
    expected = np.mean(np.cumsum(risks[::-1]) / np.arange(1, 5))
    assert tied == expected


def test_evaluate_refuses_missing_scores():
    """A missing score must fail loudly, never shrink the case set."""
    cases = pd.DataFrame({"case_id": list("abcdefgh"), "site": ["A"] * 4 + ["B"] * 4,
                          "dice": np.linspace(0.5, 0.9, 8)})
    scores = pd.DataFrame({"case_id": list("abcdefg"), "site": ["A"] * 4 + ["B"] * 3,
                           "method": "m", "score": np.arange(7.0)})
    with pytest.raises(ValueError, match="incomplete"):
        ev.evaluate(cases, scores, ["A", "B"], methods=["m"], pairs=[], n_bootstrap=10)


def test_evaluate_available_methods_subset():
    """Only the methods that were run are evaluated; paired tests need both methods."""
    rng = np.random.default_rng(0)
    cases = pd.DataFrame({"case_id": [f"c{i}" for i in range(12)], "site": ["A"] * 6 + ["B"] * 6,
                          "dice": rng.random(12)})
    scores = pd.concat([cases.assign(method=m, score=rng.random(12))[["case_id", "site", "method", "score"]]
                        for m in ("rad_md_mask", "dinov3_md_ovw")])
    methods, pairs = ev.available_methods(scores)
    assert methods == ["rad_md_mask", "dinov3_md_ovw"]
    assert pairs == [("dinov3_md_ovw", "rad_md_mask")]
    res = ev.evaluate(cases, scores, ["A", "B"], methods, pairs, n_bootstrap=20)
    assert set(res["methods"]) == set(methods) and len(res["diffs"]) == 1
    assert len(ev.eaurc_table(res)) == 2 and len(ev.paired_tests_table(res)) == 1

    # All methods present: the same call as the paper's evaluation.
    full = pd.DataFrame({"method": list(ev.METHODS)})
    assert ev.available_methods(full) == (list(ev.METHODS), list(ev.PAIRED_TESTS))


def test_complete_methods_drops_methods_not_cases():
    """A method missing a score on an evaluated case is left out, with the case named."""
    cases = pd.DataFrame({"case_id": list("abcdefgh"), "site": ["A"] * 4 + ["B"] * 4,
                          "dice": np.linspace(0.5, 0.9, 8)})
    full = cases.assign(method="m", score=np.arange(8.0))
    partial = cases.assign(method="atlas_rca", score=[np.nan] + list(np.arange(7.0)))
    extra = pd.DataFrame({"case_id": ["z"], "site": ["C"], "method": "n", "score": [1.0]})
    scores = pd.concat([full, partial.iloc[1:], extra])
    kept, dropped = ev.complete_methods(cases, scores, ["A", "B"], ["m", "atlas_rca", "n"])
    assert kept == ["m"]
    assert dropped == {"atlas_rca": ["A/a"], "n": sorted(f"{s}/{c}" for c, s in
                                                         zip(cases.case_id, cases.site))}
    # A NaN score counts as missing, too.
    kept, dropped = ev.complete_methods(cases, pd.concat([full, partial]), ["A", "B"],
                                        ["m", "atlas_rca"])
    assert kept == ["m"] and dropped == {"atlas_rca": ["A/a"]}


def test_negated_scores_are_bitwise_confidences():
    """Storing quality q as score -q gives confidence -(-q), identical to q."""
    q = np.random.default_rng(0).random(1000)
    assert np.array_equal((-(-q)).view(np.uint64), q.view(np.uint64))


def test_benchmark_config(tmp_path):
    from failure_scores.config import load

    cfg = load("benchmarks/hip_msd.yaml", data_root="/data", out_root="/out")
    assert cfg.sites == ["DecathHip", "Dryad", "HarP"]          # YAML order
    assert cfg.bilateral_axis("HarP") == 0 and cfg.bilateral_axis("DecathHip") is None
    assert str(cfg.dataset_dirs["Dryad"]) == "/data/Task098_Dryad"
    assert str(cfg.out_dir) == "/out/hip_msd" and cfg.atlas_flip_r_half
    assert load("benchmarks/prostate_msd.yaml").query_sites == ("BMC", "I2CVB", "UCL", "BIDMC", "HK")

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nreference_site: A\nquery_sites: [B]\n"
                   "dataset_dirs: {A: a}\npred_dirs: {A: a, B: b}\n"
                   "task_name_to_tag: {}\nfold_json_name: f.json\n")
    with pytest.raises(ValueError, match="dataset_dirs"):
        load(bad)


# ── Majority vote and Dice ────────────────────────────────────────────────────

def _write(arr, path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))
    return path


def test_majority_vote_and_dice(tmp_path):
    rng = np.random.default_rng(0)
    folds = [(rng.random((6, 7, 8)) > 0.5).astype(np.uint8) for _ in range(5)]
    paths = [_write(f, tmp_path / f"f{i}.nii.gz") for i, f in enumerate(folds)]
    mv = sitk.GetArrayFromImage(masks.majority_vote(paths))
    assert np.array_equal(mv, (np.sum(folds, axis=0) >= 3).astype(np.uint8))

    gt = _write(folds[0], tmp_path / "gt.nii.gz")
    a, b = folds[0].astype(bool), folds[1].astype(bool)
    assert masks.dice_label1(paths[1], gt) == 2 * (a & b).sum() / (a.sum() + b.sum())
    empty = _write(np.zeros((6, 7, 8)), tmp_path / "empty.nii.gz")
    assert np.isnan(masks.dice_label1(empty, gt))



def test_bilateral_split_keeps_anisotropic_masks(tmp_path):
    """Mask halves are resampled with nearest neighbour, not B-spline."""
    zz, yy, xx = np.indices((20, 40, 64))
    mask = ((((zz - 10) / 6) ** 2 + ((yy - 20) / 12) ** 2 + ((xx - 16) / 10) ** 2) <= 1)
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing((0.7, 0.7, 3.0))
    src = tmp_path / "mask.nii.gz"
    sitk.WriteImage(img, str(src))

    io.split_nifti_half(src, tmp_path / "L.nii.gz", 2, "L", is_mask=True)
    half = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "L.nii.gz")))
    nn = sitk.GetArrayFromImage(io.resample_to_isotropic(img, img)[1])
    assert np.array_equal(half, io.split_array_half(nn, 2, "L"))
    assert set(np.unique(half)) == {0, 1}
    # At 1 mm the split does not resample: the half is the input half.
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(src))
    io.split_nifti_half(src, tmp_path / "L1.nii.gz", 2, "L", is_mask=True)
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "L1.nii.gz"))),
                          io.split_array_half(mask.astype(np.uint8), 2, "L"))

# ── FM rendering and pooling ──────────────────────────────────────────────────

def test_overlay_pixels():
    img = np.tile(np.linspace(0, 100, 6, dtype=np.float32), (4, 1))   # 4x6 slice
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    out = np.asarray(render.overlay_rgb(img, mask))
    assert out.shape == (6, 6, 3)                       # square pad, image top-left
    assert not out[4:].any()                            # padded rows are 0
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    gray = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8).astype(np.float32)
    for r, c in [(1, 2), (2, 3)]:                       # inside: alpha 0.4, RGB (255, 80, 0)
        expected = np.clip([0.6 * gray[r, c] + 0.4 * 255, 0.6 * gray[r, c] + 0.4 * 80,
                            0.6 * gray[r, c] + 0.4 * 0], 0, 255).astype(np.uint8)
        assert np.array_equal(out[r, c], expected)
    assert np.array_equal(out[0, 5], [gray[0, 5]] * 3)  # outside: plain gray


def test_patch_weights_inter_area_and_threshold():
    mask = np.zeros((28, 28), dtype=np.uint8)
    mask[:14, :14] = 1          # patch (0, 0) fully inside
    mask[14:16, 14:28] = 1      # patch (1, 1) 1/7 covered
    mask[27, 0] = 1             # patch (1, 0) 1/196 covered
    cov = pooling.patch_weights_2d(mask, 2)
    np.testing.assert_allclose(cov, [1.0, 0.0, 1 / 196, 2 / 14], rtol=1e-6)
    tokens = np.arange(8, dtype=np.float32).reshape(4, 2)
    pooled = pooling.pool_patches_2d(tokens, cov)       # 1/196 < 0.10 is dropped
    w = np.array([1.0, 0, 0, 2 / 14]) / (1 + 2 / 14)
    np.testing.assert_allclose(pooled, (tokens * w[:, None]).sum(0), rtol=1e-6)
    assert pooling.PATCH_COV_THR == 0.10
    # Empty mask -> plain mean of the patch tokens.
    assert np.array_equal(pooling.pool_patches_2d(tokens, np.zeros(4, np.float32)),
                          tokens.mean(0))


@pytest.mark.parametrize("n_reg", [4, 0])   # DINOv3, RadDINO
def test_register_tokens_skipped(n_reg):
    tokens = np.arange((1 + n_reg + 9) * 2).reshape(-1, 2)
    cls, patches = vit2d.split_tokens(tokens, n_reg)
    assert np.array_equal(cls, tokens[0])
    assert np.array_equal(patches, tokens[1 + n_reg:]) and len(patches) == 9


def test_3d_patch_weights_c_order():
    """Weight index = C-order (D, H, W) index of the patch, as the 3D ViT flattens."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[:8, 8:, :8] = True                  # patch (d=0, h=1, w=0) on a 2³ grid
    cov = pooling.patch_weights_3d(mask, 2)
    assert cov.argmax() == np.ravel_multi_index((0, 1, 0), (2, 2, 2))
    assert cov[np.ravel_multi_index((0, 1, 0), (2, 2, 2))] == pytest.approx(1.0)
    assert cov.sum() == pytest.approx(1.0)


# ── Baselines ─────────────────────────────────────────────────────────────────

def test_mean_pairwise_dice():
    a = np.zeros((4, 4), np.uint8); a[:2] = 1
    b = np.zeros((4, 4), np.uint8); b[:1] = 1
    c = np.zeros((4, 4), np.uint8)
    assert ens_mpd.dice(a, b) == 2 * 4 / (8 + 4)
    assert np.isnan(ens_mpd.dice(c, c))
    # nanmean over the 3 pairs: (a,b)=2/3, (a,c)=0, (b,c)=0
    assert ens_mpd.mean_pairwise_dice([a, b, c]) == pytest.approx((2 / 3) / 3)
    assert np.isnan(ens_mpd.mean_pairwise_dice([a]))
    empty = np.zeros_like(a)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no "Mean of empty slice"
        assert np.isnan(ens_mpd.mean_pairwise_dice([empty, empty, empty]))


def test_2d_slice_selection(monkeypatch):
    """Slices with >= MIN_MASK_AREA mask pixels; all non-empty slices if none has."""
    seen = []
    def fake_slice(img_sl, mask_sl, processor, model, device):
        seen.append(int(mask_sl.sum()))
        return {m: np.full(2, float(mask_sl.sum())) for m in ("img", "ov", "w", "ovw")}
    monkeypatch.setattr(vit2d, "embed_slice", fake_slice)
    img = np.zeros((8, 8, 4), np.float32)
    mask = np.zeros((8, 8, 4), np.uint8)
    mask[:4, :3, 0] = 1                                   # 12 pixels
    mask[:2, :2, 2] = 1                                   # 4 pixels
    e = vit2d.embed_case(img, mask, None, None, "cpu")
    assert seen == [12] and e["ovw"][0] == 12.0           # the small slice is left out
    seen.clear()
    mask[:, :, 0] = 0
    mask[0, :3, 1] = 1                                    # 3 pixels
    e = vit2d.embed_case(img, mask, None, None, "cpu")
    assert seen == [3, 4]                                 # fallback: every non-empty slice
    assert e["ovw"][0] == pytest.approx((3 * 3 + 4 * 4) / 7)   # area-weighted
    assert vit2d.embed_case(img, np.zeros_like(mask), None, None, "cpu") is None


def test_atlas_rca_skips_empty_masks(tmp_path, monkeypatch):
    """An empty mask is skipped before any registration."""
    def no_registration(*args, **kwargs):
        raise AssertionError("registered an empty mask")
    monkeypatch.setattr(atlas_rca, "register_and_warp", no_registration)
    empty = tmp_path / "empty.nii.gz"
    # Volume-sized, so that reading freed image memory would show up.
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((24, 80, 96), np.uint8)), str(empty))
    case = {"case_id": "c0", "img_path": empty, "pred_path": empty}
    ref = {"case_id": "r0", "img_path": empty, "gt_path": empty}
    for _ in range(20):
        out = atlas_rca.score_site("A", [case], [ref], tmp_path / "cache", flip_right=False)
        assert out.empty and not (tmp_path / "cache").exists()


def test_shape_skips_masks_too_small_for_pyradiomics(tmp_path):
    """Single-voxel and line masks get no shape features instead of stopping the stage."""
    from failure_scores import shape
    site_dir, mv_dir = tmp_path / "Task001_A", tmp_path / "mv"
    (site_dir / "imagesTr").mkdir(parents=True)
    mv_dir.mkdir()
    rng = np.random.default_rng(0)
    masks_ = {"voxel": (slice(8, 9), slice(10, 11), slice(10, 11)),
              "line": (slice(8, 9), slice(10, 11), slice(8, 14)),
              "blob": (slice(6, 10), slice(8, 14), slice(8, 14))}
    for case, sl in masks_.items():
        sitk.WriteImage(sitk.GetImageFromArray(rng.random((16, 24, 24)).astype(np.float32)),
                        str(site_dir / "imagesTr" / f"{case}_0000.nii.gz"))
        m = np.zeros((16, 24, 24), np.uint8)
        m[sl] = 1
        sitk.WriteImage(sitk.GetImageFromArray(m), str(io.mv_mask_path(mv_dir, "A", case)))
    feats = shape.extract_site("A", site_dir, False, mv_dir, shape.build_extractor())
    assert feats["case_id"].tolist() == ["blob"]


def test_frd_keep_columns():
    """Columns that are not finite on the reference are dropped."""
    from failure_scores.baselines import frd
    ref = np.array([[1.0, 2.0, np.nan], [2.0, 3.0, 1.0], [3.0, 5.0, 2.0]], dtype=np.float32)
    assert frd.keep_columns(ref).tolist() == [True, True, False]


def test_warped_dice():
    w = np.array([[0, 1, 1], [0, 2, 0]])
    g = np.array([[0, 1, 0], [0, 2, 2]])
    assert atlas_rca.warped_dice(w, g) == 2 * 1 / (2 + 1)      # label 1 only
    assert np.isnan(atlas_rca.warped_dice(np.zeros((2, 2)), g))


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_rejects_unknown_norm():
    with pytest.raises(ValueError):
        MahalanobisScorer("none")
