# Foundation Model and Radiomics Distance Scores for Post-Hoc Segmentation Failure Detection

Code for the MICCAI UNSURE 2026 paper:

> S. J. Joham, G. Guglielmo, M. Kozinski, M. Urschler.
> *Foundation Model and Radiomics Distance Scores for Post-Hoc Segmentation
> Failure Detection.* UNSURE workshop, MICCAI 2026.

Video: a short overview of the work at the UNSURE workshop, on
[YouTube](https://www.youtube.com/watch?v=DMRhocp0Egw).

A segmentation model is audited after deployment, without labels: each test
case gets a **failure score** (higher = higher predicted risk, risk = 1 − Dice).
Both proposed scores are the Mahalanobis distance of the predicted case to a
labelled reference set (the model's training site):

- **Rad-MD-Mask**: PyRadiomics 3D shape features of the predicted mask,
  z-scored, Ledoit-Wolf Mahalanobis distance.
- **DINOv3-MD-OvW**: frozen DINOv3 embeddings of the image with the predicted
  mask overlaid, patch tokens pooled with mask-coverage weights, Mahalanobis++
  (L2-normalised embeddings, Ledoit-Wolf).

Detectors are compared by **E-AURC**, pooled over all query cases and
within-site (mean over sites), with site-stratified paired bootstrap CIs.

---

## Methods and terminology

| Method | method key | Feature vector | Scorer |
|---|---|---|---|
| Ens-Mpd ([Roy et al. 2019](https://doi.org/10.1016/j.neuroimage.2019.03.042)) | `ens_mpd` | mean pairwise Dice of the 5 fold masks (negated) | – |
| Atlas-RCA ([Valindria et al. 2017](https://doi.org/10.1109/TMI.2017.2665165)) | `atlas_rca` | max Dice of the mask registered (ANTs SyNRA) to every reference (negated) | – |
| Rad-L2 ([Konz et al. 2026](https://doi.org/10.1016/j.media.2026.103943)) | `rad_l2` | FRD v1 whole-image radiomics | L2 of the reference-z-scored vector |
| Rad-MD-Img (ours) | `rad_md_img` | FRD v1 whole-image radiomics ([Konz et al. 2026](https://doi.org/10.1016/j.media.2026.103943)) | Mahalanobis++ |
| Rad-MD-Mask (ours) | `rad_md_mask` | 3D shape features of the mask | z-score + Mahalanobis |
| {DINOv3, RadDINO, 3DINO}-MD-{Img, Ov, W, OvW} (ours) | `{dinov3,raddino,3dino}_md_{img,ov,w,ovw}` | FM embedding: [DINOv3](https://arxiv.org/abs/2508.10104), [RadDINO](https://doi.org/10.1038/s42256-024-00965-w), [3DINO](https://doi.org/10.1038/s41746-025-02035-w) | Mahalanobis++ |

Mahalanobis++ ([Müller & Hein 2025](https://arxiv.org/abs/2505.18032)): Mahalanobis
distance on L2-normalised features; here with a Ledoit-Wolf covariance, as for
Rad-MD-Mask.

FM modes: **Img** CLS token of the image; **Ov** CLS token of the image with
the mask overlaid (α = 0.4, RGB 255/80/0; in 3D: mask voxels +0.5); **W**
patch tokens of the image, averaged with mask-coverage weights (2D: patches
less than 10 % covered get weight 0); **OvW** the same on the overlay image. The 2D models embed every axial slice with ≥ 10
mask pixels (every slice with any mask pixel if none has 10) and average the
slices weighted by mask area.

The reference is fit on the reference site's ground-truth masks; query cases
are scored on the majority vote (≥ 3 of 5) of the nnU-Net fold predictions.
Reference cases are scored leave-one-out (they are not part of the
evaluation). The per-case Dice used as risk is, for query cases, the mean over
the folds of each fold's Dice, and for reference cases the Dice of the fold
that held the case out.

---

## Installation

```bash
conda create -n seg_qc python=3.10
conda activate seg_qc
python -m pip install -r requirements.txt
```

DINOv3 is gated on Hugging Face: accept its license on the model page and log
in with `hf auth login` (the token must allow access to gated repos).

**3DINO** is optional and not run by default (`--fm_models` defaults to
`dinov3 raddino`). It needs `numpy<2`, so it runs in a second environment that
only does the 3DINO embeddings; everything else runs in `seg_qc`:

```bash
git clone https://github.com/AICONSlab/3DINO
git -C 3DINO checkout 85bd4435c1b2ada41cd34cd15cad17c4d3c88d89   # pinned 3DINO version
# weights (gated, access is granted automatically): https://huggingface.co/AICONSlab/3DINO-ViT
hf download AICONSlab/3DINO-ViT 3dino_vit_weights.pth --local-dir 3DINO   # in seg_qc
conda create -n seg_qc_3dino python=3.10
conda activate seg_qc_3dino
# the parts of 3DINO's requirements needed for inference
python -m pip install torch==2.0.0 torchvision==0.15.0 xformers==0.0.18 omegaconf fvcore iopath \
    --extra-index-url https://download.pytorch.org/whl/cu117
# what this package needs for the FM stage
python -m pip install "numpy<2" pandas==2.2.3 scipy==1.15.3 scikit-learn==1.7.2 \
    simpleitk==2.5.5 PyYAML==6.0.3
export THREEDINO_CODE_DIR=$PWD/3DINO
python -m failure_scores.run --benchmark benchmarks/<your>.yaml --stage fm --fm_models 3dino
conda activate seg_qc                                    # back to the main environment
python -m failure_scores.run --benchmark benchmarks/<your>.yaml --stage evaluate
```

Run the `extract` stage first (in `seg_qc`): 3DINO embeds the majority-vote
masks it writes. The 3DINO scores go to `scores/fm_3dino.csv` next to the
others and are picked up by `evaluate`. 3DINO's README states Python 3.9;
Python 3.10 with the packages above works. The full `requirements.txt` of
3DINO (cuML, MONAI, ...) is only needed for its training code.

## Using your own data

You need a binary segmentation model trained with 5-fold cross-validation
(e.g. nnU-Net), a labelled **reference site** (the model's training data) and
one or more **query sites** to score. Put the data in nnU-Net layout:

```
datasets/<Task>/imagesTr/<case>_0000.nii.gz       images, every site
datasets/<Task>/labelsTr/<case>.nii.gz            labels (structure = label 1)
predictions/fold_<k>/<Preds_Task>/<case>.nii.gz   k = 0..4, predictions of fold k
predictions/fold_<k>/<fold_json_name>             reference cases fold k validated on
```

- **Reference site:** images, ground-truth labels, and each fold's
  predictions of the cases it held out (the cross-validation predictions).
  The fold JSON says which fold held out which case; the last top-level key
  is used: `{"<epoch>": {"<task name>": {"<case>": {...}, ...}}}`. List
  every reference case there: a case missing from the fold JSONs has no Dice
  and is left out of Rad-MD-Mask's reference, while the FM and Rad-L2 /
  Rad-MD-Img references use every labelled reference image.
- **Query sites:** images and the predictions of all five folds for every
  case. The fold predictions are combined by majority vote (≥ 3 of 5), which is
  the mask that gets scored.
- **Labels on the query sites are optional.** Every query case with an image
  gets failure scores. Cases with a label (`labelsTr/`, or `labels/`) also
  get their Dice (the risk) and are evaluated (E-AURC); a query site is
  evaluated when it has at least 4 such cases. Labelled, unlabelled and
  partially labelled sites can be mixed.
- **Empty predictions.** A case whose majority-vote mask is empty (the model
  found nothing) is named in a warning and kept in `failure_scores.csv`; the
  methods that need the mask give it no score, and it is not evaluated. The
  same holds for a mask too small for shape features (a single voxel or a
  line of voxels), except that the other mask-based methods still score it.
- **Incomplete scores.** A method that cannot score an evaluated case (e.g.
  Atlas-RCA when every registration of the case failed) is left out of the
  evaluation with a warning naming the cases; the other methods are evaluated
  on the full case set.

Copy `benchmarks/template.yaml`, fill in your site and folder names, and run

```bash
python -m failure_scores.run --benchmark benchmarks/<your>.yaml --stage all
```

Which methods need what: Rad-MD-Mask uses only the masks (majority-vote masks
for query cases, reference labels for the reference); the image files must
still be present, since cases are found through them and PyRadiomics reads
their geometry, but their intensities are not used. The FM scores need the
images, the majority-vote masks and the reference labels; Ens-Mpd needs the
five fold masks; Rad-L2 / Rad-MD-Img only the images; Atlas-RCA the query
images and majority-vote masks and the reference images and labels. The evaluation uses whichever methods have been run, so,
for example, `--stage extract` followed by `--stage fm --fm_models dinov3` and
`--stage evaluate` evaluates Rad-MD-Mask and the four DINOv3 modes.

## Running

Stages, in order (`--stage extract|fm|baselines|evaluate|all`); outputs go to
`outputs/<name>/`:

| Stage | What | Output |
|---|---|---|
| `extract` | majority-vote masks, Dice, shape features, Rad-MD-Mask | `mv_masks/`, `cases.csv`, `shape_raw.csv`, `shape_features.csv`, `scores/shape.csv` |
| `fm` | FM embeddings (GPU) and -MD scores | `fm_cache/`, `scores/fm_<backbone>.csv` |
| `baselines` | Ens-Mpd, Rad-L2, Rad-MD-Img, Atlas-RCA | `scores/*.csv`, `frd_cache/`, `atlas_rca_cache/` |
| `evaluate` | all scores in one table; E-AURC and paired tests on the labelled query sites | `failure_scores.csv`, `evaluation/eaurc_table.csv`, `evaluation/paired_tests.csv` |

`failure_scores.csv` has one row per query case (`case_id, site, dice` and one
column per method; higher = higher predicted failure risk); the stages write
the same scores to `scores/*.csv` as `case_id, site, method, score`. Embeddings, features and
registrations are cached, so an interrupted run resumes where it stopped.
The caches are not invalidated: after changing images, labels or predictions,
delete `outputs/<name>/` (or use a new `--out_dir`) and run again.

Atlas-RCA registers every query case to every reference case with ANTs SyNRA
(prostate: 86 × 30 registrations; hours for larger reference sets).

### The paper's prostate benchmark

`predictions/` contains the five-fold nnU-Net predictions used in the paper for
the multi-site prostate benchmark (sites RUNMC, BMC, I2CVB, UCL, BIDMC, HK;
RUNMC is the reference). The images and labels are not included: use the
multi-site prostate setup of Liu et al. (MICCAI 2020, SAML) in the
preprocessed version of Gao et al. (MICCAI 2024, DeSAM), placed under
`datasets/` with the folder names in `benchmarks/prostate_msd.yaml`, then run

```bash
python -m failure_scores.run --benchmark benchmarks/prostate_msd.yaml --stage all
```

This runs every method except 3DINO. For the 3DINO scores, run
`--stage fm --fm_models 3dino` in the 3DINO environment (see Installation) and
then `--stage evaluate` again.

`benchmarks/hip_msd.yaml` is the paper's hippocampus benchmark (bilateral
volumes, each hippocampus scored as its own case); its predictions are not
included.

## Code layout

```
failure_scores/
  config.py       benchmark YAML -> Benchmark
  io.py           loading, 1 mm resampling, bilateral split, R->L flip
  masks.py        majority-vote masks, Dice (risk)
  scoring.py      MahalanobisScorer(norm="l2" | "zscore"), leave-one-out
  shape.py        Rad-MD-Mask: extraction, feature selection, score
  fm/             render.py, pooling.py, vit2d.py, dino3d.py, extract.py
  baselines/      ens_mpd.py, frd.py, atlas_rca.py
  evaluation.py   AURC, E-AURC, paired site-stratified bootstrap
  run.py          CLI
```

## Notes

**Atlas-RCA is not deterministic.** ANTs SyNRA samples voxels at random for
its mutual-information metric; even with a fixed seed and one thread the same
pair of images gives a different Dice from run to run (e.g. 0.49–0.53), so
Atlas-RCA scores vary between runs. A registration that fails (an exception
or an empty warped mask) is stored as NaN and skipped when taking the maximum
over the references.

**Bootstrap.** The confidence intervals and p-values use a site-stratified
paired bootstrap (2000 replicates, seed 42). The random stream depends on the
case set, the case order within a site and the order of `query_sites`.

### Shape feature selection

The shape features are selected on the reference site only: drop
near-constant features (coefficient of variation < 0.01) and heavy-tailed
ones (excess kurtosis > 5), rank the rest by stability (1/std), and greedily
drop the feature with the higher mean |r| until no pair has |r| > 0.90.
Selected: prostate 6 (Elongation, Flatness, Maximum2DDiameterRow,
Maximum2DDiameterSlice, Sphericity, SurfaceVolumeRatio), hippocampus 11.

## Maintenance

This is research code, released to reproduce the paper and to let others apply
the scores to their own data. It is maintained by a PhD student alongside
ongoing research, so maintenance is sporadic: issues and pull requests are
welcome, but replies may take a while, and there is no guarantee of fixes,
new features or support for newer package versions. The pinned versions in
`requirements.txt` are the tested ones.

## License

The code is licensed under the [Apache License 2.0](LICENSE). The prostate
predictions in `predictions/` are derived from public datasets, which have
their own terms, and the pretrained models (DINOv3, RadDINO, 3DINO) are
downloaded separately under their own licenses.

## Citation

```bibtex
@inproceedings{joham2026foundation,
  title     = {Foundation Model and Radiomics Distance Scores for Post-Hoc
               Segmentation Failure Detection},
  author    = {Joham, Simon Johannes and Guglielmo, Gianluca and
               Kozinski, Mateusz and Urschler, Martin},
  booktitle = {Uncertainty for Safe Utilization of Machine Learning in
               Medical Imaging (UNSURE), MICCAI 2026 Workshop},
  year      = {2026}
}
```

## Acknowledgements

This research was funded in whole or in part by the Austrian Science Fund
(FWF) [10.55776/PAT1748423](https://doi.org/10.55776/PAT1748423). Some
computational results have been achieved using the Austrian Scientific
Computing (ASC) infrastructure.
