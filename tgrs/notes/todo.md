# TGRS notes and open questions

## 1. Decide the remote-sensing framing (blocks the intro and experiments)

TGRS is IEEE Transactions on Geoscience and Remote Sensing. The code in the repo root is a generic monocular depth pipeline (images + `_depth` maps in a folder). The paper needs a geoscience/remote-sensing angle and matching data, for example:

- height / elevation estimation from single aerial or satellite images
- depth estimation for underwater or UAV imagery
- something else you have in mind

Whichever you pick determines the datasets, the baselines, and the related-work section. Nothing in the draft assumes one yet.

## 2. Experiments needed before the paper can be written

- [ ] Choose dataset(s) and put the loader/config under `experiments/`
- [ ] Pick baselines and run them under the same split
- [ ] Add standard depth metrics. The current evaluation reports RMSE and MAE only (`chebfeatdepth.py`, `validate` and `test_model`); add AbsRel, RMSE-log, and delta < 1.25 / 1.25^2 / 1.25^3.
- [ ] Ablations: each loss term, Chebyshev order K, superpixel vs. grid nodes, k-NN k, depth of the ChebNet
- [ ] Multiple seeds, so results can be reported with a spread
- [ ] Runtime / memory (graph construction runs per image on CPU in a Python loop; reviewers will ask)

## 3. Things to check in the code before trusting any numbers

These are observations from reading `chebfeatdepth.py`, not confirmed bugs. Nothing was run.

- Output size: `GraphToImageProjector` already produces features at full `(H, W)`, and `DepthEstimationHead` then applies two x2 bilinear upsamples (the `i >= len(channels) - 3` branch). Confirm the prediction matches the target size, otherwise the loss will fail or compare mismatched shapes.
- `cheb_graph_loss` is described as Dirichlet energy but is the mean unsquared L2 difference across edges (a total-variation-style term). Either rename it in the paper or square it in the code.
- The attention-pooled `global_features` is computed but not used by the depth head.
- Edges are a k-NN graph on superpixel centroids, not the superpixel adjacency graph; the repo README calls it a region-adjacency graph. Pick one description and make code and text agree.
- Barycentric projection loops over every pixel in Python; fine for 256x256, slow for remote-sensing image sizes.

## 4. Writing

- [ ] Fill in the author block (affiliation, co-authors) in `paper/main.tex`
- [ ] Confirm the current TGRS submission requirements (length, template, data availability, code statement)
- [ ] Verify every entry in `paper/references.bib` and add the baseline and remote-sensing references
- [ ] Create the overview figure (`figures/overview`)
