# Slicing Wasserstein over Wasserstein

Code accompanying:

**Moritz Piening, Robert Beinert**  
*Slicing Wasserstein over Wasserstein via Functional Optimal Transport*  
(ICLR 2026)

This repository provides implementations and experiments for comparing **meta-measures** (measures over measures) using sliced constructions, including a functional slicing approach based on 1D Wasserstein–quantile isometries.

---

## Repository Structure

### Notebooks

- `1_L2_Slicing_Example.ipynb`  
  Introductory example of functional / L2 slicing.

- `2_PointCloudSet_Comparison.ipynb`  
  Comparison of sets of point clouds (meta-measures).

- `3_PerceptualPatchComparison.ipynb`  
  Patch-based perceptual comparison experiments.

- `4_MNIST_sOTTD_Adapted.ipynb`  
  MNIST experiment (adapted sOTDD baseline).

- `5_FashionMNIST_sOTTD_Adapted.ipynb`  
  FashionMNIST experiment.

- `6_CIFAR_sOTDD_Adapted.ipynb`  
  CIFAR experiment.

- `Plotting_Correlations.ipynb`  
  Correlation analysis and visualizations.

---

### Core Python Modules

- `functional_swd.py`  
  Functional sliced Wasserstein distances:
  - random GP-based projection directions  
  - trapezoidal-rule projections  
  - 1D Wasserstein computation via POT  

- `sliced_hierarchical_OT.py`  
  Sliced and hierarchical Wasserstein-over-Wasserstein utilities.

- `point_cloud_utils.py`  
  Point cloud experiment helpers.

- `perceptual_patch_utils.py`  
  Patch extraction and perceptual utilities.

- `plotting_utils.py`  
  Visualization utilities.

- `knn_acc.py`  
  kNN-based evaluation utilities.

- `SFTLB_CodeCopy/`  
  Supporting code copied into the repository from  
  [slicing_fused_gromov_wasserstein](https://github.com/MoePien/slicing_fused_gromov_wasserstein)

---


## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{piening2026slicing,
  title={Slicing Wasserstein over Wasserstein via Functional Optimal Transport},
  author={Piening, Moritz and Beinert, Robert},
  booktitle={ICLR},
  year={2026}
}
