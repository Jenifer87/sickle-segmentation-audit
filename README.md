sickle-segmentation-audit

Code and result tables for:

Segmentation errors that track the disease: a globally scaled seed threshold biases the sickle fraction in automated blood film analysis A. Philomina Jenifer, Priti Rishi. Submitted to Medical & Biological Engineering & Computing.

We audit the erythrocyte segmentation released with a published weakly supervised sickle cell pipeline against instance-level ground truth. The released method recovers 55.8% of cells, loses more cells from sickle cell films than from batch-matched controls (43.5% vs 62.3%), and under-reports the sickle fraction by about 17% in relative terms. The disparity is traced to a seed threshold scaled to the field-wide maximum of the distance transform; extended-maxima seeding (fix_C, h = 9) reduces it from 18.8 to 4.6 points.

Data

The thin blood film images are the public collection of Manescu et al. (2020): https://doi.org/10.5522/04/12407567

The ground-truth instance masks (too large for GitHub) are archived on Zenodo: [Zenodo DOI]

Repository contents
File	Purpose
paper-2.ipynb	Full analysis notebook (Kaggle): ground-truth generation with SAM, the audited and modified segmenters, IoU scoring, shape and contact analysis, sickle-fraction error, fix_C depth selection
paper2_figures.py	Regenerates Figures 1–5 and the graphical abstract from the result tables
audit_cells_full.csv	One row per ground-truth cell × method: IoU, recovery at IoU 0.3/0.5/0.7/0.75, contact fraction, fate
gt_shape.csv	Ground-truth cell shape descriptors (area, eccentricity, extent)
audit_per_field.csv, audit_trace_full.csv	Per-field recall and pipeline stage counts
bm_ridge_ratio.csv	Batch-matched cohort, per slide: phenotype, recall, ridge-to-field-maximum ratio
bm_alpha_summary.csv, bm_alpha_sweep.csv	Seed-coefficient (α) sweep
fixC_h9_batchmatched.csv, fixC_h9_percell.csv	fix_C (h = 9) results
sickle_fraction_error.csv, sickle_fraction_perslide.csv	Error in the reported sickle fraction
slide_labels.csv, gt_summary.csv, gt_config.json	Cohort labels and ground-truth settings
Methods compared
Released: faithful reimplementation of the distributed segmentation (Otsu, distance transform, seeds above 0.3 × field maximum, watershed, area filter [5,000, 17,000) px)
fix_A: keep watershed instances rather than re-deriving components
fix_B: seed threshold per connected component (local maximum)
fix_C: extended-maxima (h-maxima) seeding, h = 9, selected on 12 held-in slides and evaluated on 13 held-out slides
Cellpose-SAM: learned reference, diameter 115 px
Reproducing
Run paper-2.ipynb on Kaggle (Python 3.12, OpenCV, scikit-image, SciPy, pandas, statsmodels; SAM ViT-B and Cellpose-SAM need a GPU only for ground-truth generation and the learned reference).
Regenerate the figures from the CSVs:
bash
P2_DATA=. P2_OUT=figures python paper2_figures.py
Citation

If you use this code, please cite the paper (details will be added on publication).

Licence

Code: MIT. Result tables: CC BY 4.0.
