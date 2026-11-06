# Percept Predictability

Code and analysis notebooks supporting the manuscript titled:

>Chronic intracranial neural dynamics forecast treatment response and therapeutic engagement during deep brain stimulation for OCD

This repository contains the analysis code used to reproduce the main-text figures and the Extended Data and Supplementary figures and tables for this study. The project investigates whether chronic neural recordings from sensing-enabled deep brain stimulation (DBS) can provide objective biomarkers of therapeutic engagement and treatment response in treatment-resistant obsessive-compulsive disorder (OCD).

<p align="left">
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python 3.13">
  <img src="https://img.shields.io/badge/status-Under_Review-orange" alt="Status: under review">
</p>

---

## Table of contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducing the Results](#reproducing-the-results)
- [Data Availability](#data-availability)
- [Ethics](#ethics)
- [Companion application](#companion-application)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

Clinical improvement following psychiatric DBS can take weeks to months, making it difficult to determine early in treatment whether stimulation is engaging the intended neural circuitry. This study asks whether longitudinal neural dynamics contain information about eventual treatment response before clinical improvement becomes apparent.

We analyzed nearly 120,000 patient-hours of chronic local field potential (LFP) recordings from the DBS target region in 24 patients undergoing bilateral ventral capsule DBS for treatment-resistant OCD. Leads followed one of two trajectories through the ventral capsule, ending near its border with either the ventral striatum (VC/VS) or the bed nucleus of the stria terminalis (BNST). The recordings consisted of average LFP power in the 8.79 ± 2.5 Hz band (referred to as 9 Hz), sampled every 10 minutes through the Medtronic BrainSense Timeline system.

The primary biomarker is neural predictability, quantified using the coefficient of determination ($R^2$) between observed neural activity and predictions from a causal, lag-1 linear autoregressive model (LinAR-1). Within a three-day sliding window, the model is trained on the first two days and evaluated prospectively on the third, allowing neural predictability to be calculated continuously throughout DBS therapy.

The analyses in this repository examine:

* Changes in neural predictability following DBS
* Differences in neural predictability between clinical responders and non-responders
* Relationships between neural predictability and OCD symptom severity
* Classification of clinical state using neural predictability
* Early post-DBS neural changes as predictors of eventual treatment response
* Circadian organization of neural activity in the DBS target region
* Baseline neural features that distinguish eventual responders from non-responders
* Additional analyses evaluating model structure, hemispheric differences, lead trajectory, partial response, relapse, and longitudinal neural dynamics

The study found that responders exhibited reductions in neural predictability following DBS, with early changes occurring substantially before clinical response. Baseline circadian features of neural activity also contained information about eventual treatment response.

---

## Repository Structure

```text
.
├── README.md
│
├── process_data.ipynb
│   └── Main analysis pipeline, main-text figures, and some Extended Data and Supplementary tables
│
├── extended_supp_figures.ipynb
│   └── Extended Data and Supplementary figures
│
├── extended_supp_tables.ipynb
│   └── Extended Data and Supplementary tables
│
├── *_utils.py
│   └── Various helper functions to process data, create figures, and perform analyses
│
├── data/
│   └── Input and intermediate data files
│
├── figures/
│   └── Generated manuscript figures
│
├── tables/
│   └── Generated Extended Data and Supplementary tables
│
└── ...
```

---

## Getting Started

### Prerequisites

- Windows 11 (the only operating system tested)
- Python 3.13 (developed and tested on 3.13; other versions untested)
- ~2 GB free disk for cached intermediates
- At least 16 GB RAM
- Jupyter Notebook
- No non-standard hardware

### Installation

Clone the repository and create a Python environment (we like Conda) containing the dependencies used by the analysis notebooks. A clean install takes about 4 minutes on a normal desktop or laptop computer with a broadband connection.

```bash
git clone https://github.com/ProvenzaLab/PerceptPredictability.git
cd PerceptPredictability
conda create -n "percept_predictability_env" python=3.13
conda activate percept_predictability_env
pip install -r requirements.txt
```

### Obtain the Data

The analyses use deidentified neural and clinical data, which are available upon request (see [Data Availability](#data-availability)). Place the files in `data/DABI_upload/`, the folder `config.py` reads from.

### Try the Demo Dataset

To run the code without requesting the study data, use the synthetic dataset in `data/demo/`. It contains four simulated patients (`DEMO001` and `DEMO002` are responders; `DEMO003` and `DEMO004` are non-responders) with 50 days of data each. **The data is simulated, not recorded from patients, and results computed from it say nothing about the manuscript's findings.**

1. Set `USE_DEMO_DATA = True` in `config.py`, and restart the kernel of any notebook that is already open.
2. Run `process_data.ipynb`, then `extended_supp_figures.ipynb`, then `extended_supp_tables.ipynb`. The extended notebooks use files written by `process_data.ipynb`, so the order matters. All three together take about 2 minutes on a normal desktop or laptop computer (timed on Windows 11 with an 8-core Intel Core Ultra 9 288V and 32 GB RAM).

The notebooks should run without errors and write 35 files to `figures/demo/`, 17 to `tables/demo/`, and 5 to `data/demo_derived/`. These folders are separate from the study outputs, so the demo never overwrites results from the study data. To regenerate the dataset, run `python make_demo_data.py`, which uses a fixed seed and reads no patient data.

---

## Reproducing the Results.

The following steps allow you to reproduce each of the elements present in the manuscript and its supplement. Note that the SVG files produced and saved by the notebooks were post-processed by hand in Adobe Illustrator to generate the figures seen in the manuscript and supplement.

### Reproduce the main-text figures

Start with `process_data.ipynb`. This notebook contains the primary data modeling and processing workflow used to generate the manuscript's main results and figures. Running the notebook should take less than 10 minutes on a modern laptop.

The main analysis includes:

1. Calculation of acrophase using the cosinor model
1. Calculation of neural predictability using the LinAR-1 model
1. Comparison of neural predictability across clinical states and response groups
1. Clinical-state classification using logistic regression and leave-one-patient-out cross-validation
1. Analysis of early post-DBS neural changes
1. Analysis of baseline and treatment-related circadian dynamics
1. Generation of the main-text figures and associated statistics

The main text figures generated are the following:

1. Neural predictability/LinAR-1 framework
1. Neural predictability versus clinical state and Y-BOCS
1. Clinical state classification with sliding windows and different primary features
1. Early neural changes and treatment-response forecasting
1. Circadian neural dynamics

The core preprocessing pipeline (not included in the published code) extracts BrainSense Timeline data, stimulation settings, and metadata from the JSON files downloaded from the Percept device. It then addresses clock synchronization, removes overvoltage artifacts, interpolates small gaps, and z-scores LFP power within each day. The data available on DABI are already preprocessed.

### Reproduce the Extended Data and Supplementary figures

Run `extended_supp_figures.ipynb` after `process_data.ipynb`. It runs in less than 30 minutes on a laptop.

This notebook generates:

* Extended Data Figure 1: Right hemisphere neural predictability
* Extended Data Figure 2: Neural predictability during partial response and clinical relapse
* Extended Data Figure 3: Comparison of autoregressive model architectures
* Supplementary Figure 3: Classification accuracy by lead trajectory
* Supplementary Video 1: Change in neural predictability over time following DBS activation

It also generates per-patient FFT and polar plots that are not part of the current manuscript. Supplementary Figures 1 and 2 (contact locations) and Supplementary Figure 4 (application screenshot) are not generated by this code.

### Reproduce the Extended Data and Supplementary tables

Run `extended_supp_tables.ipynb` after `process_data.ipynb`. It runs in less than 120 minutes on a laptop. Tables are written to `tables/`.

This notebook generates:

* Extended Data Table 1: Patient demographics, clinical outcomes, and recording durations (`pt_demo.xlsx`)
* Extended Data Table 2 and Supplementary Table 2: Left hemisphere per-patient pre-DBS versus stable state statistics with and without effective sample-size correction (`pt_stats.xlsx`)
* Supplementary Table 3: Right hemisphere per-patient pre-DBS versus stable state statistics (`right_pt_stats.xlsx`)
* Extended Data Table 3 and Supplementary Table 4: Pooled pre-DBS versus stable state statistics for the left and right hemispheres (`pooled_stats.xlsx`, `right_pooled_stats.xlsx`)
* Extended Data Table 5: Logistic regression classification statistics for both hemispheres, compared with chance (`all_reg_stats.xlsx`, `all_reg_stats_circular.xlsx`)
* Extended Data Table 6: Classification performance of the different autoregressive models (`ar_reg_stats.xlsx`)
* Extended Data Table 7: Per-fold logistic regression performance and decision thresholds (`perfold_logreg_stats.xlsx`)
* Supplementary Table 5: Model comparison with DeLong's test (`ark_delong_test_results.xlsx`)

`process_data.ipynb` generates the remaining tables:

* Extended Data Table 4: Classification statistics across moving average windows (`windowed_stats.csv`)
* Supplementary Table 6: Average change in neural predictability over the first 14 days of DBS (`first_14d_avg_delta_R2.xlsx`)
* Supplementary Table 7: Stimulation parameters 14 days after DBS activation (`stim_settings_at_14d.xlsx`)
* Supplementary Table 8: Power at the 6-, 12-, and 24-hour components (`5c_power_stats.xlsx`)
* Supplementary Table 9: Average pre-DBS acrophase (`5g_acrophase_stats.xlsx`)
* Supplementary Table 10: Average pre-DBS power by time of day (`5f_quadrant_power_stats.xlsx`)

Supplementary Table 1 (MNI coordinates of the contacts) is not generated by this code.

---

## Data availability

Deidentified participant level neural and clinical datasets that support the findings of this study are available upon request via this [DABI registry](https://dabi.loni.usc.edu/projects/AZB3H0HMN433/).

---

## Ethics

All participants gave fully informed consent. Protocols were approved by the institutional review boards at Baylor College of Medicine (IRB H-48392; H-56119) and the University of Utah (IRB 00169174).

---

## Companion application

For interactive processing and visualization of Percept BrainSense Timeline data, see the lab's standalone GUI:

[ProvenzaLab/Percept_Data_Analysis_App](https://github.com/ProvenzaLab/Percept_Data_Analysis_App)

Built with PyQt, installable as an executable on Windows and macOS, handles multiple patients at once, and processes roughly one month of data in ~30 seconds on a laptop. All computed variables and plots export to CSV and PNG.

---

## Citation

If you use this code, please cite the associated manuscript. Jeffrey Zhou and Rick R. Hanish contributed equally.

```bibtex
@ARTICLE{Zhou2026-zt,
  title     = "Chronic intracranial neural dynamics forecast treatment
               response and therapeutic engagement during deep brain
               stimulation for {OCD}",
  author    = "Zhou, Jeffrey and Hanish, Rick R and Merk, Timon and Bao,
               Royal and Soubra, Sarah and Allam, Anthony K and Reddy,
               Sandesh and Gates, Victoria R and Allam, Vincent and Libowitz,
               Mark and Paulo, Danika L and Katlowitz, Kalman A and Mickey,
               Brian J and Kious, Brent M and Herron, Jeffrey A and
               Fr{\k{a}}czek, Tomasz M and Shofty, Ben and Patel, Ankit B and
               Heilbronner, Sarah R and Storch, Eric A and Goodman, Wayne K
               and Sheth, Sameer A and Provenza, Nicole R",
  journal   = "medRxiv",
  note      = "Preprint",
  month     =  sep,
  year      =  2026,
  doi       = {10.64898/2026.09.18.26363320},
  url       = {https://doi.org/10.64898/2026.09.18.26363320},
}
```

Please also cite the prior work that introduced the neural predictability biomarker:

```bibtex
@Article{Provenza2024,
  author  = {Provenza, Nicole R. and Reddy, Sandesh and Allam, Anthony K. and
              Rajesh, Sameer V. and Diab, Nabeel and Reyes, Gabriel and
              Caston, Rose M. and Katlowitz, Kalman A. and Gandhi, Ajay D. and
              Bechtold, Raphael A. and Dang, Huy Q. and Najera, Ricardo A. and
              Giridharan, Nisha and Kabotyanski, Katherine E. and Momin, Faiza and
              Hasen, Mohammed and Banks, Garrett P. and Mickey, Brian J. and
              Kious, Brent M. and Shofty, Ben and Hayden, Benjamin Y. and
              Herron, Jeffrey A. and Storch, Eric A. and Patel, Ankit B. and
              Goodman, Wayne K. and Sheth, Sameer A.},
  title   = {Disruption of neural periodicity predicts clinical response after
              deep brain stimulation for obsessive-compulsive disorder},
  journal = {Nature Medicine},
  year    = {2024},
  month   = {Oct},
  day     = {01},
  volume  = {30},
  number  = {10},
  pages   = {3004-3014},
  issn    = {1546-170X},
  doi     = {10.1038/s41591-024-03125-0},
  url     = {https://doi.org/10.1038/s41591-024-03125-0}
}
```

---

## Acknowledgements

This work was supported by the NIH NINDS BRAIN Initiative (UH3 NS136631), NIH NIMH (R01 MH139889), the Brain and Behavior Research Foundation Young Investigator Award, and the McNair Foundation.

We are grateful to the patients and their families for their participation in this research program.

---

## Contact

For questions regarding the analysis or repository, please open a GitHub issue or contact the corresponding author listed in the manuscript.

Provenza Lab<br>
Baylor College of Medicine / Rice University

