# Percept Predictability

Code and analysis notebooks supporting the manuscript titled:

>Longitudinal neural dynamics forecast treatment response and therapeutic engagement during deep brain stimulation for OCD

This repository contains the analysis code used to reproduce the main-text figures, supplemental figures, and supplemental tables for this study. The project investigates whether chronic neural recordings from sensing-enabled deep brain stimulation (DBS) can provide objective biomarkers of therapeutic engagement and treatment response in treatment-resistant obsessive-compulsive disorder (OCD).

<p align="left">
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-in_preparation-orange" alt="Status: in preparation">
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

We analyzed chronic ventral striatal local field potential (LFP) recordings from 24 patients undergoing bilateral VC/VS DBS for treatment-resistant OCD. The recordings consisted of 9-Hz LFP power measurements sampled every 10 minutes through the Medtronic BrainSense Timeline system.

The primary biomarker is neural predictability, quantified using the coefficient of determination ($R^2$) between observed neural activity and predictions from a causal, lag-1 linear autoregressive model (LinAR-1). The model is trained on preceding neural activity and evaluated prospectively on subsequent data, allowing neural predictability to be calculated continuously throughout DBS therapy.

The analyses in this repository examine:

* Changes in neural predictability following DBS
* Differences in neural predictability between clinical responders and non-responders
* Relationships between neural predictability and OCD symptom severity
* Classification of clinical state using neural predictability
* Early post-DBS neural changes as predictors of eventual treatment response
* Circadian organization of ventral striatal neural activity
* Baseline neural features that distinguish eventual responders from non-responders
* Supplemental analyses evaluating model structure, hemispheric differences, partial response, relapse, and longitudinal neural dynamics

The study found that responders exhibited reductions in neural predictability following DBS, with early changes occurring substantially before clinical response. Baseline circadian features of ventral striatal activity also contained information about eventual treatment response.

---

## Repository Structure

```text
.
├── README.md
│
├── process_data.ipynb
│   └── Main analysis pipeline and reproduction of manuscript figures
│
├── supplemental.ipynb
│   └── Extended Data figures
│
├── tables.ipynb
│   └── Extended Data tables
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
│   └── Generated supplemental tables
│
└── ...
```

---

## Getting Started

### Prerequisites

- Python 3.13 (developed and tested on 3.13; other versions untested)
- ~2 GB free disk for cached intermediates
- At least 16 GB RAM
- Jupyter Notebook

### Installation

Clone the repository and create a Python environment (we like Conda) containing the dependencies used by the analysis notebooks.

```bash
git clone https://github.com/ProvenzaLab/PerceptPredictability.git
cd PerceptPredictability
conda create -n "percept_predictability_env" python=3.13
conda activate percept_predictability_env
pip install -r requirements.txt
```

### Obtain the Data

The analyses use deidentified neural and clinical data. Our datasets supporting the findings will be made publicly available (see [Data Availability](#data-availability)). Place the required files in the data folder of the repository.

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

### Reproduce the supplemental figures

Run `supplemental.ipynb`. This notebook contains analysis corresponding to the supplemental/extended data figures. It runs in less than 30 minutes on a laptop.

The supplemental analyses include:

* Extended Data Figure 2: Right hemisphere neural predictability
* Extended Data Figure 3: Partial responders and clinical relapse
* Extended Data Figure 4: Comparison of autoregressive model architectures
* Extended Data Figure 5: Longitudinal neural predictability following DBS activation
* Extended Data Figure 6: FFT and circadian/polar trajectories across patients
* Extended Data Figure 7: Neural predictability analysis application

### Reproduce the supplemental tables

Run `tables.ipynb`. This notebook generates the supplemental/extended data tables accompanying the manuscript. It runs in less than 120 minutes on a laptop.

These include:

* Extended Data Table 2: Patient demographics, clinical outcomes, and recording durations
* Extended Data Table 3/4: Per-patient pre-DBS versus stable state statistics with and without effective sample-size correction
* Extended Data Table 5: Pooled pre-DBS versus stable state statistics
* Extended Data Table 6: Logistic regression classification statistics
* Extended Data Table 7: Per-fold logistic regression performance
* Extended Data Table 8: Comparison of different models
* Extended Data Table 9: Performance comparison between linear autoregressive paradigms

---

## Data availability

Deidentified participant-level neural and clinical datasets supporting these findings will be made publicly available through the Data Archive for the BRAIN Initiative (DABI) registry.

> Status: the DABI accession link is pending and will be added here upon publication. In the interim, please contact the corresponding author regarding data access.

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

If you use this code, please cite the associated manuscript:

```bibtex
@Unpublished{Zhou2026,
  title   = {Longitudinal neural dynamics forecast treatment response and
             therapeutic engagement during deep brain stimulation for OCD},
  author  = {Zhou, Jeffrey and Hanish, Rick R. and Merk, Timon and
             Soubra, Sarah and Allam, Anthony K. and Reddy, Sandesh and
             Bao, Royal and Fraczek, Tomasz M. and Gates, Victoria R. and
             Libowitz, Mark and Paulo, Danika L. and Katlowitz, Kalman A. and
             Mickey, Brian J. and Kious, Brent M. and Herron, Jeffrey A. and
             Shofty, Ben and Patel, Ankit B. and Storch, Eric A. and
             Goodman, Wayne K. and Sheth, Sameer A. and Provenza, Nicole R.},
  year    = {2026},
  note    = {Manuscript in preparation}
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

