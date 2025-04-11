# ADHD-MultiModal
# Predicting Cognitive Functioning in ADHD Using Multimodal Neuroimaging

This repository contains the code for the manuscript “Predicting Cognitive Functioning in ADHD Using Multimodal Neuroimaging Across Populations and Clinics”. The study develops multimodal neuroimaging models to predict cognitive functioning in children and tests their generalizability across ADHD and non-ADHD groups using the Adolescent Brain Cognitive Development (ABCD) study and an external clinical cohort (Lytle et al.). 

## Overview

We trained stacked machine learning models integrating structural and functional neuroimaging features (fMRI task contrasts and connectivity, structural MRI, DTI) to predict cognitive scores. The models were validated internally (ABCD) and externally (Lytle et al.). This code supports ADHD identification, feature preprocessing, model training, and result analysis.

## Repository Structure

The repository is organized into four folders, each corresponding to a stage of the analysis pipeline:

- **`1_ADHD-selection`**: Scripts for identifying ADHD participants in the ABCD dataset using Cordova et al.’s tier system (Tiers 1–4) and Lytle et al.’s criteria (ADHD Rating Scale-IV, K-SADS-PL).
- **`2_Features-preprocessing`**: Scripts for preprocessing 81 neuroimaging feature sets from ABCD (fMRI task contrasts, FC, sMRI, DTI) and Lytle et al.’s features (Nback contrasts, FC, sMRI).
- **`3_modeling`**: Scripts for training and validating multimodal stacked models using nested leave-one-site-out cross-validation (ABCD) and applying them to internal (ABCD ADHD) and external (Lytle et al.) datasets.
- **`4_analysis&plotting`**: Scripts for analyzing model performance (e.g., Pearson’s r, group differences) and generating figures (e.g., scatterplots, feature importance).

## Prerequisites

To run the scripts, you’ll need:

- **Python**: Version [e.g., 3.8+].
- **Dependencies**: 
  - [List key libraries, e.g., `numpy`, `pandas`, `scikit-learn`, `nilearn`, `matplotlib`]
- **Data**:
  - ABCD dataset: Access via NIMH Data Archive (https://abcdstudy.org/scientists/data-sharing/).
  - Lytle et al.’s dataset: Publicly available at https://openneuro.org/datasets/ds002424/versions/1.2.0.
  - Note: Data access requires approval from respective repositories.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/[your-repo-name].git
