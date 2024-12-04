# AICO: Feature Signiﬁcance Tests for Supervised Learning

This repository provides the implementation of **AICO (Add-In Covariates)**, a model-agnostic feature significance test for supervised learning models. Developed by the **Advanced Financial Technologies Laboratory (AFTLab)** at **Stanford University**, AICO aims to enhance model transparency and provide insights into feature importance for machine learning algorithms.

## Overview

Supervised learning algorithms are increasingly being used to guide economic and social decisions. However, the opaqueness of these algorithms presents significant challenges, particularly in highly regulated domains such as financial services, healthcare, and the judicial sector, where transparency is crucial.

This project introduces **AICO**, a feature significance test that improve model transparency by testing the significance of input features (variables) in a supervised learning algorithm. AICO evaluates the incremental effect of each feature on model predictive or decision performance, relative to a baseline where all features are kept constant.

## Contributions

### Model-Agnostic Hypothesis Testing Framework
AICO provides a versatile framework for hypothesis testing in machine learning models, with the following contributions:
- **Model-Agnostic**: A model-agnostic hypothesis testing framework that can be applied to any supervised machine learning models including regression and classification without assumption of the model's underlying structure
- **Statistically-Grounded**: Delivers measures of feature importance including exact non-asymptotic p-value, exact non-asymptotic confidence interval, and feature importance score
- **Computationally Efficient**: No refitting required

## Repository Structure

- **Source Code**: The main source code is located in the `src` folder.
- **Demonstration**: Usage of the AICO test is demonstrated in `Demo.ipynb`, a Jupyter notebook that walks through applying AICO to different types of datasets.
- **Environment Setup**: The conda environment used to develop this project is provided in `environment.yml`. This file contains all the necessary dependencies and packages for running the code in the repository.

## Installation

To set up the environment, use the following command:

```sh
conda env create -f environment.yml
```

This command will create a conda environment with all the required packages.

## Running the Demonstration

The Demo.ipynb notebook includes an end-to-end demonstration of the AICO test with simulated regression data.
