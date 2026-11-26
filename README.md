# abc-nn-summary-statistic

## Abstract
Approximate Bayesian Computing (ABC) works by comparing simulated and observed data using some summary statistic. The goal of this project is to access the impact of neural networks to learn summary statistics for time-series data. To this end the project will compare the process of ABC using "canonical", model-derived and neural network based summary statistics. Additionally, comparisons will be made in many different situations involving varying dimensions of SDE and changing the parameters.

## Repository

This repository contains 2 main folders 'DataConditionalABC-main' which contains the package from https://github.com/perojov/DataConditionalABC.

The other folder is 'abc-nn', this is a mini-project similar to the package but contains experiments for canonical and model-based summary statistics.

- To test canonical run exp_canonical.py
- To test model-based run exp_model_based.py
- To test NN run forward_inference.py

The results of ABC-SMC experiments for OU process are stored in OU-results as csv files. The file results.py in OU-results produces the plots.

## Train model

The PEN model is pre-trained and stored in ou/models