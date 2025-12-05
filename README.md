# DS4420-Final-Project

This repository contains our DS4420 project predicting next-week NFL injuries using publicly available performance and workload data. We implement two models:
	•	Bayesian hierarchical logistic regression (R) — manually coded Metropolis–Hastings sampler.
	•	Multilayer Perceptron (Python) — captures nonlinear patterns in weekly player workloads.

Key Files
	•	data_clean.py — loads, merges, and engineers features from nfl_data_py.
	•	train_mlp.py — trains the MLP and outputs evaluation metrics and prediction files.
	•	DS4420 Final Proj.Rmd / .html — trains the Bayesian hierarchical logistic regression model in R.
	•	DS4420_Project_Playground.ipynb — exploratory analysis.
