# GAP-LSTM: Graph-based Autocorrelation Preserving Networks for Geo-Distributed Forecasting
### GAP-LSTM is a novel geo-distributed forecasting method that leverages the synergic interaction of graph convolution, attention-based LSTM, 2D-convolution, and latent memory states to effectively exploit spatio-temporal autocorrelation in multivariate data generated at different nodes, resulting in improved modeling capabilities over state-of-the-art methods.

This repository contains all the materials necessary to replicate the experiments in the paper.

The **preds** folder contains a .csv file for every run (model, dataset).
Each file contains a column for the ground truth and a column for the predicted value, for every prediction timestep and every node.
The row order is nodes first (changing fastest), then timestep, and test sequences last (changing slowest).
You can use this file to calculate the used metrics and ensure that they match with the results in the paper.

We provide a simple script, **results-collector.py**, that you can use for this goal.
Before launching it, make sure that it is in the same folder as the preds folder.
You will also need to have numpy and pandas installed in your python environment.
Then, just run `python results-collector.py` from the same folder.
The results will be output in **results.csv**.
For convenience, this file is already provided in this repository.

The **data** folder contains the test sequences' indexes (dataset-name_0.1.npy), the indexes of sequences used for hyperparameter optimization (dataset-name_0.01.npy), and the closeness graph matrix (closeness-dataset-name.npy).
  
The source code of GAP-LSTM as well as the competitors is currently undergoing a polishing and documenting procedure.
It will be published as soon as it is completed, along with a complete and detailed description of how to run the models.
We will also provide a additional driver program to run the experiments.
