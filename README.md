# GAP-LSTM: Graph-based Autocorrelation Preserving Networks for Geo-Distributed Forecasting
### GAP-LSTM is a novel geo-distributed forecasting method that leverages the synergic interaction of graph convolution, attention-based LSTM, 2D-convolution, and latent memory states to effectively exploit spatio-temporal autocorrelation in multivariate data generated at different nodes, resulting in improved modeling capabilities over state-of-the-art methods.

This repository is a WIP and will contain all the materials necessary to replicate experiments in the paper.

The **preds** folder contains a .csv file for every run (model, dataset).
Each file contains a column for the ground truth and a column for the predicted value, for every prediction timestep and every node.
The row order is nodes first (changing the fastest), then timestep, and test sequences last (changing the slowest).
You can use this file to calculate the used metrics and ensure that they match with the results in the paper.

We provide a simple script, **results-collector.py**, that you can use for this goal.
Before launching it, make sure that it is in the same folder as the preds folder.
Then, just run `python results-collector.py` from the same folder.
The results will be output in **results.csv**.
