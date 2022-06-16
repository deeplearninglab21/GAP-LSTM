import os
import re
import pandas as pd
import numpy as np

PATH = 'preds'
OUTPUT_PATH = 'results.csv'
model_regex = '(ARIMA|((Bi|CNN|SVD|GCN|Attention|GAP)-)?LSTM(-(Default|Weighted))?|GRU)-(AUTO|BS\d\d?LR1e-\d(LRS)?(ES)?)'
collected_results = pd.DataFrame(columns=['model', 'dataset', 'avg_mae', 'std_mae', 'avg_rmse', 'std_rmse', 'avg_smape', 'std_smape'])
runs = os.listdir(PATH)

for run in runs:
    preds_df = pd.read_csv(os.path.join(PATH, run))
    model = re.search('(?<=preds-)(ARIMA|((Bi|CNN|SVD|GCN|Attention|GAP)-)?LSTM(-(Default|Weighted))?|GRU)(?=-)', run).group(0)
    dataset = re.search('(?<=-)(lightsource|wind-nrel|pv-italy|pems-sf-weather|beijing-airquality)', run).group(0)
    pred_column = re.search(model_regex, preds_df.columns[2]).group(0)
    preds_df['abs_diff'] = (preds_df['truth'] - preds_df[pred_column]).abs()
    preds_df['mse'] = preds_df['abs_diff']**2
    preds_df['rel_diff'] = ((preds_df['truth'] - preds_df[pred_column]) / preds_df['truth']).abs()
    preds_df['rel_sym_diff'] = ((preds_df['truth'] - preds_df[pred_column]).abs() / (preds_df['truth'].abs() + preds_df[pred_column].abs()))

    n = 0
    t = 0
    if dataset == 'lightsource':
        n = 7
        t = 19
    if dataset == 'wind-nrel':
        n = 5
        t = 24
    if dataset == 'pv-italy':
        n = 17
        t = 19
    if dataset == 'pems-sf-weather':
        n = 163
        t = 6
    if dataset == 'beijing-airquality':
        n = 11
        t = 6

    avg_mae = np.mean(preds_df['abs_diff'])
    std_mae = np.std(preds_df['abs_diff'])
    mse = np.mean([preds_df['mse'][i*t*n:(i+1)*t*n] for i in range(round(len(preds_df)/t/n))], axis=1)
    rmse = [np.sqrt(mse[i]) for i in range(len(mse))]
    avg_rmse = np.mean(rmse, axis=0)
    std_rmse = np.std(rmse)
    smape = np.mean([preds_df['rel_sym_diff'][i*t*n:(i+1)*t*n] for i in range(round(len(preds_df)/t/n))], axis=1)
    avg_smape = np.mean(smape, axis=0)
    std_smape = np.std(smape, axis=0)
    row = [model, dataset, avg_mae, std_mae, avg_rmse, std_rmse, avg_smape, std_smape]
    collected_results.loc[len(collected_results)] = row

collected_results.to_csv(OUTPUT_PATH)
