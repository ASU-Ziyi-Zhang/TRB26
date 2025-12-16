#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pathflow_vs_movements.py

Evaluates predicted movements reconstructed from path flows against observed
intersection movements. It computes per-(Node, Direction) error metrics and
exports a summary Excel plus two heatmaps (observed vs predicted), and a
per-minute aggregated error series on the subset of movements with observations.

Inputs (relative to this script):
- Movement_Data.xlsx                        (observed movements: Time, Node, Direction, Value)
- output/export_data.xlsx                   (predicted path flows: Time + R route columns)
- Route_Movement_Matrix.xlsx                (movement-to-route matrix; first 3 cols are meta)

Outputs:
- output/evaluation/error_data.xlsx         (per-node sheet + overall summary)
- output/evaluation/observed_heatmap.png
- output/evaluation/predicted_heatmap_gd.png
- output/evaluation/per_minute_errors.xlsx
- output/evaluation/worst_minutes_top50.xlsx
- output/evaluation/per_minute_series.png
- output/evaluation/per_minute_abs_error.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytz
from datetime import datetime
from pathlib import Path

# -----------------------------
# Paths (relative & reproducible)
# -----------------------------

# Input files
a_path = "Movement_Data.xlsx"      # observed movements
b_path = "output/export_data.xlsx"               # predicted path flows (X)
c_path = "Route_Movement_Matrix.xlsx"      # A-matrix

# Output files
metrics_xlsx = "output/evaluation/error_data.xlsx"
obs_heatmap_png = "output/evaluation/observed_heatmap.png"
pred_heatmap_png = "output/evaluation/predicted_heatmap_gd.png"

# -----------------------------
# Step 1: Load data
# -----------------------------
# a: observed movements
a_df = pd.read_excel(a_path)  # columns: Time, Node, Direction, Value
a_df['Time'] = pd.to_datetime(a_df['Time'])

# b: path-flow predictions (T × R) with first column "Time"
b_df = pd.read_excel(b_path)
b_df['Time'] = pd.to_datetime(b_df['Time'])

# Timezone handling (convert to America/Chicago safely)
chicago_tz = pytz.timezone('America/Chicago')

def _ensure_tz_and_convert(series, src_tz='UTC', dst_tz=chicago_tz):
    """
    Ensure a pandas datetime Series is timezone-aware.
    If tz-naive, localize to `src_tz`, then convert to `dst_tz`.
    """
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        s = s.dt.tz_localize(src_tz)
    return s.dt.tz_convert(dst_tz)

# Convert to tz-aware then drop tz in string formatting for downstream alignment
a_df['Time'] = _ensure_tz_and_convert(a_df['Time'])
b_df['Time'] = _ensure_tz_and_convert(b_df['Time'])
a_df['Time'] = a_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
b_df['Time'] = b_df['Time'].dt.strftime('%Y-%m-%d %H:%M')

# Extract predicted X (T × R) and time list
X_pred = b_df.iloc[:, 1:].to_numpy()  # shape: (T, R), columns after "Time"

# c: movement-to-route matrix (L × R); first 3 columns are metadata
c_df = pd.read_excel(c_path)
route_matrix = c_df.iloc[:, 3:].to_numpy()  # shape: (L, R)
node_dir_list = list(zip(c_df['Node'], c_df['Direction']))

# -----------------------------
# Step 2: Reconstruct predicted movements
# -----------------------------
# movement_pred has shape (L, T)
movement_pred = route_matrix @ X_pred.T

# -----------------------------
# Step 3: Build predicted movement DataFrame
# -----------------------------
time_list = b_df['Time']   # first column is Time
L, T = movement_pred.shape

records = []
for l in range(L):
    node, direction = node_dir_list[l]
    for t in range(T):
        records.append({
            'Node': node,
            'Direction': direction,
            'Time': time_list[t],
            'Predicted': movement_pred[l, t]
        })
pred_df = pd.DataFrame(records)

# -----------------------------
# Step 4: Preprocess observed movements
# -----------------------------
a_grouped = a_df.groupby(['Node', 'Direction', 'Time'])['Value'].sum().reset_index()
a_grouped.rename(columns={'Value': 'Observed'}, inplace=True)

# -----------------------------
# Step 5: Merge predicted vs observed
# -----------------------------
merged = pd.merge(pred_df, a_grouped, on=['Node', 'Direction', 'Time'], how='inner')

# -----------------------------
# Step 6: Error metrics per (Node, Direction)
# -----------------------------
group_metrics = []
for (node, direction), group in merged.groupby(['Node', 'Direction']):
    n_total = len(group)
    group_valid = group[~group['Observed'].isna()]
    n_used = len(group_valid)
    if n_used == 0:
        group_metrics.append({
            'Node': node,
            'Direction': direction,
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE(%)': np.nan,
            'sMAPE(%)': np.nan,
            'N': n_total,
            'N_used': n_used,
            'N_mape_excluded': 0
        })
        continue

    y_true = group_valid['Observed'].values
    y_pred = group_valid['Predicted'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # legacy-safe RMSE

    # MAPE: exclude zero true-values to avoid division blow-up
    nonzero_mask = np.abs(y_true) > 1e-8
    n_mape_excluded = int((~nonzero_mask).sum())
    if nonzero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

    # sMAPE works even when y_true == 0
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

    group_metrics.append({
        'Node': node,
        'Direction': direction,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE(%)': mape,
        'sMAPE(%)': smape,
        'N': n_total,
        'N_used': n_used,
        'N_mape_excluded': n_mape_excluded
    })

metrics_df = pd.DataFrame(group_metrics)

# Overall averages (ignore NaNs)
avg_cols = ['MAE', 'RMSE', 'MAPE(%)', 'sMAPE(%)']
overall = metrics_df[avg_cols].mean(skipna=True).to_frame().T
overall.insert(0, 'Summary', ['Overall Average'])

# Write metrics to Excel (per-node + overall)
with pd.ExcelWriter(metrics_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='per_node', index=False)
    overall.to_excel(writer, sheet_name='overall', index=False)

# -----------------------------
# Step 7: Heatmaps (observed vs predicted)
# -----------------------------
merged['Node_Dir'] = merged['Node'] + "_" + merged['Direction']
pivot_pred = merged.pivot_table(index='Node_Dir', columns='Time', values='Predicted', aggfunc='sum')
pivot_obs = merged.pivot_table(index='Node_Dir', columns='Time', values='Observed', aggfunc='sum')

def plot_heatmap(data, save_path):
    """Draw and save a heatmap for the movement matrix."""
    plt.figure(figsize=(18, 10))
    ax = sns.heatmap(
        data,
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Vehicles'},
        vmin=0, vmax=35,            # fixed color range for comparability
        xticklabels=50,             # show every 50th tick label on x-axis
        yticklabels=True
    )
    # Adjust colorbar label size if present
    if len(ax.figure.axes) > 1:
        cbar_ax = ax.figure.axes[-1]
        try:
            cbar_ax.yaxis.label.set_size(14)
        except Exception:
            try:
                cbar_ax.xaxis.label.set_size(14)
            except Exception:
                pass
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Node_Direction", fontsize=14)
    ax.set_xticks(np.arange(0, data.shape[1], 50))
    ax.set_xticklabels(data.columns[::50], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    # Optional: emphasize plot border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.close()

# Generate and save both heatmaps
plot_heatmap(pivot_obs,  obs_heatmap_png)
plot_heatmap(pivot_pred, pred_heatmap_png)



# -----------------------------
# Step 8: Per-minute aggregated errors (on observed subset)
# -----------------------------
# How to define "has data" for a movement at a minute:
OBS_POS_THRESHOLD = 0.0    # >0 means "has observation"; change to 1.0 if you want ">=1"
MAPE_MIN_DENOM    = 5.0    # only compute MAPE when the per-minute observed sum >= 5

# Back to datetime for grouping & plotting (we earlier formatted to string for alignment)
g = merged.copy()
g['Time'] = pd.to_datetime(g['Time'])

# Keep only rows (movements) that have observation > threshold at that minute
g_obspos = g[g['Observed'] > OBS_POS_THRESHOLD].copy()

# Aggregate per minute: sum Observed and Predicted over the same subset of movements
per_min = (g_obspos
           .groupby('Time')[['Observed', 'Predicted']]
           .sum()
           .rename(columns={'Observed': 'Obs_sum', 'Pred_sum': 'Pred_sum'})
           .reset_index())
# The rename above had a typo; fix Pred_sum name explicitly
per_min.rename(columns={'Predicted': 'Pred_sum'}, inplace=True)

# Minute-wise errors
per_min['AE'] = (per_min['Obs_sum'] - per_min['Pred_sum']).abs()
per_min['sMAPE%'] = 100.0 * 2.0 * per_min['AE'] / (
    per_min['Obs_sum'].abs() + per_min['Pred_sum'].abs() + 1e-8
)
per_min['APE%'] = np.where(
    per_min['Obs_sum'] >= MAPE_MIN_DENOM,
    100.0 * per_min['AE'] / per_min['Obs_sum'].clip(lower=1e-8),
    np.nan
)

# Summary metrics (over time, on this per-minute series)
MAE  = per_min['AE'].mean()
RMSE = np.sqrt((per_min['AE'] ** 2).mean())
MAPE = per_min['APE%'].mean(skipna=True)
WAPE = 100.0 * per_min['AE'].sum() / (per_min['Obs_sum'].abs().sum() + 1e-8)

print(f"[Per-minute aggregated on observed subset] MAE={MAE:.2f}, RMSE={RMSE:.2f}, "
      f"MAPE(denom>={MAPE_MIN_DENOM:.0f})={MAPE if not np.isnan(MAPE) else 'NaN'}%, "
      f"WAPE={WAPE:.2f}%")

# Save per-minute table and top-50 worst minutes by absolute error
per_min_out = "output/evaluation/per_minute_errors.xlsx"
per_min.to_excel(per_min_out, index=False)

worst50_out = "output/evaluation/worst_minutes_top50.xlsx"
(per_min
 .nlargest(50, 'AE')[['Time', 'Obs_sum', 'Pred_sum', 'AE', 'APE%', 'sMAPE%']]
 .to_excel(worst50_out, index=False))

# Plot per-minute series (Observed vs Predicted)
plt.figure(figsize=(14, 6))
plt.plot(per_min['Time'], per_min['Obs_sum'],  label='Observed (observed-movements subset)', lw=2)
plt.plot(per_min['Time'], per_min['Pred_sum'], label='Predicted (same subset)', lw=2, ls='--')
plt.title('Per-minute Total on Observed Movement Subset')
plt.xlabel('Time'); plt.ylabel('Vehicles')
# Use hour:minute formatter like the heatmaps and increase tick density (twice as dense)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig("output/evaluation/per_minute_series.png", dpi=300)
plt.show()

# Plot absolute error per minute
plt.figure(figsize=(14, 3.6))
plt.plot(per_min['Time'], per_min['AE'], color='tab:red', lw=1.5)
plt.title('Absolute Error per Minute (Observed-subset aggregation)')
plt.xlabel('Time'); plt.ylabel('AE (veh/min)')
ax2 = plt.gca()
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig("output/evaluation/per_minute_abs_error.png", dpi=300)
plt.show()
