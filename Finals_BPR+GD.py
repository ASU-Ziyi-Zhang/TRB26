#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finals_BPR_GD.py

Path-flow calibration using movement counts and TMC travel times.
- Core idea: estimate time-varying path flows X (nonnegative) so that:
  1) Movement prediction matches observed movements: c_pred = A @ X
  2) Aggregated TMC travel time (via BPR-like mapping) matches observed TMC travel time.

This script:
1) Loads movement counts (per minute), an A-matrix (movement-to-route incidence),
   TMC mapping & BPR parameters, and observed TMC travel times.
2) Builds tensors for observed movements (c_obs) and observed travel time (y_obs).
3) Defines a Torch model with learnable path flows X and learnable global BPR
   parameters alpha, beta (bounded in reasonable ranges).
4) Optimizes X, alpha, beta by minimizing a joint loss over movements and travel time.
5) Saves the estimated path-flow time series to Excel.

NOTE
- File paths are currently absolute Windows paths as provided.
- No changes to your modeling logic or units.
- alpha, beta are now learnable within specified bounds; cap and t0 are fixed.

Requirements
- Python 3.9+
- pandas, numpy, torch, tqdm, geopy

Author: Ziyi Zhang
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from geopy.distance import geodesic
import torch.nn.functional as F
from collections import defaultdict

### === 1. Load and preprocess data === ###
# Load files
a_df = pd.read_excel('Movement_Data.xlsx')  # movement observations (c_obs)
b_df = pd.read_excel('Route_Movement_Matrix.xlsx')             # A-matrix
c_df = pd.read_excel('TMC_mapping.xlsx')                 # TMC mapping and BPR params
d_df = pd.read_csv('TMC_Data.csv')    # observed TMC travel time
e_df = pd.read_csv('TMC_Identification.csv')

# Normalize timestamp to minute string
a_df['Time'] = pd.to_datetime(a_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
d_df['measurement_tstamp'] = pd.to_datetime(d_df['measurement_tstamp']).dt.strftime('%Y-%m-%d %H:%M')

# Route list from A-matrix columns (skip first 3 metadata columns)
route_cols = b_df.columns[3:]
routes = list(route_cols)

# A-matrix is organized by (Node, Direction) rows
a_index = a_df[['Node', 'Direction']].drop_duplicates()
b_index = b_df[['Node', 'Direction']]
A_matrix = b_df[route_cols].values  # shape: (L movements, R routes)

# Map (Node, Direction) -> row index in A
index_map = {(node, direction): i for i, (node, direction) in enumerate(zip(b_index['Node'], b_index['Direction']))}

def get_c_obs_tensor():
    """Build c_obs tensor of shape (L, T) aligned to sorted unique time stamps."""
    time_list = sorted(a_df['Time'].unique())
    l_list = list(index_map.keys())
    c_obs = torch.zeros((len(l_list), len(time_list)))
    for i, (node, direction) in enumerate(l_list):
        df = a_df[(a_df['Node'] == node) & (a_df['Direction'] == direction)]
        for j, t in enumerate(time_list):
            val = df[df['Time'] == t]['Value']
            if not val.empty:
                c_obs[i, j] = float(val.values[0])
    return c_obs, time_list, l_list

c_obs, time_list, l_list = get_c_obs_tensor()

### === 2. Compute BPR params === ###

def compute_distance(lon1, lon2, lat=41.867):
    """Approximate longitudinal distance (in meters) at fixed latitude."""
    p1 = (lat, lon1)
    p2 = (lat, lon2)
    return geodesic(p1, p2).meters

# Free-flow time t_free = miles / reference_speed * 3600 (sec)
tmc_to_miles = dict(zip(e_df['tmc'], e_df['miles']))
c_df['miles'] = c_df['tmc_id'].map(tmc_to_miles)
c_df['t_free'] = (c_df['miles'] / c_df['reference_speed']) * 3600

# Capacity per segment: cap * lane (units already per-minute per your data)
c_df['cap'] = c_df['cap'] * c_df['lane']

def compute_overlap(row):
    """
    Compute longitudinal overlap ratio between segment and TMC if directions match.
    Returns a value in [0,1] (rounded to 1e-6), else 0 if no overlap.
    """
    if row['segment_direction'].lower() != row['tmc_direction'].lower():
        return 0.0

    tmc_lon_start, tmc_lon_end = sorted([row['tmc_lon_start'], row['tmc_lon_end']])
    seg_lon_start, seg_lon_end = sorted([row['segment_lon_start'], row['segment_lon_end']])

    # Overlap in longitudes
    overlap_start = max(tmc_lon_start, seg_lon_start)
    overlap_end = min(tmc_lon_end, seg_lon_end)
    if overlap_end <= overlap_start:
        return 0.0

    # Convert to distances (meters) and take ratio
    overlap_distance = compute_distance(overlap_start, overlap_end)
    tmc_total_distance = compute_distance(tmc_lon_start, tmc_lon_end)

    if tmc_total_distance == 0:
        return 0.0
    return round(overlap_distance / tmc_total_distance, 6)

c_df = c_df.copy()
c_df['radio'] = c_df.apply(compute_overlap, axis=1)

# Observed network travel time per minute: sum of travel_time_seconds over TMCs
y_obs_df = d_df.groupby('measurement_tstamp')['travel_time_seconds'].sum().reset_index()
y_obs_df.columns = ['Time', 'y_obs']

# Align y_obs with c_obs time_list to avoid time mismatch
y_obs_df = y_obs_df[y_obs_df['Time'].isin(time_list)]
y_obs_series = y_obs_df.set_index('Time').reindex(time_list).fillna(0)['y_obs']
y_obs = torch.tensor(y_obs_series.values, dtype=torch.float32)
print(y_obs)

### === 3. Define model === ###

class TrafficModel(nn.Module):
    def __init__(self, n_routes, n_times, A_matrix, c_df, l_list, index_map):
        super().__init__()
        # 1) Optimization variables: path flows X (R x T), nonnegative via ReLU
        self.x = nn.Parameter(torch.rand(n_routes, n_times))

        # 2) Fixed A-matrix as buffer to keep device/dtype consistent
        self.register_buffer('A', torch.as_tensor(A_matrix, dtype=torch.float32))

        # 3) Keep copies for forward computations
        self.c_df = c_df.reset_index(drop=True).copy()
        self.l_list = l_list

        # 4) Pre-build Node -> indices of movements (aggregate L/T/R to the same inbound approach)
        node_to_idxs = defaultdict(list)
        for (n, d), idx in index_map.items():
            node_to_idxs[n].append(idx)
        self.node_to_idxs = {k: torch.as_tensor(v, dtype=torch.long) for k, v in node_to_idxs.items()}

        # 5) Learnable global BPR parameters alpha, beta (bounded)
        #    You can adjust the ranges as needed.
        self.alpha_lo, self.alpha_hi = 0.01, 1.0
        self.beta_lo,  self.beta_hi  = 1.0,  8.0
        alpha_anchor = 0.15  # prior center
        beta_anchor  = 4.0

        # Initialize raw parameters so that sigmoid(raw) maps to the anchors above
        alpha_init = torch.logit(torch.tensor((alpha_anchor - self.alpha_lo)/(self.alpha_hi - self.alpha_lo)))
        beta_init  = torch.logit(torch.tensor((beta_anchor  - self.beta_lo )/(self.beta_hi  - self.beta_lo )))

        self.alpha_raw = nn.Parameter(alpha_init)  # scalar (unconstrained)
        self.beta_raw  = nn.Parameter(beta_init)   # scalar (unconstrained)

    def forward(self):
        # Convert raw alpha/beta into bounded actual values
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(self.alpha_raw)
        beta  = self.beta_lo  + (self.beta_hi  - self.beta_lo ) * torch.sigmoid(self.beta_raw)

        # Nonnegative path flows and movement predictions
        x = torch.relu(self.x).clamp(min=0)   # (routes, T)
        c_pred = self.A @ x                   # (L movements, T)

        T = c_pred.shape[1]
        zero_flow = c_pred.new_zeros(T)

        # Aggregate movements (L/T/R) to inbound-approach-level flow per Node
        node_flow_map = {}
        for node, idxs in self.node_to_idxs.items():
            if len(idxs) > 0:
                node_flow_map[node] = c_pred.index_select(0, idxs.to(c_pred.device)).sum(dim=0)
            else:
                node_flow_map[node] = zero_flow

        # For each mapping row, compute BPR-based travel time contribution with overlap "radio"
        y_parts = []

        for _, row in self.c_df.iterrows():
            node = row['Node']
            flow = node_flow_map.get(node, zero_flow)

            t0 = row['t_free']
            cap = row['cap']
            radio = row.get('radio', 1.0)

            # Basic sanitization (keep your units and pipeline unchanged)
            t0 = 0.0 if pd.isna(t0) else float(t0)
            cap = 0.0 if pd.isna(cap) else float(cap)
            radio = 0.0 if pd.isna(radio) else float(radio)

            # Convert to tensors (same device/dtype as flow)
            t0_t    = torch.as_tensor(t0,    dtype=c_pred.dtype, device=c_pred.device)
            cap_t   = torch.as_tensor(cap,   dtype=c_pred.dtype, device=c_pred.device)
            radio_t = torch.as_tensor(radio, dtype=c_pred.dtype, device=c_pred.device)

            # Your original BPR-style formula with radio on both base and congestion terms
            delay_cong = t0_t * alpha * radio_t * ((flow / cap_t).clamp(min=1e-6) ** beta)
            y_parts.append(t0_t * radio_t + delay_cong)

        # Aggregate all TMC-row contributions into the network travel time per minute
        y_l_t_hat = torch.stack(y_parts, dim=0).sum(dim=0)  # (T,)

        return x, c_pred, y_l_t_hat

def loss_fn(model, c_pred, c_obs, y_hat, y_obs,
            w_c=1.0, w_y=1.0, normalize=True,
            lam_alpha=1e-3, lam_beta=1e-3):
    """
    Joint loss:
    - Movement fit (c_pred vs c_obs)
    - Travel-time fit (y_hat vs y_obs)
    - Small priors anchoring alpha≈0.15 and beta≈4.0
    """
    # Data terms
    if normalize:
        c_scale = c_obs.std().clamp_min(1e-6)
        y_scale = y_obs.std().clamp_min(1e-6)
        c_loss = ((c_pred - c_obs) / c_scale).pow(2).mean()
        y_loss = ((y_hat - y_obs) / y_scale).pow(2).mean()
    else:
        c_loss = (c_pred - c_obs).pow(2).mean()
        y_loss = (y_hat - y_obs).pow(2).mean()

    # Weak priors on alpha, beta (pull toward 0.15 / 4.0)
    alpha = model.alpha_lo + (model.alpha_hi - model.alpha_lo) * torch.sigmoid(model.alpha_raw)
    beta  = model.beta_lo  + (model.beta_hi  - model.beta_lo ) * torch.sigmoid(model.beta_raw)
    reg = lam_alpha * (alpha - 0.15).pow(2) + lam_beta * (beta - 4.0).pow(2)

    return w_c * c_loss + w_y * y_loss + reg

### === 5. Optimization === ###
model = TrafficModel(n_routes=len(routes), n_times=len(time_list),
                     A_matrix=A_matrix, c_df=c_df, l_list=l_list, index_map=index_map)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

c_obs = c_obs.to(device)
y_obs = y_obs.to(device)

optimizer = Adam(model.parameters(), lr=0.01)
epochs = 3000

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    x, c_pred, y_hat = model()
    loss = loss_fn(model, c_pred, c_obs, y_hat, y_obs,
                   w_c=1.0, w_y=1.0, normalize=True,
                   lam_alpha=1e-3, lam_beta=1e-3)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            alpha_val = (model.alpha_lo + (model.alpha_hi - model.alpha_lo) * torch.sigmoid(model.alpha_raw)).item()
            beta_val  = (model.beta_lo  + (model.beta_hi  - model.beta_lo ) * torch.sigmoid(model.beta_raw)).item()
        tqdm.write(f"Epoch {epoch}, Loss: {loss.item():.4f} | alpha={alpha_val:.4f}, beta={beta_val:.4f}")

### === 6. Save results === ###
x_result = model.x.detach().clamp(min=0).round().int().numpy().T
x_df = pd.DataFrame(x_result, index=[str(t) for t in time_list], columns=routes)
# Write Excel and set the index column label to 'Time' so the first column has a header
x_df.to_excel("output/export_data.xlsx", index_label='Time')

print("Optimization complete and results saved.")
