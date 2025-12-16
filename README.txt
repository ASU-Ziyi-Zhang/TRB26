Roosevelt_Calibration — BPR+GD Calibration & Evaluation

This project models and evaluates traffic flow on Roosevelt Rd, Chicago, IL 60607 using:
1) movement counts from Sage nodes,
2) per-minute travel times from HERE (TMC), and
3) a physics-guided learning model combining BPR with gradient descent (PyTorch).

It includes:
- Route–Movement reconstruction via an A-matrix
- BPR+GD hybrid calibration (path flows + learnable BPR parameters)
- Two evaluation modes: per (Node, Direction) and per-minute aggregated

Repository layout
--------------------------------
.
├─ data/
│  ├─ Movement_Data.xlsx                # Observed movements: Time, Node, Direction, Value
│  ├─ Route_Movement_Matrix.xlsx        # A-matrix: first 3 meta cols, then route columns
│  ├─ TMC_data.csv                      # HERE TMC travel times (per minute)
│  ├─ TMC_Identification.csv            # HERE TMC ID ↔ length (miles), etc.
│  └─ TMC_mapping.xlsx          # Final mapping & params (lane, cap, reference_speed, Node)
│
├─ output/
│  ├─ export_data.xlsx                  # Predicted path flows (X): Time + R route columns (model output)
│  └─ evaluation/
│     ├─ error_data.xlsx                # Per-node metrics + overall summary
│     ├─ observed_heatmap.png           # Heatmap of observed movements
│     ├─ predicted_heatmap_gd.png       # Heatmap of predicted movements
│     ├─ per_minute_errors.xlsx         # Per-minute totals & errors
│     ├─ worst_minutes_top50.xlsx       # Top-50 worst minutes by AE
│     ├─ per_minute_series.png          # Observed vs Predicted per-minute series
│     └─ per_minute_abs_error.png       # Absolute error per minute
│
├─ Finals_BPR_GD.py         # BPR+GD calibration (PyTorch)
└─ Evaluate.py    # Evaluation: per-node + per-minute (observed-subset)

If a folder under output/ is missing, the scripts will create it.

Environment
-----------
- Python ≥ 3.9
- Packages: pandas, numpy, torch, tqdm, geopy, openpyxl,
  matplotlib, seaborn, scikit-learn, pytz

Quick install:
pip install pandas numpy torch tqdm geopy openpyxl matplotlib seaborn scikit-learn pytz

Data assumptions & units
------------------------
- Time resolution: 1 minute (all inputs aligned at minute level).
- A-matrix: maps routes (R) to intersection movements (L); first 3 columns are meta (Node, Direction, …),
  then route columns.
- Movement meaning: (Node, Direction) denotes lane movements within one approach (e.g., Left/Straight/Right).
- BPR params & units:
  - t_free (seconds) from miles / reference_speed * 3600
  - cap is per-minute capacity for the whole approach (already lane-adjusted: cap * lane)
  - radio is the overlap ratio (0–1) between a SUMO segment and a TMC segment (by longitude span).
- TZ: scripts normalize timestamps to America/Chicago for joining/plots.

Method — BPR + Gradient Descent (physics-guided)
------------------------------------------------
Variables
- Path flows X_{r,t} ≥ 0 (decision variable, optimized per route per minute).
- Movements C_{l,t} = Σ_r A_{l,r} X_{r,t}.
- Per-minute network travel time (observed) y_obs,t from HERE (summed across matched TMCs).

BPR travel time model (learnable)
For each mapped row (Node, tmc_id):
  y_{l,t} = t_free,l * radio_l + t_free,l * α * radio_l * ((flow_node(l),t / cap_l)^β)

- Flow used in BPR is the approach-level sum of movements belonging to the same Node (L+T+R),
  consistent with capacity being the whole-approach capacity.
- α, β are global learnable scalars with bounds (e.g., α ∈ [0.01,1.0], β ∈ [1,8]) and a small prior (anchor)
  at (0.15, 4.0).

Loss (multi-task, normalized)
  L = MSE((C - C_obs)/σ_C) + MSE((y - y_obs)/σ_y) + λ_α(α-0.15)^2 + λ_β(β-4)^2

- Normalization by empirical stds improves balancing.
- Non-negativity on X via ReLU.
- Optimization: Adam, gradient clipping optional.
Outputs: The script writes output/export_data.xlsx with per-minute path flows X; these are then used by the evaluation.

How to run
----------
Run from the repository root. Make sure the data files are in the paths shown above.

1) Calibrate with BPR+GD (produces output/export_data.xlsx):
   python Finals_BPR_GD.py

2) Evaluate (per-node + per-minute):
   python Evaluate.py

What you get from evaluation
- output/evaluation/error_data.xlsx
  • per_node: MAE/RMSE/MAPE/sMAPE per (Node, Direction)
  • overall: column-wise averages over nodes
  • per_minute: observed/predicted totals and per-minute AE/APE/sMAPE
- Heatmaps: observed_heatmap.png, predicted_heatmap_gd.png
- Time-series: per_minute_series.png, per_minute_abs_error.png
- Worst minutes: worst_minutes_top50.xlsx

Per-minute aggregation rule
- For each minute, sum only movements that have an observation at that minute
  (threshold configurable inside the script: OBS_POS_THRESHOLD; default > 0).
- Error metrics on this per-minute series:
  AE (absolute error), sMAPE, MAPE (only when denom ≥ MAPE_MIN_DENOM, default 5), WAPE.

Practical tips
--------------
- Time alignment: Put all inputs on the same clock. The scripts convert to America/Chicago and align at minute strings.
- Routes consistency: Route columns in Route_Movement_Matrix.xlsx must match those in output/export_data.xlsx.
- MAPE stability: If many minutes have tiny observed totals, consider raising MAPE_MIN_DENOM (e.g., 10) or prefer sMAPE/WAPE.
- Movement subset rule: If 0 means missing rather than true zero in observations, keep OBS_POS_THRESHOLD > 0.
  If 0 is a valid count, set it to 0.0.
- Units: Ensure cap is per-minute, and t_free is in seconds; radio ∈ [0,1].
- BPR parameters: You can narrow/widen α,β bounds if the corridor shows different sensitivity.

Reproducibility checklist
-------------------------
- Put files under data/ or the exact relative paths listed above.
- Confirm output/ exists or let the scripts create it.
- Record your epochs, learning rate, and chosen HAS_DATA_THRESH/MAPE_DENOM_THRESH.
- Save the metrics.xlsx and plots alongside your commit for traceability.

Acknowledgements
----------------
- Sage node data: https://vto.sagecontinuum.org/nodes
- HERE TMC data (travel times)
- SUMO network files for Roosevelt Rd
