# Roosevelt_Calibration — BPR + Gradient Descent Calibration & Evaluation

A compact toolkit for modeling and evaluating traffic flow on **Roosevelt Rd, Chicago, IL 60607** using:

1. Movement counts from **Sage** nodes  
2. Per-minute travel times from **HERE (TMC)**  
3. A **physics-guided learning model** combining a BPR function with gradient descent (PyTorch)

The workflow is:

1. Reconstruct route–movement relationships via an **A-matrix**.  
2. Calibrate path flows and BPR parameters jointly using a **BPR+GD hybrid model**.  
3. Evaluate performance at the **(Node, Direction)** level and as **per-minute aggregates**.

---

## 📁 Repository Layout

```text
.
├─ data/
│  ├─ Movement_Data.xlsx            # Observed movements: Time, Node, Direction, Value
│  ├─ Route_Movement_Matrix.xlsx    # A-matrix: first 3 meta cols, then route columns
│  ├─ TMC_data.csv                  # HERE TMC travel times (per minute)
│  ├─ TMC_Identification.csv        # HERE TMC ID ↔ length (miles), etc.
│  └─ TMC_mapping.xlsx              # Final mapping & params (lane, cap, ref_speed, Node)
│
├─ output/
│  ├─ export_data.xlsx              # Predicted path flows X: Time + R route columns
│  └─ evaluation/
│     ├─ error_data.xlsx            # Per-node metrics + overall summary
│     ├─ observed_heatmap.png       # Heatmap of observed movements
│     ├─ predicted_heatmap_gd.png   # Heatmap of predicted movements (BPR+GD)
│     ├─ per_minute_errors.xlsx     # Per-minute totals & errors
│     ├─ worst_minutes_top50.xlsx   # Top-50 worst minutes by AE
│     ├─ per_minute_series.png      # Observed vs Predicted per-minute series
│     └─ per_minute_abs_error.png   # Absolute error per minute
│
├─ Finals_BPR_GD.py                 # BPR+GD calibration (PyTorch)
└─ Evaluate.py                      # Evaluation: per-node + per-minute
```

If a folder under `output/` is missing, the scripts will create it automatically.

---

## 🧩 Environment

- **Python** ≥ 3.9  
- Required packages:
  - `pandas`, `numpy`, `torch`, `tqdm`, `geopy`, `openpyxl`,  
    `matplotlib`, `seaborn`, `scikit-learn`, `pytz`

Quick installation:

```bash
pip install pandas numpy torch tqdm geopy openpyxl matplotlib seaborn scikit-learn pytz
```

---

## 📊 Data Assumptions & Units

- **Time resolution**: 1 minute  
  All inputs are aligned at the **minute** level.

- **A-matrix (`Route_Movement_Matrix.xlsx`)**  
  Maps **routes** (R) to **intersection movements** (L).
  - First 3 columns: meta information (e.g., Node, Direction, …).
  - Remaining columns: one column per route.

- **Movement definition**  
  `(Node, Direction)` denotes lane movements within one approach  
  (e.g., Left / Through / Right of a given approach).

- **BPR parameters & units**
  - `t_free` (seconds) derived from `miles / reference_speed * 3600`.
  - `cap` is **per-minute capacity** for the whole approach  
    (already lane-adjusted: `cap * lane`).
  - `radio` is the overlap ratio `∈ [0,1]` between a SUMO segment and a TMC segment  
    (by longitude span).

- **Time zone**  
  Scripts normalize timestamps to **America/Chicago** for joins and plots.

---

## ⚙️ Method — BPR + Gradient Descent (Physics-Guided)

The method combines a classical BPR travel-time model with a learnable, gradient-based calibration in PyTorch.

### Variables

- **Path flows**:  
  \(X_{r,t} ≥ 0\) — decision variables, optimized per route per minute.

- **Movements**:  
  \(C_{l,t} = \sum_r A_{l,r} X_{r,t}\).

- **Network travel time (observed)**:  
  \(y_{\text{obs},t}\) — per-minute travel time, aggregated from HERE TMCs.

### BPR travel-time model (learnable)

For each mapped row (Node, `tmc_id`):

\[
y_{l,t} = t_{\text{free},l} \cdot radio_l + t_{\text{free},l} \cdot \alpha \cdot radio_l
\left( \frac{\text{flow}_{\text{node}(l),t}}{cap_l} \right)^{\beta}
\]

- The **flow** used in BPR is the **approach-level sum** of movements belonging to the same Node (L + T + R), consistent with capacity being the whole-approach capacity.
- \(\alpha, \beta\) are **global learnable scalars** with bounds, e.g.:
  - \(\alpha ∈ [0.01, 1.0]\)
  - \(\beta ∈ [1, 8]\)
- A small prior (anchor) is placed at \((\alpha, \beta) = (0.15, 4.0)\).

### Loss (multi-task, normalized)

The training objective is:

\[
L = \text{MSE}\left(\frac{C - C_{\text{obs}}}{\sigma_C}\right)
  + \text{MSE}\left(\frac{y - y_{\text{obs}}}{\sigma_y}\right)
  + \lambda_\alpha (\alpha - 0.15)^2
  + \lambda_\beta (\beta - 4)^2
\]

- Normalization by empirical standard deviations \(\sigma_C, \sigma_y\) helps balance the two tasks.
- Non-negativity on \(X\) is enforced via **ReLU**.
- Optimization uses **Adam** (with optional gradient clipping).

### Outputs

The calibration script writes:

- `output/export_data.xlsx` — per-minute **path flows** \(X\),  
  with one column per route.  
These flows are then consumed by `Evaluate.py` for movement-level and time-series evaluation.

---

## ▶️ How to Run

Run all commands from the repository root. Ensure that the input files under `data/` exist.

### 1) Calibrate with BPR+GD

This step learns:

- Path flows \(X_{r,t}\)
- Global BPR parameters \(\alpha, \beta\)

```bash
python Finals_BPR_GD.py
```

This produces:

- `output/export_data.xlsx` — calibrated path flows X.

### 2) Evaluate (per-node + per-minute)

```bash
python Evaluate.py
```

This consumes:

- `data/Movement_Data.xlsx` (observed movements)
- `data/Route_Movement_Matrix.xlsx` (A-matrix)
- `data/TMC_data.csv` (HERE travel times)
- `output/export_data.xlsx` (model path flows)

and generates evaluation plots and tables under `output/evaluation/`.

---

## 📈 Outputs & Evaluation

After running both steps, you will obtain:

### 1. Movement-level & node-level metrics

- `output/evaluation/error_data.xlsx`
  - `per_node` sheet:
    - MAE, RMSE, MAPE, sMAPE per `(Node, Direction)`
  - `overall` summary:
    - Column-wise averages over nodes
  - `per_minute` sheet:
    - Observed/predicted totals and per-minute AE/APE/sMAPE

- Heatmaps:
  - `observed_heatmap.png` — observed movements.
  - `predicted_heatmap_gd.png` — predicted movements (BPR+GD).

### 2. Per-minute aggregate series

- `per_minute_series.png` — observed vs. predicted per-minute totals.
- `per_minute_abs_error.png` — per-minute absolute error.
- `per_minute_errors.xlsx` — underlying per-minute data and error metrics.
- `worst_minutes_top50.xlsx` — top 50 worst minutes by absolute error (AE).

#### Per-minute aggregation rule

For each minute:

- Sum only those movements that have an **observation at that minute**  
  (threshold configurable inside the script via `OBS_POS_THRESHOLD`, default `> 0`).

Error metrics on the per-minute series include:

- AE (absolute error)
- sMAPE
- MAPE (only when denominator ≥ `MAPE_MIN_DENOM`, default 5)
- WAPE

---

## 💡 Practical Tips

- **Time alignment**  
  Make sure all inputs share the same clock.  
  The scripts convert timestamps to **America/Chicago** and align at **minute** strings.

- **Route consistency**  
  Route columns in `Route_Movement_Matrix.xlsx` must match those in `output/export_data.xlsx`.

- **MAPE stability**  
  If many minutes have tiny observed totals:
  - Consider raising `MAPE_MIN_DENOM` (e.g., from 5 to 10), or
  - Rely more on sMAPE / WAPE for robustness.

- **Movement subset rule**  
  - If `0` means **missing** rather than a true zero in observations, keep `OBS_POS_THRESHOLD > 0`.  
  - If `0` is a valid count, set `OBS_POS_THRESHOLD = 0.0`.

- **Units sanity check**
  - `cap`: per-minute capacity of the whole approach (already lane-adjusted).
  - `t_free`: travel time in **seconds**.
  - `radio`: overlap ratio in `[0, 1]`.

- **BPR parameters**
  - You can narrow/widen bounds on \(\alpha, \beta\) if the corridor exhibits different sensitivity.
  - Record the final \(\alpha, \beta\) for reporting and reproducibility.

---

## ✅ Reproducibility Checklist

To reproduce results:

1. Place all required files under `data/` using the exact relative paths in this README.
2. Confirm that `output/` exists or let the scripts create it.
3. Run `python Finals_BPR_GD.py` then `python Evaluate.py`.
4. Record:
   - Number of epochs, learning rate, optimizer settings.
   - `HAS_DATA_THRESH`, `MAPE_DENOM_THRESH`, and related thresholds.
5. Save `error_data.xlsx`, other `.xlsx` outputs, and plots in `output/evaluation/` alongside your code commit.

---

## 🙏 Acknowledgements

- Sage node data: https://vto.sagecontinuum.org/nodes  
- HERE TMC travel time data  
- SUMO network files for Roosevelt Rd

This codebase is intended for research and prototyping purposes; please cite appropriately if used in academic work.
