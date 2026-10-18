# Expiry-Aware Implied Volatility Imputation for NIFTY Options

**Context & Affiliation**: Finance Club Open Project, IIT Roorkee  
**Domain**: Quantitative Finance, Derivatives Pricing, Surface Imputation & Time-Series Modeling  

This project resolves missing implied-volatility (IV) values across call and put strikes for NIFTY index options. The final solution combines an iterative multivariate imputer (PowerMICE) with a local 2D time-strike volatility surface smoother.

---

## Performance Summary

- **Best Actual MSE**: `0.0000658973`
- **Final Model Architecture**:
  ```text
  0.90 * PowerMICE(power=1.0, alpha=0.03, rich_features) + 0.10 * Local 2D IV Surface Smoother
  ```

---

## 1. Problem Statement & Dataset
The dataset contains timestamped NIFTY option IV observations across 28 call and put strike columns (`NIFTY27JAN26`) alongside spot underlying prices. Option liquidity variations cause missing IV cells across strikes and timestamps, requiring robust surface imputation without lookahead leakage.

- `dataset.csv`: Historical 5-minute option chain timestamps, underlying price, and 28 option IV series.

---

## 2. Key Methodological Findings

### Native-Pattern Masking vs. Random Masking
Uniform random masking yielded unrealistically optimistic validation errors that failed to generalize. We engineered a native-pattern masking validator: for each row, mask the exact distribution of observed cells that matched empirical missingness patterns.

### Expiry-Day Regimes ($DTE < 1$ vs. $DTE \ge 1$)
Decomposing IV across log-moneyness revealed sharp regime shifts between expiry-day and non-expiry sessions. Expiry-day observations exhibited steep local skew and curvature, requiring specialized local time-decay constraints.

### PowerMICE Backbone
PowerMICE implements a transformed iterative ridge imputation scheme:
- Iteratively models each option strike column conditional on all peer strikes.
- Integrates row-level market features: underlying spot price, days-to-expiry (DTE), intraday timestamp, and short-term underlying momentum.
- Applies ridge regularization ($\alpha = 0.03$) to stabilize collinear strike relationships.

### Local 2D Volatility Surface Smoothing
A kernel-weighted 2D smoother fits nearby observations across both temporal and strike dimensions independently for CE and PE surfaces, with tighter temporal windows on expiry day to handle pin risk.

---

## 3. How To Run

### Environment Setup
```bash
pip install -r requirements.txt
```

### Execute Final Pipeline
```bash
python run_final_model.py
```

Outputs generated:
```text
outputs/filled_blend_p1rich03_90_surface2d10.csv
outputs/submission_blend_p1rich03_90_surface2d10.csv
```

Interactive analysis notebook:
```text
final_model_p1rich03_surface2d10.ipynb
```

---

## 4. Repository Structure & Deliverables

- `dataset.csv`: Raw option chain dataset.
- `final_model_p1rich03_surface2d10.ipynb`: End-to-end exploration, missingness EDA, IV surface visualization, and model blend.
- `run_final_model.py`: Standalone execution script for the final PowerMICE + 2D smoother blend.
- `robust_imputation_search.py`: Benchmarking script for baseline imputation strategies.
- `imputation_work_summary.md`: Detailed validation methodology notes.
- `Finclub_Open_Project_2.pdf`: Original problem statement and project guidelines.
