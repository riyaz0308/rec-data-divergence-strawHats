# Track 1: Machine Failure Prediction (Industrial Intelligence)
**Hackathon:** Data Divergence — Rajalakshmi Engineering College (DEVS REC)  
**Team Name:** StrawHats  

---

##  Project Overview
An end-to-end predictive maintenance machine learning pipeline built to detect imminent CNC machine failures from high-frequency telemetry while minimizing false alarms.

##  Key Results
- **Official Validation PR-AUC:** `0.8825`
- **Optimal Decision Threshold:** `0.65`
- **Official Validation F1-Score:** `0.8429` (+26.1% over raw baseline)

## ⚙️ Physics-Informed Engineered Features
1. **Mechanical Power (`power_W`):** $\tau \times \omega$ to detect electrical motor overload and spindle stalls.
2. **Thermal Dissipation Gradient (`temp_diff`):** Process Temp − Air Temp to capture cooling failure.
3. **Overstrain Index:** $\text{Torque} \times \text{Tool Wear [min]}$ to flag sudden cutter breakage.

## 📂 Repository Structure
- `solution.ipynb` : Complete EDA, physics-based feature engineering, and Stratified 5-Fold CV training.
- `predict.py` : Production-ready inference script accepting raw test data.
- `models/trained_pipeline.joblib` : Serialized LightGBM model artifact & optimal threshold.
- `Methodology_Summary.pdf` : 2-page detailed technical and operational methodology report.
