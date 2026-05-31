# Machine Learning Approaches for Predicting and Classifying Vehicle CO₂ Emissions

An engineering-focused machine learning project utilizing real-world data to analyze, estimate, and classify vehicular carbon dioxide emissions based on engine profiles and fuel consumption metrics.

## 📌 Project Overview
With global emphasis shifting heavily toward environmental sustainability and electric mobility transitions, modeling traditional internal combustion engine (ICE) carbon footprints remains critical. This project applies supervised machine learning workflows to evaluate a dataset of vehicle profiles, tackling two core tasks:
1. **Regression:** Estimating precise continuous $CO_2$ emission outputs ($\text{g/km}$).
2. **Classification:** Evaluating whether a vehicle meets regulatory compliance thresholds based on emission benchmarks.

## 📊 Dataset Profile
The analysis uses real-world vehicular data (`CO2 Emissions_Canada.csv`) encompassing **7,385 vehicle entries** with technical parameters including:
* Engine Size (L) & Cylinder Count
* Transmission type and Fuel Type
* Multi-cycle Fuel Consumption metrics (City, Highway, and Combined)

## ⚙️ Methodology & Models Implemented
Data preprocessing includes categorical feature mapping, scaling via `StandardScaler`, and rigorous train-test splitting. The following architectures are implemented and compared:

### 1. Regression Frameworks
* **Linear Regression:** Leveraged to capture strong linear dynamics present between fuel consumption parameters and mass emission outputs.
* **Decision Tree Regressor:** Implemented to study non-linear cross-feature interactions.

### 2. Classification Frameworks
* **Logistic Regression:** Used to establish baseline statistical classification boundaries for vehicle emissions compliance.
* **Decision Tree Classifier:** Implemented to extract rule-based decision trees and rank feature importance weights.

## 📈 Key Performance Metrics
* **Linear Regression:** $R^2$ Score ~ `0.99` (Train & Test)
* **Decision Tree Regressor:** $R^2$ Score ~ `0.95` (Train) / `0.94` (Test)
* **Logistic Regression:** Accuracy ~ `99.5%` (Train) / `99.4%` (Test)
* **Decision Tree Classifier:** Accuracy ~ `97.4%` (Train) / `97.1%` (Test)

## 🛠️ Requirements & Setup
To run the notebooks locally, ensure you have the standard scientific computing stack installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn