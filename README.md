# HKE Air Quality Analysis Project

Collects weekly air quality and weather data for selected Turkish cities, saves it to CSV, generates visualizations, and trains prediction models.

## Project files
- **haftalik_veri_topla.py** — collects air quality & weather data from OpenWeatherMap API and appends to `haftalik_hava_kalitesi.csv`
- **grafik_analiz.py** — generates time-series and scatter plots, saves to `grafikler/` folder
- **tahmin_modeli.py** — trains/evaluates PM2.5 prediction models (Linear Regression, Random Forest), saves to `models/`
- **rf_actual_vs_pred.png** — shows the result of `tahmin_modeli.py` with graphs
- **haftalik_hava_kalitesi.csv** — collected raw data
- **grafikler/** — output PNG files
- **models/** — saved trained models and feature importance CSV
- **.env** — (not committed) store API key: `API_KEY=your_openweathermap_api_key`

## Requirements
- Python 3.8+
- Packages: pandas, matplotlib, seaborn, requests, python-dotenv, scikit-learn, joblib, numpy

Install (Windows PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install pandas matplotlib seaborn requests python-dotenv scikit-learn joblib numpy
```

## Setup
1. Create `.env` file in project root with your OpenWeatherMap API key:
   ```
   API_KEY=your_api_key_here
   ```
2. Activate virtual environment and install dependencies.

## Usage
**Collect data:**
```powershell
python haftalik_veri_topla.py
```
Output: `haftalik_hava_kalitesi.csv`

**Generate plots:**
```powershell
python grafik_analiz.py
```
Output: `grafikler/pm25_zaman.png`, `sicaklik_zaman.png`, `nem_pm25.png`

**Train models:**
```powershell
python tahmin_modeli.py
```
Output: `models/linear_regression.joblib`, `models/random_forest.joblib`, `models/feature_importances.csv`, `rf_actual_vs_pred.png`

## Troubleshooting
- **Import error**: install missing package:
  ```powershell
  python -m pip install seaborn python-dotenv scikit-learn
  ```
- **API timeout**: check network/proxy connection.
- **CSV not found**: run `haftalik_veri_topla.py` first to generate data.

## License
Choose an appropriate license for this project.
