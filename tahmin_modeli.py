import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

CSV = "haftalik_hava_kalitesi.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # Zorunlu kolon kontrolü
    required = {"date","city","pm2_5","temp","humidity","wind_speed"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV eksik kolonlar. Gerekli: {required}")

    # datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city","date"]).reset_index(drop=True)

    # city bazlı feature mühendisliği: lag ve rolling
    out_frames = []
    for city, g in df.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        # laglar
        for lag in [1,2,3]:
            g[f"pm2_5_lag{lag}"] = g["pm2_5"].shift(lag)
        # rolling
        g["pm2_5_roll3"] = g["pm2_5"].rolling(window=3, min_periods=1).mean().shift(1)
        # zaman özellikleri
        g["weekday"] = g["date"].dt.weekday
        g["dayofyear"] = g["date"].dt.dayofyear
        out_frames.append(g)
    df2 = pd.concat(out_frames, ignore_index=True)

    # target: ertesi günün pm2_5 (günlük tahmin hedefi)
    df2 = df2.sort_values(["city","date"]).reset_index(drop=True)
    df2["pm2_5_target"] = df2.groupby("city")["pm2_5"].shift(-1)

    # Drop last day per city where target NaN
    df2 = df2[~df2["pm2_5_target"].isna()].copy()

    # Basit eksik değer işlemi: lag eksikleri olanları bırak (az veri yerine güven)
    df2 = df2.dropna().reset_index(drop=True)

    return df2

def features_and_target(df):
    feature_cols = [
        "pm2_5", "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3",
        "pm2_5_roll3", "temp", "humidity", "wind_speed",
        "weekday", "dayofyear"
    ]
    X = df[feature_cols].astype(float)
    y = df["pm2_5_target"].astype(float)
    return X, y, feature_cols

def time_split_train_test(df, frac_train=0.8):
    # global zaman sırası: use date sorted across entire dataset per city groups interleaved
    # Simpler: split by index preserving time order
    n = len(df)
    n_train = int(n * frac_train)
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n)
    return train_idx, test_idx

def evaluate_and_save(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # save model
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
    return y_pred, mae, rmse, r2

def plot_actual_vs_pred(df_test, y_test, y_pred, title, filename):
    plt.figure(figsize=(8,4))
    plt.plot(df_test["date"].values, y_test, label="Actual", marker='o')
    plt.plot(df_test["date"].values, y_pred, label="Predicted", marker='x')
    plt.xticks(rotation=45)
    plt.ylabel("PM2.5")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Grafik kaydedildi: {filename}")

def main():
    print("Veri yükleniyor ve hazırlanıyor...")
    df = load_and_prepare(CSV)
    if len(df) < 10:
        print("UYARI: Veri sayısı çok az (<10). Model sonuçları güvenilir olmayabilir.")
    X, y, feat = features_and_target(df)

    # time split
    train_idx, test_idx = time_split_train_test(df, frac_train=0.8)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    df_test = df.iloc[test_idx]

    # Baseline: persistence (bugün = yarın)
    baseline_pred = X_test["pm2_5"].values  # persistence baseline
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"\nBasline (persistence) MAE: {baseline_mae:.3f}, RMSE: {baseline_rmse:.3f}")

    # Linear Regression
    lr = LinearRegression()
    y_pred_lr, mae_lr, rmse_lr, r2_lr = None, None, None, None
    try:
        y_pred_lr, mae_lr, rmse_lr, r2_lr = evaluate_and_save(lr, X_train, y_train, X_test, y_test, "linear_regression")
    except Exception as e:
        print("Linear regression hata:", e)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    y_pred_rf, mae_rf, rmse_rf, r2_rf = evaluate_and_save(rf, X_train, y_train, X_test, y_test, "random_forest")

    # Plot (use RF results if available, else LR)
    if y_pred_rf is not None:
        plot_actual_vs_pred(df_test, y_test, y_pred_rf, "Actual vs Predicted (RandomForest)", "rf_actual_vs_pred.png")
    elif y_pred_lr is not None:
        plot_actual_vs_pred(df_test, y_test, y_pred_lr, "Actual vs Predicted (LinearRegression)", "lr_actual_vs_pred.png")

    # Feature importances for RF
    try:
        importances = rf.feature_importances_
        feat_imp = pd.Series(importances, index=feat).sort_values(ascending=False)
        print("\nFeature importances (RandomForest):")
        print(feat_imp)
        feat_imp.to_csv(os.path.join(MODEL_DIR, "feature_importances.csv"))
        print("Feature importance kaydedildi.")
    except Exception as e:
        print("Feature importance hata:", e)

    print("\nBitti. Modeller models/ dizinine kaydedildi. Grafikleri kontrol et.")

if __name__ == "__main__":
    main()
