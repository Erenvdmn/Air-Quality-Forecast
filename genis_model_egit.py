import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Ayarlar
CSV_FILE = "genis_hava_kalitesi.csv"
MODEL_DIR = "modeller"
GRAPH_DIR = "grafikler"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Görselleştirme Ayarları (Rapor için şık görünsün)
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def train_and_visualize():
    print("⏳ Veri yükleniyor ve hazırlanıyor...")
    
    # 1. VERİ HAZIRLIĞI
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # Feature Engineering
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["date"].dt.weekday >= 5
    df["city"] = df["city"].astype("category")
    
    # Lag Features (AQI üzerinden)
    df = df.sort_values(["city", "date"])
    for lag in [1, 2]:
        df[f"aqi_lag{lag}"] = df.groupby("city")["aqi"].shift(lag)
    
    # HEDEF: Yarının AQI değeri
    df["target_aqi"] = df.groupby("city")["aqi"].shift(-1)
    
    # Boş verileri temizle
    df = df.dropna()
    
    # Girdiler
    features = [
        "city", "month", "day_of_year", "is_weekend",
        "temp", "humidity", "wind_speed", "wind_dir", "rain",
        "aqi", "aqi_lag1", "aqi_lag2",
        "pm10", "pm2_5", "no2", "o3"
    ]
    
    X = df[features]
    y = df["target_aqi"]
    
    # Veriyi Bölme (Son %10 Test, İlk %90 Eğitim)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    print(f"🧠 Model eğitiliyor... (Eğitim Verisi: {len(X_train)} satır)")
    
    # Model Eğitimi
    model = HistGradientBoostingRegressor(categorical_features=[0], max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Metrikler
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("-" * 40)
    print(f"✅ MODEL SONUÇLARI:")
    print(f"   R² Skoru (Başarı): {r2:.3f}")
    print(f"   MAE (Ortalama Hata): {mae:.2f}")
    print(f"   RMSE (Karesel Hata): {rmse:.2f}")
    print("-" * 40)
    
    # Modeli Kaydet
    joblib.dump(model, f"{MODEL_DIR}/aqi_model.joblib")
    print("💾 Model kaydedildi.")

    # --- GRAFİK 1: GERÇEK vs TAHMİN (SCATTER PLOT) ---
    print("📊 Grafik 1 çiziliyor: Gerçek vs Tahmin...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#4c72b0", edgecolor=None)
    
    # İdeal Doğru (y=x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="İdeal Tahmin Çizgisi")
    
    plt.title(f"Model Doğruluğu: Gerçek vs Tahmin (R²: {r2:.2f})")
    plt.xlabel("Gerçek AQI Değeri (İstasyon/Uydu)")
    plt.ylabel("Model Tahmini AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/1_gercek_vs_tahmin.png", dpi=300)
    plt.close()

    # --- GRAFİK 2: ZAMAN SERİSİ (SON 100 GÜN) ---
    print("📊 Grafik 2 çiziliyor: Zaman Serisi...")
    plt.figure(figsize=(12, 6))
    # Test setinin son 100 verisini alalım ki grafik karışmasın
    limit = 100
    plt.plot(np.arange(limit), y_test.values[:limit], label="Gerçek Değer", color="black", linewidth=2)
    plt.plot(np.arange(limit), y_pred[:limit], label="Model Tahmini", color="orange", linestyle="--", linewidth=2)
    
    plt.title("Zaman İçindeki Değişim (Son 100 Test Verisi)")
    plt.xlabel("Günler")
    plt.ylabel("AQI Değeri")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/2_zaman_serisi_tahmin.png", dpi=300)
    plt.close()

    # --- GRAFİK 3: ÖZNİTELİK ÖNEM DÜZEYİ (FEATURE IMPORTANCE) ---
    # Not: HistGradientBoosting'de feature_importances_ yoktur, Permutation Importance kullanılır.
    print("📊 Grafik 3 çiziliyor: Özelliklerin Önemi (Bu biraz sürebilir)...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(features)[sorted_idx], result.importances_mean[sorted_idx], color="#55a868")
    plt.title("Hava Kalitesini En Çok Etkileyen Faktörler")
    plt.xlabel("Etki Düzeyi (Önem Skoru)")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/3_etkileyen_faktorler.png", dpi=300)
    plt.close()
    
    print(f"\n🎉 İşlem Tamam! Grafikler '{GRAPH_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    train_and_visualize()