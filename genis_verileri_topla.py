import requests
import pandas as pd
import time

# --- AYARLAR ---
CITIES = {
    "Malatya":  {"lat": 38.3552, "lon": 38.3095},
    "Istanbul": {"lat": 41.0082, "lon": 28.9784},
    "Ankara":   {"lat": 39.9334, "lon": 32.8597},
    "Izmir":    {"lat": 38.4192, "lon": 27.1287},
    "Bursa":    {"lat": 40.1826, "lon": 29.0662}
}

START = "2023-01-01"
# DİKKAT: Bitiş tarihini dünden 5 gün öncesi yapıyoruz ki API kesin veri döndürsün
from datetime import date, timedelta
END = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")

def get_data():
    all_data = []
    print(f"Veriler çekiliyor... ({START} - {END})")
    
    for city, coords in CITIES.items():
        # 1. Hava Kalitesi (Tüm Gazlar)
        url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params_aq = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": START, "end_date": END,
            "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone,european_aqi", # HEPSİNİ İSTİYORUZ
            "timezone": "auto"
        }
        
        # 2. Hava Durumu
        url_w = "https://archive-api.open-meteo.com/v1/archive"
        params_w = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": START, "end_date": END,
            "hourly": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }
        
        try:
            r_aq = requests.get(url_aq, params=params_aq).json()
            r_w = requests.get(url_w, params=params_w).json()
            
            df_aq = pd.DataFrame(r_aq["hourly"])
            df_w = pd.DataFrame(r_w["hourly"])
            
            # Zaman sütununu datetime yap
            df_aq["time"] = pd.to_datetime(df_aq["time"])
            df_w["time"] = pd.to_datetime(df_w["time"])
            
            # İki tabloyu birleştir
            df = pd.merge(df_w, df_aq, on="time")
            
            # İsimleri düzelt
            df = df.rename(columns={
                "temperature_2m": "temp", "relative_humidity_2m": "humidity",
                "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
                "european_aqi": "aqi", "nitrogen_dioxide": "no2", "ozone": "o3"
            })
            
            # Günlük Özet (Resample)
            # Mantık: Kirlilikte 'Maksimum' değer riski belirler.
            df_daily = df.resample("D", on="time").agg({
                "temp": "mean", "humidity": "mean", "wind_speed": "mean", "wind_dir": "mean", "rain": "sum",
                "pm10": "max", "pm2_5": "max", "no2": "max", "o3": "max", "aqi": "max"
            }).reset_index()
            
            df_daily["city"] = city
            df_daily.rename(columns={"time": "date"}, inplace=True)
            
            # Veri setinde boşluk varsa temizle (Eğitim verisi temiz olmalı)
            df_daily = df_daily.dropna()
            
            all_data.append(df_daily)
            print(f"✅ {city} tamamlandı.")
            
        except Exception as e:
            print(f"❌ Hata {city}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv("genis_hava_kalitesi.csv", index=False)
        print("\n🎉 Tüm veriler 'genis_hava_kalitesi.csv' dosyasına kaydedildi.")
        print(final_df.head())

if __name__ == "__main__":
    get_data()