import requests
import pandas as pd
import datetime
import time
import os
import dotenv

dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

cities = {
    "Ankara": (39.93, 32.85),
    "Istanbul": (41.01, 28.97),
    "Izmir": (38.42, 27.14),
    "Malatya": (38.35, 38.31)
}

def get_data():
    data = []
    for city, (lat, lon) in cities.items():
        # Hava kalitesi
        aq_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        aq_res = requests.get(aq_url).json()
        aq = aq_res["list"][0]["components"]

        # Hava durumu
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        w_res = requests.get(w_url).json()
        main = w_res["main"]
        wind = w_res["wind"]

        d = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "city": city,
            "co": aq["co"],
            "no": aq["no"],
            "no2": aq["no2"],
            "o3": aq["o3"],
            "so2": aq["so2"],
            "pm2_5": aq["pm2_5"],
            "pm10": aq["pm10"],
            "nh3": aq["nh3"],
            "temp": main["temp"],
            "humidity": main["humidity"],
            "wind_speed": wind["speed"]
        }
        data.append(d)
        time.sleep(1)
    return data

# Dosya varsa ekle, yoksa oluştur
file = "haftalik_hava_kalitesi.csv"
new_data = get_data()
df_new = pd.DataFrame(new_data)

if os.path.exists(file):
    df_old = pd.read_csv(file)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_all = df_new

df_all.to_csv(file, index=False)
print(f"Veriler kaydedildi: {file}")
