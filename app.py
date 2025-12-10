import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AirQualityAI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI TASARIMI ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 14px;
        color: #666666 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #111111 !important;
        margin: 10px 0;
    }
    .metric-unit {
        font-size: 12px;
        color: #888888 !important;
    }
    /* Streamlit bileşenlerini zorla siyah yap */
    [data-testid="stMetricValue"] div { color: #000000 !important; }
    [data-testid="stMarkdownContainer"] p { color: ##ffffff !important; }
</style>
""", unsafe_allow_html=True)

# --- AYARLAR ---
MODEL_PATH = "modeller/aqi_model.joblib"
CITIES = {
    "Malatya":  {"lat": 38.3552, "lon": 38.3095},
    "Istanbul": {"lat": 41.0082, "lon": 28.9784},
    "Ankara":   {"lat": 39.9334, "lon": 32.8597},
    "Izmir":    {"lat": 38.4192, "lon": 27.1287},
    "Bursa":    {"lat": 40.1826, "lon": 29.0662}
}

# --- MODELİ YÜKLE ---
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

model = load_model()
if model is None:
    st.error("🚨 Model dosyası bulunamadı! Lütfen önce `genis_model_egit.py` dosyasını çalıştırın.")
    st.stop()

def get_live_data(city_name, lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "past_days": 5, "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,wind_direction_10m,pm10,pm2_5,nitrogen_dioxide,ozone,european_aqi",
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params)
        df = pd.DataFrame(r.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        
        df = df.rename(columns={
            "temperature_2m": "temp", "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
            "european_aqi": "aqi", "nitrogen_dioxide": "no2", "ozone": "o3", "rain": "rain"
        })

        # --- VERİ TEMİZLİĞİ ---
        df = df.interpolate(method='linear')
        df = df.ffill().bfill()
        
        # --- HATA ALINAN KISIM DÜZELTİLDİ ---
        # "TimedeltaIndex has no attribute abs" hatasını çözmek için:
        # İndeksi Series'e çevirip işlemi öyle yapıyoruz. Bu her sürümde çalışır.
        now = datetime.now()
        
        # 1. Zaman farklarını hesapla
        time_diffs = (df.index.to_series() - now).abs()
        
        # 2. En küçük farka sahip olan zamanın indeksini bul
        closest_time_idx = time_diffs.idxmin()
        
        # 3. O satırı seç
        current_row = df.loc[closest_time_idx]
        current_conditions = current_row.to_dict()

        # Sözlük içindeki NaN/None/0 kontrolü (Güvenlik)
        for key, val in current_conditions.items():
            if pd.isna(val) or val is None or val == 0.0:
                 try:
                     # Sütundaki 0 olmayan son değeri bulmaya çalış
                     non_zero_val = df[df[key] > 0][key].iloc[-1]
                     current_conditions[key] = non_zero_val
                 except:
                     # Hiç yoksa varsayılan
                     if key == 'pm2_5': current_conditions[key] = 25.0
                     elif key == 'pm10': current_conditions[key] = 40.0
                     elif key == 'no2': current_conditions[key] = 15.0
                     elif key == 'o3': current_conditions[key] = 60.0

        # --- GÜNLÜK ÖZET (MODEL GİRDİSİ) ---
        df_daily = df.resample("D").agg({
            "temp": "mean", "humidity": "mean", "wind_speed": "mean", "wind_dir": "mean", "rain": "sum",
            "aqi": "max", "pm10": "max", "pm2_5": "max", "no2": "max", "o3": "max"
        })
        
        tomorrow = datetime.now().date() + timedelta(days=1)
        
        if tomorrow in df_daily.index.date:
            target_row = df_daily.loc[df_daily.index.date == tomorrow].iloc[0].copy()
        else:
            target_row = df_daily.iloc[-1].copy()
            
        target_idx = df_daily.index.get_loc(target_row.name)
        today_agg = df_daily.iloc[target_idx - 1]

        target_row["month"] = target_row.name.month
        target_row["day_of_year"] = target_row.name.dayofyear
        target_row["is_weekend"] = 1 if target_row.name.weekday() >= 5 else 0
        target_row["city"] = city_name
        
        target_row["aqi"] = today_agg["aqi"]
        target_row["aqi_lag1"] = today_agg["aqi"]
        target_row["aqi_lag2"] = df_daily.iloc[target_idx - 2]["aqi"]
        
        target_row["pm10"] = today_agg["pm10"]
        target_row["pm2_5"] = today_agg["pm2_5"]
        target_row["no2"] = today_agg["no2"]
        target_row["o3"] = today_agg["o3"]
        
        features = ["city", "month", "day_of_year", "is_weekend",
                    "temp", "humidity", "wind_speed", "wind_dir", "rain",
                    "aqi", "aqi_lag1", "aqi_lag2",
                    "pm10", "pm2_5", "no2", "o3"]
        
        return pd.DataFrame([target_row[features]]), current_conditions

    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return None, None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌍 AirQuality AI")
    st.markdown("---")
    selected_city = st.selectbox("📍 Şehir Seçiniz", list(CITIES.keys()))
    if st.button("ANALİZ ET 🚀", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True
    st.markdown("---")
    st.caption("Veriler Open-Meteo Uydusu ile Anlık Çekilmektedir.")

# --- ANA EKRAN ---
st.title(f"Hava Kalitesi Raporu: {selected_city}")

if st.session_state.get('run_analysis'):
    with st.spinner("Uydudan son veriler alınıyor ve model çalışıyor..."):
        coords = CITIES[selected_city]
        X_input, current = get_live_data(selected_city, coords["lat"], coords["lon"])
        
        if X_input is not None:
            # TAHMİN
            pred = model.predict(X_input)[0]
            
            # DURUM BELİRLEME
            if pred < 20: 
                status = "Mükemmel"; bg_color = "#d4edda"; text_color = "#155724"; icon="🟢"
            elif pred < 40: 
                status = "İyi"; bg_color = "#fff3cd"; text_color = "#856404"; icon="🟡"
            elif pred < 60: 
                status = "Orta"; bg_color = "#ffeeba"; text_color = "#d63384"; icon="🟠"
            elif pred < 80: 
                status = "Kötü"; bg_color = "#f8d7da"; text_color = "#721c24"; icon="🔴"
            else: 
                status = "Tehlikeli"; bg_color = "#f5c6cb"; text_color = "#3d0e14"; icon="☠️"

            # 1. BÜYÜK SKOR KARTI
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:15px; border:1px solid {text_color}; text-align:center; margin-bottom:30px;">
                <h3 style="color:{text_color}; margin:0; opacity:0.8;">YARINKİ TAHMİNİ AQI</h3>
                <h1 style="color:{text_color}; font-size:64px; margin:5px 0; font-weight:800;">{pred:.0f}</h1>
                <h2 style="color:{text_color}; margin:0;">{icon} {status}</h2>
            </div>
            """, unsafe_allow_html=True)

            # YARDIMCI KART FONKSİYONU
            def create_card(title, value, unit):
                # None veya NaN kontrolü (Güvenlik)
                if value is None or pd.isna(value):
                    val_str = "0.0"
                else:
                    val_str = f"{value:.1f}"
                    
                return f"""
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{val_str}</div>
                    <div class="metric-unit">{unit}</div>
                </div>
                """

            # 2. ATMOSFERİK DEĞERLER (CANLI)
            st.subheader("🧪 Canlı Atmosferik Değerler")
            st.caption(f"Şu anki saate ({datetime.now().strftime('%H:%M')}) en yakın ölçüm değerleri:")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(create_card("PM2.5", current.get('pm2_5'), "µg/m³"), unsafe_allow_html=True)
            with c2: st.markdown(create_card("PM10", current.get('pm10'), "µg/m³"), unsafe_allow_html=True)
            with c3: st.markdown(create_card("NO2", current.get('no2'), "µg/m³"), unsafe_allow_html=True)
            with c4: st.markdown(create_card("OZON", current.get('o3'), "µg/m³"), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # 3. METEOROLOJİ
            st.subheader("🌤️ Meteorolojik Durum")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(create_card("Sıcaklık", current.get('temp'), "°C"), unsafe_allow_html=True)
            with m2: st.markdown(create_card("Nem", current.get('humidity'), "%"), unsafe_allow_html=True)
            with m3: st.markdown(create_card("Rüzgar Hızı", current.get('wind_speed'), "km/s"), unsafe_allow_html=True)
            with m4: st.markdown(create_card("Yağış", current.get('rain'), "mm"), unsafe_allow_html=True)

else:
    st.info("👈 Analize başlamak için lütfen sol menüden butona basın.")