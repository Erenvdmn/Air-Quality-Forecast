import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını oku
df = pd.read_csv("haftalik_hava_kalitesi.csv")

# Stil ayarı
sns.set(style="whitegrid", font_scale=1.1)

# 1) PM2.5 vs Sıcaklık

# Grafik oluştur
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="temp", y="pm2_5", scatter=True, color="blue", line_kws={"color": "red"})

# Noktaların yanına şehir isimlerini ekle
for i in range(len(df)):
    plt.text(df["temp"][i] + 0.1, df["pm2_5"][i] + 0.3, df["city"][i], fontsize=8, alpha=0.7)

plt.title("PM2.5 vs Sıcaklık (°C)")
plt.xlabel("Sıcaklık (°C)")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.show()

# 2) PM2.5 vs Rüzgar Hızı
plt.figure(figsize=(8,5))
sns.regplot(x="wind_speed", y="pm2_5", data=df, scatter=True, line_kws={'color':'red'})
for i in range(len(df)):
    plt.text(df["wind_speed"][i] + 0.1, df["pm2_5"][i] + 0.3, df["city"][i], fontsize=8, alpha=0.7)

plt.title("PM2.5 vs Rüzgar Hızı (m/s)")
plt.xlabel("Rüzgar Hızı (m/s)")
plt.ylabel("PM2.5 (µg/m³)")
plt.show()

# 3) PM2.5 vs Nem
plt.figure(figsize=(8,5))
sns.regplot(x="humidity", y="pm2_5", data=df, scatter=True, line_kws={'color':'red'})
for i in range(len(df)):
    plt.text(df["humidity"][i] + 0.1, df["pm2_5"][i] + 0.3, df["city"][i], fontsize=8, alpha=0.7)
plt.title("PM2.5 vs Nem (%)")
plt.xlabel("Nem (%)")
plt.ylabel("PM2.5 (µg/m³)")
plt.show()
