import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CSV oku
df = pd.read_csv("haftalik_hava_kalitesi.csv")

# Klasör oluştur (yoksa)
os.makedirs("grafikler", exist_ok=True)

# Tarihi sıralama (grafikler düzgün çıksın)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["city", "date"])

# 1) PM2.5 değişimi (zaman serisi)
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="date", y="pm2_5", hue="city", marker="o")
plt.title("PM2.5 Değişimi (Şehir Bazında)")
plt.xlabel("Tarih")
plt.ylabel("PM2.5 (µg/m³)")
plt.grid(True)
plt.legend(title="Şehir")
plt.savefig("grafikler/pm25_zaman.png")
plt.close()

# 2) Sıcaklık değişimi
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="date", y="temp", hue="city", marker="o")
plt.title("Sıcaklık Değişimi (Şehir Bazında)")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık (°C)")
plt.grid(True)
plt.legend(title="Şehir")
plt.savefig("grafikler/sicaklik_zaman.png")
plt.close()

# 3) PM2.5 - Nem ilişkisi (scatter)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="humidity", y="pm2_5", hue="city", s=80)
plt.title("Nem ↔ PM2.5 İlişkisi")
plt.xlabel("Nem (%)")
plt.ylabel("PM2.5 (µg/m³)")
plt.grid(True)
plt.legend(title="Şehir")
plt.savefig("grafikler/nem_pm25.png")
plt.close()

print("Grafikler kaydedildi: 'grafikler/'")
