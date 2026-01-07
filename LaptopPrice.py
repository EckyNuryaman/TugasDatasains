import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Input CSV
df = pd.read_csv('laptop_price.csv', encoding='latin-1')

# Membersihkan data GB dan Kg pada data Ram dan Weight
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# mengnalisis Harga  terbesar dan terkecil
max_price = df['Price_euros'].max()
min_price = df['Price_euros'].min()

# Mencari data harga laptop
laptop_termahal = df[df['Price_euros'] == max_price].iloc[0]
laptop_termurah = df[df['Price_euros'] == min_price].iloc[0]

print("=== Analisis Harga Laptop ===")
print(f"Harga Termahal: €{max_price}")
print(f"Detail Laptop Termahal: {laptop_termahal['Company']} {laptop_termahal['Product']} - {laptop_termahal['Ram']}GB RAM")
print("-" * 30)
print(f"Harga Termurah: €{min_price}")
print(f"Detail Laptop Termurah: {laptop_termurah['Company']} {laptop_termurah['Product']} - {laptop_termurah['Ram']}GB RAM")
print("=" * 30)

# Melihat hubungan antara RAM (X) dan Harga (y)
plt.figure(figsize=(10, 6))
plt.scatter(df['Ram'], df['Price_euros'], alpha=0.5)
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euros)')
plt.title('Hubungan antara RAM dan Harga Laptop')
plt.grid(True)
plt.show()

# Harga RAM
X = df[['Ram']]
y = df['Price_euros']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bangun model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi harga pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
r_squared = model.score(X_test, y_test)
print(f"R-squared (Kekuatan hubungan RAM vs Harga): {r_squared:.4f}")

# Visualisasi hasil prediksi
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Data Asli (Test)', alpha=0.5)
plt.plot(X_test, y_pred, color='red', label='Garis Regresi (Prediksi)')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euros)')
plt.title('Prediksi Harga Laptop berdasarkan RAM')
plt.legend()
plt.grid(True)
plt.show()

# Inpit data baru yang tidak ada dalam data set
input_baru = [[64]] 
prediksi_harga = model.predict(input_baru)
print(f"Prediksi harga untuk laptop RAM 64GB: €{prediksi_harga[0]:.2f}")