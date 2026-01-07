import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate data penumpang
df = pd.read_csv("titanic.csv")
df.dropna()

# Bagi data menjadi fitur (X) dan target (y)
X = df[['age', 'pclass', 'fare']]  # eksplisit & aman
y = df['survived']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skala fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bangun model k-Nearest Neighbors
k = 5  # Jumlah tetangga terdekat
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test_scaled)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
print("\nMatriks Konfusi:")
print(confusion_matrix(y_test, y_pred))

#Mengambil 10 data randpm dari 715 data uji
data_test = X_test.sample(10, random_state=715).copy()
data_test_scaled = scaler.transform(data_test)

#Prediksi data
prediksi = model.predict(data_test_scaled)
probabilitas = model.predict_proba(data_test_scaled)

#Membuat Tabel Hasil
hasil = data_test.copy()
hasil['Status'] = ['SELAMAT' if p == 1 else 'TIDAK SELAMAT' for p in prediksi]
hasil = hasil.rename(columns={'age': 'age', 'pclass': 'pclass', 'fare': 'fare'})

#Menampilkan Tabel
print("\nHasil Prediksi")
print(hasil)

#Input data baru yang tidak ada didalam set
print("\nPrediksi Data Penumpang Baru")
data_baru = np.array([[30, 1, 100]]) 
data_baru_scaled = scaler.transform(data_baru)
prediksi_baru = model.predict(data_baru_scaled)
if prediksi_baru[0] == 1:
    print(f"Hasil Prediksi: Penumpang ini SELAMAT")
else:
    print(f"Hasil Prediksi: Penumpang ini TIDAK SELAMAT")