import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB

# --- 1. DEFINISI DATASET (Berdasarkan Tabel 2 & 3) ---

# Data Gejala (Symptom)
gejala_dict = {
    'G01': 'Gigi ngilu',
    'G02': 'Gigi berdenyut',
    'G03': 'Gigi goyang',
    'G04': 'Gigi baru muncul, gigi lama masih ada',
    'G05': 'Gusi bengkak',
    'G06': 'Gigi berlubang',
    'G07': 'Pipi bengkak dan terasa hangat',
    'G08': 'Nyeri saat mengunyah',
    'G09': 'Sariawan',
    'G10': 'Sakit gigi bungsu',
    'G11': 'Gigi berlubang tanpa sakit',
    'G12': 'Gigi sakit saat diketuk',
    'G13': 'Radang',
    'G14': 'Karang gigi'
}

# Data Penyakit (Disease)
penyakit_dict = {
    'P01': 'Pulpitis Irreversible',
    'P02': 'Pulpitis Reversible',
    'P03': 'Periodontitis',
    'P04': 'Cellulitis and abscess of mouth',
    'P05': 'Periapical abscess without sinus',
    'P06': 'Carries limmited to enamel',
    'P07': 'Persis Tensi',
    'P08': 'Stomatitis',
    'P09': 'Impaksi',
    'P10': 'Acute apical periodontitis of pulpa origin',
    'P11': 'Necrosis of pulp',
    'P12': 'Gingitivis kronis'
}

# --- 2. BASIS PENGETAHUAN (TRAINING DATA) ---
# Berdasarkan Tabel 4 (Hubungan Penyakit & Gejala)
# Buat matriks 1 (Ada Gejala) dan 0 (Tidak Ada) untuk melatih Naive Bayes

# Struktur: [G01, G02, ..., G14]
X_train = []
y_train = []

# Mendefinisikan rule manual berdasarkan Tabel 4 dan validasi Tabel 5
# Format: Kode Penyakit: [List Gejala yang relevan]
knowledge_base = {
    'P01': ['G02', 'G06'], # Pulpitis Irreversible
    'P02': ['G01', 'G06'], # Pulpitis Reversible
    'P03': ['G03'], # Periodontitis
    'P04': ['G07', 'G13', 'G06'], # Cellulitis (Lihat Kasus Pasien 6 & 14)
    'P05': ['G05', 'G06'], # Periapical abscess (Lihat Kasus Pasien 9)
    'P06': ['G06', 'G11'],  # Carries limited
    'P07': ['G04'], # Persis Tensi
    'P08': ['G09'], # Stomatitis (Sariawan)
    'P09': ['G10'], # Impaksi
    'P10': ['G02', 'G08', 'G12','G06'], # Acute apical periodontitis
    'P11': ['G06', 'G11'], # Necrosis of pulp
    'P12': ['G14']  # Gingitivis kronis
}

# Konversi Knowledge Base menjadi Dataset Training
fitur_gejala = list(gejala_dict.keys())

for penyakit, gejala_terkait in knowledge_base.items():
    # Buat baris data (row) dengan nilai 0 semua
    row = [0] * len(fitur_gejala)
    # Isi nilai 1 jika gejala terkait dengan penyakit tersebut
    for g in gejala_terkait:
        if g in fitur_gejala:
            index = fitur_gejala.index(g)
            row[index] = 1

    # Tambahkan ke data training (Kita duplikasi sedikit agar model 'belajar' bobotnya)
    # Dalam implementasi nyata, ini adalah data pasien historis (50 data).
    # Di sini kita simulasi 5 sampel per penyakit agar model stabil.
    for _ in range(5):
        X_train.append(row)
        y_train.append(penyakit)

X_train = np.array(X_train)
y_train = np.array(y_train)

# --- 3. MELATIH MODEL NAIVE BAYES ---
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model Sistem Pakar Berhasil Dilatih!")

# --- 4. FUNGSI DIAGNOSA ---
def diagnosa_penyakit(input_gejala_user):
    # Konversi input user (list kode gejala) ke format binary array
    input_vector = [0] * len(fitur_gejala)
    for g in input_gejala_user:
        if g in fitur_gejala:
            idx = fitur_gejala.index(g)
            input_vector[idx] = 1

    # Prediksi
    input_array = np.array([input_vector])
    prediksi_kode = model.predict(input_array)[0]
    probabilitas = model.predict_proba(input_array)[0]

    return prediksi_kode, probabilitas

# --- 5. CONTOH KASUS (STUDI KASUS) ---
# Mengambil contoh dari Tabel 5: Pasien 6
# Gejala: G07 (Pipi bengkak) dan G13 (Radang)
kasus_gejala = ['G06','G07','G13']

print(f"--- DIAGNOSA KASUS BARU ---")
print(f"Gejala yang dirasakan: {kasus_gejala}")
for g in kasus_gejala:
    print(f"- {gejala_dict[g]}")

# Lakukan Prediksi
hasil_kode, hasil_proba = diagnosa_penyakit(kasus_gejala)
hasil_penyakit = penyakit_dict[hasil_kode]

print(f"\n[HASIL DIAGNOSA]")
print(f"Penyakit yang paling mungkin: {hasil_penyakit} ({hasil_kode})")

# --- 6. VISUALISASI HASIL (Bar Chart) ---
# Mirip dengan logika Tabel 6 di jurnal (Nilai Probabilitas Akhir)

# Siapkan data untuk plot
classes = model.classes_
nama_penyakit_sorted = [penyakit_dict[c] for c in classes]

plt.figure(figsize=(12, 6))
sns.barplot(x=hasil_proba, y=nama_penyakit_sorted, palette='viridis')

plt.title(f'Probabilitas Diagnosa Penyakit Gigi (Metode Naive Bayes)\nBerdasarkan Gejala: {", ".join(kasus_gejala)}')
plt.xlabel('Nilai Probabilitas')
plt.ylabel('Jenis Penyakit')
plt.xlim(0, 1.0) # Probabilitas 0 sampai 1
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Tampilkan nilai di ujung bar
for index, value in enumerate(hasil_proba):
    if value > 0.01: # Hanya tampilkan jika probabilitas signifikan
        plt.text(value, index, f'{value:.4f}', va='center')

plt.show()