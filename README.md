# 🚴‍♂️ Bike Sharing Dashboard

Dashboard interaktif untuk analisis data sistem bike sharing menggunakan Streamlit.

## 📋 Prerequisites

Pastikan Anda memiliki Python 3.8+ terinstall di sistem Anda.

## 🚀 Cara Menjalankan Dashboard

### 1. Install Dependencies


pip install -r requirements.txt


### 2. Siapkan Data

Pastikan file `day.csv` dan `hour.csv` berada dalam direktori yang sama dengan file dashboard. Jika file tidak tersedia, dashboard akan menggunakan data sample untuk demonstrasi.

### 3. Jalankan Dashboard


streamlit run dashboard.py


### 4. Akses Dashboard

Dashboard akan terbuka secara otomatis di browser atau akses manual melalui:
- URL: `http://localhost:8501`

## 📊 Fitur Dashboard

### Filter Interaktif
- **Rentang Tanggal**: Pilih periode analisis
- **Musim**: Filter berdasarkan Spring, Summer, Fall, Winter
- **Kondisi Cuaca**: Filter berdasarkan Clear, Misty, Light Rain/Snow, Heavy Rain/Snow

### Visualisasi Utama

#### 1. Ringkasan Metrics
- Total penyewaan dalam periode terpilih
- Rata-rata penyewaan harian
- Penyewaan tertinggi
- Persentase pengguna casual

#### 2. Analisis Musim dan Cuaca
- Bar chart rata-rata penyewaan per musim
- Bar chart rata-rata penyewaan per kondisi cuaca
- Heatmap korelasi faktor lingkungan

#### 3. Analisis Pola Temporal
- Line chart pola penggunaan mingguan
- Perbandingan weekday vs weekend
- Pola penggunaan per jam
- Heatmap penggunaan hari vs jam

### 💡 Key Insights
Dashboard menampilkan insight utama dari analisis data, termasuk:
- Pengaruh musim dan cuaca terhadap penyewaan
- Pola temporal penggunaan harian dan per jam
- Rekomendasi bisnis berdasarkan temuan

## 🔧 Struktur File

```
📁 project/
├── dashboard.py          # File utama dashboard
├── requirements.txt      # Dependencies Python
├── day.csv              # Data harian (optional)
├── hour.csv             # Data per jam (optional)
└── README.md            # Instruksi ini
```


## 📝 Notes

- Dashboard akan otomatis memuat data dari `day.csv` dan `hour.csv`
- Jika file data tidak ditemukan, akan menggunakan data sample
- Filter akan mempengaruhi semua visualisasi secara real-time


## 🤝 Support

Jika mengalami masalah, pastikan:
1. Python version 3.8+
2. Semua dependencies terinstall dengan benar
3. File data dalam format yang sesuai
4. Port 8501 tidak digunakan aplikasi lain