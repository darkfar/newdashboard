# 🚴‍♂️ Bike Sharing Analytics Dashboard


**Dashboard interaktif untuk analisis komprehensif data sistem bike sharing (2011-2012)**

[🚀 Live Demo](https://newdashboardkhallifah.streamlit.app/) | 



## 📋 Prerequisites

Pastikan sistem Anda memenuhi persyaratan berikut:

- **Python**: Version 3.8 atau lebih tinggi
- **RAM**: Minimal 4GB (disarankan 8GB)
- **Storage**: Minimal 500MB ruang kosong
- **Browser**: Chrome, Firefox, Safari, atau Edge terbaru

## 🚀 Setup dan Instalasi

### 1. Clone Repository


# Clone repository
git clone https://github.com/darkfar/newdashboard.git
cd bike-sharing-dashboard


### 2. Setup Virtual Environment

#### Menggunakan `venv` (Recommended)


# Buat virtual environment
python -m venv bike_sharing_env

# Aktivasi virtual environment
# Windows
bike_sharing_env\Scripts\activate

# macOS/Linux
source bike_sharing_env/bin/activate


#### Menggunakan Conda (Alternative)


# Buat conda environment
conda create --name bike_sharing python=3.8

# Aktivasi environment
conda activate bike_sharing


### 3. Install Dependencies


# Install semua dependencies dari requirements.txt
pip install -r requirements.txt

# Atau install secara manual
pip install streamlit pandas numpy matplotlib seaborn plotly


### 4. Siapkan Data (Opsional)


# Pastikan file data berada di direktori yang tepat
📁 project/
├── dashboard.py
├── requirements.txt
├── day.csv          # Data harian (opsional)
├── hour.csv         # Data per jam (opsional)
└── README.md


> **📝 Note:** Jika file `day.csv` dan `hour.csv` tidak tersedia, dashboard akan secara otomatis menggunakan data sampel yang realistis untuk demonstrasi.

### 5. Jalankan Dashboard


# Jalankan dashboard
streamlit run dashboard.py

# Atau dengan path lengkap
streamlit run /path/to/your/dashboard.py


### 6. Akses Dashboard

Dashboard akan otomatis terbuka di browser. Jika tidak, buka manual:


🌐 Local URL: http://localhost:8501
📱 Network URL: http://your-ip:8501




## 📊 Fitur Utama Dashboard

### 🎯 Key Performance Indicators (KPIs)
- **Total Rentals**: Jumlah keseluruhan penyewaan
- **Average Daily Usage**: Rata-rata penggunaan harian
- **Peak Day Usage**: Hari dengan penggunaan tertinggi
- **User Distribution**: Persentase casual vs registered users
- **Weather Impact**: Pengaruh cuaca terhadap penggunaan

### 📈 Analisis Komprehensif

#### 1. 📊 Core Analytics
- **Seasonal Analysis**: Analisis pola penggunaan per musim
- **Weather Impact**: Pengaruh kondisi cuaca terhadap rental
- **Temporal Patterns**: Pola penggunaan harian dan per jam
- **User Segmentation**: Perbandingan casual vs registered users

#### 2. 🔍 Advanced Analytics
- **Time Series Decomposition**: Analisis tren dan pola temporal
- **Multi-dimensional Analysis**: Bubble chart dan radar chart
- **Correlation Analysis**: Matriks korelasi faktor lingkungan
- **Seasonal Performance**: Radar chart performa musiman

#### 3. 🎯 Manual Clustering Analysis
- **Business Rule-Based Clustering**: 6 cluster berdasarkan pola penggunaan
  - **Peak Performance**: Kondisi optimal dengan penggunaan tertinggi
  - **Weather Affected**: Penggunaan rendah akibat cuaca buruk
  - **Weekend Warriors**: Penggunaan tinggi di akhir pekan
  - **Regular Commuters**: Pola komuter harian yang konsisten
  - **Hot Weather Users**: Penggunaan moderat saat cuaca panas
  - **Standard Usage**: Penggunaan normal

#### 4. 📈 RFM-Style Analysis (Adapted)
- **Recency**: Hari sejak periode penggunaan tinggi terakhir
- **Frequency**: Frekuensi hari dengan penggunaan tinggi
- **Monetary**: Total volume penggunaan (proxy untuk revenue)

### 🎨 Interactive Features
- **📅 Date Range Filter**: Pilih rentang tanggal analisis
- **🔍 Real-time Updates**: Semua visualisasi update otomatis
- **📱 Responsive Design**: Optimal di desktop dan mobile
- **🎯 Hover Details**: Informasi detail saat hover
- **📋 Exportable Data**: Data dapat di-download



## 🎨 Design Principles & Visualization

Dashboard ini menerapkan prinsip desain modern dan effective data visualization:

### 🎯 Visual Design Standards
- **Color Palette**: Konsisten menggunakan skema warna yang harmonis
- **Typography**: Hierarki informasi yang jelas dengan gradient headers
- **Spacing**: Layout yang seimbang dengan proper spacing
- **Contrast**: High contrast untuk accessibility
- **Interactive Elements**: Smooth hover effects dan animations

### 📊 Chart Design Principles
- **Data-Ink Ratio**: Maksimalkan informasi, minimalkan noise
- **Color Coding**: Konsisten untuk kategori yang sama
- **Accessibility**: Color-blind friendly palette
- **Progressive Disclosure**: Informasi tersusun dalam tab
- **Context**: Selalu menyediakan context dan legend yang jelas



## 🗂️ Struktur Project


📁 bike-sharing-dashboard/
├── 🐍 dashboard.py          # Main dashboard application
├── 📋 requirements.txt      # Python dependencies
├── 📄 README.md            # This documentation
├── 🌐 url.txt              # Deployed dashboard URL              
│── day.csv             # Daily bike sharing data
│── hour.csv            # Hourly bike sharing data


## 🔧 Technical Specifications

### 📚 Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >=1.28.0 | Web app framework |
| `pandas` | >=1.5.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical computing |
| `plotly` | >=5.15.0 | Interactive visualizations |
| `matplotlib` | >=3.6.0 | Static plots |
| `seaborn` | >=0.12.0 | Statistical visualizations |

### ⚙️ System Requirements

- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **Python Version**: 3.8 - 3.11
- **Browser**: Modern browsers with JavaScript enabled

---

## 📈 Business Insights & Recommendations

Dashboard ini menghasilkan insights bisnis yang actionable:

### 🎯 Strategic Insights
- **Seasonal Optimization**: Identifikasi periode peak dan low demand
- **Weather-Based Pricing**: Strategi dynamic pricing berdasarkan cuaca
- **Resource Allocation**: Optimasi distribusi sepeda berdasarkan pola
- **User Segmentation**: Strategi berbeda untuk casual vs registered users

### 💡 Key Findings
- Fall season menunjukkan penggunaan tertinggi
- Clear weather menghasilkan 60% lebih banyak rental
- Rush hours (8AM, 6PM) adalah periode peak
- Registered users mendominasi dengan 80%+ total usage

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. **Port Already in Use**
```bash
# Error: Port 8501 is already in use
streamlit run dashboard.py --server.port 8502
```

#### 2. **Memory Issues**
```bash
# Untuk dataset besar, tingkatkan memory limit
streamlit run dashboard.py --server.maxUploadSize 200
```

#### 3. **Module Not Found**
```bash
# Pastikan virtual environment aktif
pip install -r requirements.txt --upgrade
```

#### 4. **Data Loading Issues**
- Pastikan file `day.csv` dan `hour.csv` dalam folder yang sama
- Dashboard akan menggunakan sample data jika file tidak ditemukan
- Check file permissions dan encoding

---

## 🔄 Updates & Maintenance

### Version History
- **v2.0.0** (Current): Enhanced clustering analysis, improved design
- **v1.5.0**: Added RFM-style analysis
- **v1.0.0**: Initial release with core analytics

### Regular Updates
- Data refresh: Monthly (jika menggunakan real data)
- Library updates: Quarterly
- Security patches: As needed

---

## 🤝 Contributing

Ingin berkontribusi? Silakan:

1. Fork repository ini
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---


---

#

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - Bike Sharing Dataset
- **Framework**: Streamlit Community for excellent documentation
- **Visualization**: Plotly team for interactive charts
- **Inspiration**: Data science community and best practices

