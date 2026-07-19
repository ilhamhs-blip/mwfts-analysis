import streamlit as st

# ======================================================
# Konfigurasi Halaman
# ======================================================

st.set_page_config(
    page_title="MWFTS Financial Forecast",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# Header
# ======================================================

st.title("📈 MWFTS Financial Forecast Dashboard")
st.caption("Forecasting Financial Time Series menggunakan Markov Weighted Fuzzy Time Series")

# ======================================================
# Sidebar
# ======================================================

st.sidebar.header("⚙️ Pengaturan Analisis")

aset = st.sidebar.selectbox(
    "Aset",
    [
        "IHSG",
        "Ethereum",
        "Bitcoin"
    ]
)

periode = st.sidebar.selectbox(
    "Periode Data",
    [
        "1 Bulan",
        "3 Bulan",
        "6 Bulan",
        "1 Tahun",
        "3 Tahun",
        "5 Tahun"
    ]
)

interval = st.sidebar.selectbox(
    "Interval",
    [
        "Harian",
        "4 Harian",
        "Mingguan"
    ]
)

model = st.sidebar.selectbox(
    "Model",
    [
        "MWFTS",
        "MWFTS + PSO",
        "MWFTS + CMBO"
    ]
)

prediksi = st.sidebar.selectbox(
    "Horizon Forecast",
    [
        "H+1",
        "H+7",
        "H+30"
    ]
)

confidence = st.sidebar.select_slider(
    "Taraf Kepercayaan",
    options=[
        "90%",
        "95%",
        "99%"
    ]
)

run = st.sidebar.button("🚀 Jalankan Analisis")

# ======================================================
# MAIN PAGE
# ======================================================

if not run:

    st.info("Silakan pilih parameter analisis di sidebar kemudian tekan **Jalankan Analisis**.")

else:

    # ======================================================
    # Grafik Forecast
    # ======================================================

    st.subheader("📈 Forecast")

    chart_placeholder = st.empty()

    st.caption(
        """
        Grafik akan menampilkan:

        🔵 Biru = Data historis

        🟡 Kuning = Forecast

        🟢 Area transparan = Confidence Interval
        """
    )

    chart_placeholder.info(
        "Tempat grafik MWFTS nantinya."
    )

    # ======================================================
    # Metric
    # ======================================================

    st.divider()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "MAE",
        "-"
    )

    col2.metric(
        "RMSE",
        "-"
    )

    col3.metric(
        "MAPE",
        "-"
    )

    col4.metric(
        "Forecast",
        prediksi
    )

    # ======================================================
    # EDA
    # ======================================================

    st.divider()

    st.header("📊 Exploratory Data Analysis")

    left,right = st.columns([1,2])

    with left:

        st.subheader("Statistik")

        st.table({
            "Statistik":[
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Std Dev",
                "Variance",
                "Skewness",
                "Kurtosis"
            ],
            "Nilai":[
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-"
            ]
        })

    with right:

        st.subheader("Distribusi")

        st.info("Histogram")

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Moving Average")

        st.info("Grafik MA")

    with col2:

        st.subheader("Exponential Moving Average")

        st.info("Grafik EMA")

    # ======================================================
    # AI Analysis
    # ======================================================

    st.divider()

    st.header("🤖 AI Financial Analysis")

    st.markdown("""
### 1. Ringkasan Data

-

### 2. Analisis Tren

-

### 3. Analisis Volatilitas

-

### 4. Evaluasi Model

-

### 5. Interpretasi Forecast

-

### 6. Tingkat Risiko

-

### 7. Rekomendasi

-

### 8. Kesimpulan

-
""")
