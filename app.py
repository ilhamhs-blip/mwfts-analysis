import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Aplikasi Prediksi Sederhana",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Aplikasi Prediksi Sederhana")

st.write("""
Upload file CSV yang memiliki kolom **Tanggal** dan **Harga**.
Aplikasi akan menampilkan grafik serta prediksi sederhana menggunakan Moving Average.
""")

uploaded_file = st.file_uploader(
    "Upload file CSV", type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Data")

    st.dataframe(df.head())

    if "Harga" in df.columns:

        window = st.slider(
            "Pilih Window Moving Average",
            min_value=2,
            max_value=30,
            value=7
        )

        df["Moving Average"] = (
            df["Harga"]
            .rolling(window=window)
            .mean()
        )

        st.subheader("Grafik")

        fig, ax = plt.subplots(figsize=(10,4))

        ax.plot(df["Harga"], label="Harga")
        ax.plot(df["Moving Average"], label=f"MA {window}")

        ax.set_xlabel("Index")
        ax.set_ylabel("Harga")
        ax.legend()

        st.pyplot(fig)

        prediksi = df["Moving Average"].iloc[-1]

        st.metric(
            label="Prediksi Harga Berikutnya",
            value=f"{prediksi:,.2f}"
        )

    else:
        st.error("Kolom 'Harga' tidak ditemukan.")
