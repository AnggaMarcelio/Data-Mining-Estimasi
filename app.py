import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
svr_model = joblib.load('svr_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_Y = joblib.load('scaler_Y.pkl')

st.title("Prediksi Harga Rumah - SVR Model")

st.markdown("""
Masukkan fitur-fitur rumah di bawah ini untuk memprediksi harga rumah menggunakan model Support Vector Regression (SVR).
""")

# Input fitur (14 fitur sesuai model)
bedrooms = st.slider("Jumlah Kamar Tidur", 0, 10, 3)
bathrooms = st.slider("Jumlah Kamar Mandi", 0, 10, 2)
sqft_living = st.number_input("Luas Bangunan (sqft)", 200, 10000, 1800)
sqft_lot = st.number_input("Luas Tanah (sqft)", 500, 100000, 5000)
floors = st.slider("Jumlah Lantai", 1, 3, 1)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View (0-4)", 0, 4, 0)
condition = st.slider("Kondisi (1-5)", 1, 5, 3)
grade = st.slider("Grade (1-13)", 1, 13, 7)
sqft_above = st.number_input("Luas Di Atas Tanah (sqft)", 200, 10000, 1200)
sqft_basement = st.number_input("Luas Basement (sqft)", 0, 5000, 600)
yr_built = st.number_input("Tahun Dibangun", 1900, 2025, 1995)
yr_renovated = st.number_input("Tahun Renovasi (0 jika belum pernah)", 0, 2025, 0)
# Note: kolom 'lat', 'long', 'sqft_living15', 'sqft_lot15' tidak digunakan dalam model

# Prediksi saat tombol diklik
if st.button("Prediksi Harga"):
    input_features = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, grade, sqft_above,
        sqft_basement, yr_built, yr_renovated
    ]
    input_array = np.array([input_features])
    input_scaled = scaler_X.transform(input_array)
    pred_scaled = svr_model.predict(input_scaled)
    pred_price = scaler_Y.inverse_transform(pred_scaled.reshape(-1, 1))

    st.success(f"Harga rumah diprediksi: ${pred_price[0][0]:,.2f}")
