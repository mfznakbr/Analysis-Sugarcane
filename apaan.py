# import requests

# base_url = "http://localhost:5005"
# endpoints = [
#     "/",
#     "/predict",
#     "/health",
#     "/docs", 
#     "/swagger",
#     "/invocations",
#     "/ping",
#     "/v1/models",
#     "/model",
#     "/api/predict"
# ]

# print("Testing available endpoints:")
# for endpoint in endpoints:
#     try:
#         response = requests.get(f"{base_url}{endpoint}", timeout=5)
#         print(f"{endpoint}: {response.status_code}")
#         if response.status_code == 200:
#             print(f"Response: {response.text[:200]}...")
#     except Exception as e:
#         print(f"{endpoint}: Error - {e}")
#     print("---")

import joblib
import pandas as pd
import numpy as np
from evaluasi import processing

# Load model
def load_model(model_path='model_knn.joblib'):
    try:
        model = joblib.load(model_path)
        print("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return None

# Prediksi langsung
def predict_local(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"❌ Error prediksi: {e}")
        return None

# Test dengan sample data
def main():
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Cek info model
    print(f"Model type: {type(model)}")
    print(f"Model classes: {getattr(model, 'classes_', 'Not available')}")
    
    # Sample data untuk testing (sesuaikan dengan format yang diharapkan model)
    # Ganti dengan format yang sesuai dengan model Anda
    df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
    sample_data = df.sample(n=10, random_state=42)
    sample_data = processing(sample_data)  # Preprocessing sesuai pipeline
    
    print(f"\n📊 Data test shape: {sample_data.shape}")
    print("Data test:")
    print(sample_data)
    
    # Lakukan prediksi
    predictions = predict_local(model, sample_data)
    
    if predictions is not None:
        print(f"\n🎯 Hasil prediksi: {predictions}")
        
        # Interpretasi hasil
        for i, pred in enumerate(predictions):
            personality = "Introvert" if pred == 1 else "Ekstrovert"
            print(f"Data {i+1}: {personality} (nilai: {pred})")
    
    # Test predict probability (jika model support)
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(sample_data)
            print(f"\n📈 Probabilities: {probabilities}")
        except Exception as e:
            print(f"Predict_proba not supported: {e}")

if __name__ == "__main__":
    main()