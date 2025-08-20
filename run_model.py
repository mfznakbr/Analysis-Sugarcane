import joblib
import pandas as pd
from joblib import load
import pandas as pd 

df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
data = df.sample(n=1, random_state=42)
print(data)

def processing(data):
    load_path = r"preprocess_pipeline.joblib"
    prepro = load(load_path)
    # print(f"pipeline preprocessing dimuat dari : {load_path}")

    transformed_data = prepro.transform(data)
    # print("Data setelah preprocessing:")
    # print(transformed_data[:5])  # Print hanya 5 baris pertama
    return transformed_data

def load_model(model_path='model_knn.joblib'):
    try:
        model = joblib.load(model_path)
        # print("✅ Model berhasil dimuat!")
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

def main(data):
    # load model
    model = load_model()
    sample_data = processing(data)  # Preprocessing sesuai pipeline
    if model is None:
        return
    prediksi = predict_local(model, sample_data)
    return prediksi

if __name__ == "__main__":
    result = main(data)
    print(result)
    if result == 1:
        print("Prediksi: Introvert")
    else:
        print("Prediksi: Ekstrovert")