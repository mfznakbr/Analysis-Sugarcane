from joblib import load
import pandas as pd 
import json
import requests

df = pd.read_csv('https://raw.githubusercontent.com/mfznakbr/Eksperimen_SML_Muhammad-Fauzani-Akbar/main/personality_datasert.csv')
data = df.sample(n=10, random_state=42)
print(data)

def processing(data):
    load_path = r"preprocess_pipeline.joblib"
    prepro = load(load_path)
    # print(f"pipeline preprocessing dimuat dari : {load_path}")

    transformed_data = prepro.transform(data)
    # print("Data setelah preprocessing:")
    # print(transformed_data[:5])  # Print hanya 5 baris pertama
    return transformed_data

def prediction(preprocessed_data):
    """
    Mengirim data ke REST API dan mengambil hasil prediksi.
    """
    url = "http://localhost:5005/invocations"
    headers = {"Content-Type": "application/json"}

    # Gunakan nama kolom yang sesuai dengan schema model
    required_columns = [
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
        'Friends_circle_size', 'Post_frequency', 'Stage_fear_No',
        'Stage_fear_Yes', 'Drained_after_socializing_No', 'Drained_after_socializing_Yes'
    ]
    
    # Convert numpy array ke DataFrame dengan nama kolom yang benar
    preprocessed_df = pd.DataFrame(
        preprocessed_data, 
        columns=required_columns
    )
    
    # Format payload yang benar untuk MLflow
    payload = json.dumps({
        "dataframe_records": preprocessed_df.to_dict(orient='records')
    })
    
    # print("Payload yang dikirim:")
    # print(payload[:500] + "..." if len(payload) > 500 else payload)
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"[ERROR] Server response: {response.text}")

def main():
    try:
        preprocessed_data = processing(data)
        predictions = prediction(preprocessed_data)
        if predictions == 1:
            print("Prediksi: Introvert")
        else:
            print("Prediksi: Ekstrovert")
        print("Hasil Prediksi:")
        print(json.dumps(predictions, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()