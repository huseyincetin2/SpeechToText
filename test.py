import librosa as lr
import numpy as np
from keras.models import load_model
import sys
import soundfile as sf

# 1. Ses Dosyasını Yükleme ve İşleme
def preprocess_audio(file_path):
    data, sample_rate = lr.load(file_path, sr=16000)
    print("İlk Uzunluk:",len(data))
    expected_length = 16000

    non_silent_intervals = lr.effects.split(data, top_db=20)
    # Non-silent intervals'ü birleştir
    data = np.concatenate([data[start:end] for start, end in non_silent_intervals])

    # Eğer ses verisi beklenenden uzunsa kırp
    if len(data) > expected_length:
        data = data[:expected_length]
    # Eğer ses verisi beklenenden kısaysa sıfırlarla doldur
    elif len(data) < expected_length:

        padding = expected_length - len(data)
        data = np.pad(data, (0, padding), 'constant')

    print("Son Uzunluk: ",len(data))
    return np.expand_dims(data, axis=0)  # Tek bir örnek olarak döndür

# 2. Modeli Yükleme
model = load_model('./model/m2.keras')

# 3. Tahmin Yapma
#new_audio = preprocess_audio('./dataset/augmented_dataset/augmented_dataset/bed/3.wav')
new_audio = preprocess_audio('./testvoice/172.wav')

# İşlenmiş ses verisini dosya haline getirme
output_file_path = './output/processed_audio3.wav'

# processed_audio şekli (1, 16000) olduğundan ilk ekseni sıkıştırıyoruz
sf.write(output_file_path, new_audio.squeeze(), 16000)


predictions = model.predict(new_audio)

# 4. Sonuçları İnceleme
predicted_class_index = np.argmax(predictions)
print("Tahmin edilen sınıf indeksi:", predicted_class_index)

labels = {
    0: "bed",
    1: "bird",
    2: "cat",
    3: "dog",
    4: "down",
    5: "eight",
    6: "five",
    7: "four",
    8: "go",
    9: "happy",
    10: "house",
    11: "left",
    12: "marvel",
    13: "nine",
    14: "no",
    15: "off",
    16: "on",
    17: "one",
    18: "right",
    19: "seven",
    20: "sheila",
    21: "six",
    22: "stop",
    23: "three",
    24: "two",
    25: "up",
    26: "wow",
    27: "yes",
    28: "zero"
}


print("Tahmin edilen sınıf adı:",labels[predicted_class_index])


