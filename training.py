import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar embeddings
embeddings = []
for file in os.listdir('VoiceInputVec'):
    if file.endswith('.npy'):
        embeddings.append(np.load(os.path.join('VoiceInputVec', file)))

# Carregar espectrogramas
espectrogramas = []
for file in os.listdir('Processed dataset as spectrogram'):
    if file.endswith('.png'):
        img = Image.open(os.path.join('Processed dataset as spectrogram', file))
        img = img.convert('L')  # Converter para escala de cinza
        img = img.resize((128, 128))  # Redimensionar se necessário
        espectrogramas.append(np.array(img) / 255.0)  # Normalizar para [0, 1]


# Encontrando o comprimento máximo entre os embeddings
max_length = max(emb.shape[0] for emb in embeddings)

# Aplicando padding para que todos os embeddings tenham o mesmo comprimento
embeddings_padded = pad_sequences(embeddings, maxlen=max_length, padding='post', dtype='float32')

# Convertendo para um array NumPy
embeddings = np.array(embeddings_padded)
espectrogramas = np.array(espectrogramas)

# Dividir em treino e validação
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(embeddings, espectrogramas, test_size=0.2, random_state=42)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, TimeDistributed

# Definir o modelo
input_shape = X_train.shape[1:]  # Formato do embedding
output_shape = y_train.shape[1:]  # Formato do espectrograma

input_layer = Input(shape=input_shape)
x = LSTM(256, return_sequences=True)(input_layer)
x = LSTM(256)(x)
x = Dense(np.prod(output_shape), activation='sigmoid')(x)  # Sigmoid para normalizar entre [0, 1]
output_layer = Reshape(output_shape)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')  # Usar MSE como loss para regressão

model.summary()

# Treinando o modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Salvando o modelo
model.save('voice2piano.keras')

# Plotar a perda de treinamento e validação
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()