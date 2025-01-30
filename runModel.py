from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Carregar o modelo salvo
modelo_carregado = load_model('voice2piano.keras')

# Solicitar o diretório do arquivo de áudio (embedding .npy)
caminho_audio = input("Digite o caminho completo do arquivo de áudio (.wav): ")

# Carregar modelo Wav2Vec2
modelo = "facebook/wav2vec2-base-960h"  # Ou outro modelo que preferir
processor = Wav2Vec2Processor.from_pretrained(modelo)
model = Wav2Vec2Model.from_pretrained(modelo)
model.eval()

# Pegar embedding do arquivo de áudio
# Carregar áudio (16kHz é a taxa esperada pelo modelo)
audio, sr = librosa.load(caminho_audio, sr=16000)

# Processar entrada
input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

# Extrair embeddings
with torch.no_grad():
    embeddings = model(input_values).last_hidden_state.squeeze(0).numpy() 

# Aplicar padding, se necessário
embeddings = pad_sequences([embeddings], maxlen=699, padding='post', dtype='float32')

# Remover a dimensão extra (se houver)
if embeddings.ndim == 3:
    embeddings = np.squeeze(embeddings, axis=0)  # Remove a dimensão extra

# Adicionar dimensão de batch (o modelo espera um batch de entradas)
embeddings = np.expand_dims(embeddings, axis=0)

# Verificar a forma do tensor de entrada
print("Forma do tensor de entrada:", embeddings.shape)  # Deve ser (1, 699, 768)

# Gerar o espectrograma
espectrograma_gerado = modelo_carregado.predict(embeddings)

# Remover a dimensão de batch (se necessário)
espectrograma_gerado = espectrograma_gerado[0]

# Visualizar o espectrograma puro, sem números, eixos ou títulos
plt.figure(figsize=(10, 4))  # Ajuste o tamanho da figura, se necessário
plt.imshow(espectrograma_gerado, cmap='gray', aspect='auto')
plt.axis('off')  # Desativa os eixos e rótulos
plt.tight_layout()  # Remove espaços extras ao redor da imagem
plt.show()

# Salvar o espectrograma como uma imagem
espectrograma_img = Image.fromarray((espectrograma_gerado * 255).astype(np.uint8))

# Remover bordas brancas (se houver)
espectrograma_img = espectrograma_img.crop(espectrograma_img.getbbox())  # Corta as bordas brancas

# Salvar a imagem
espectrograma_img.save()
print(f"Espectrograma salvo com sucesso!")