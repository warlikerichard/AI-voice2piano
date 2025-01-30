import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Caminhos das pastas
pasta_entrada = "VoiceInput"
pasta_saida = "VoiceInputVec"

# Criar pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)

# Carregar modelo Wav2Vec2
modelo = "facebook/wav2vec2-base-960h"  # Ou outro modelo que preferir
processor = Wav2Vec2Processor.from_pretrained(modelo)
model = Wav2Vec2Model.from_pretrained(modelo)
model.eval()

# Processar cada arquivo na pasta de entrada
for arquivo in os.listdir(pasta_entrada):
    if arquivo.endswith(".wav"):
        caminho_audio = os.path.join(pasta_entrada, arquivo)

        # Carregar áudio (16kHz é a taxa esperada pelo modelo)
        audio, sr = librosa.load(caminho_audio, sr=16000)

        # Processar entrada
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

        # Extrair embeddings
        with torch.no_grad():
            embeddings = model(input_values).last_hidden_state.squeeze(0).numpy()  # (Tamanho, 768)

        # Salvar os vetores
        caminho_saida = os.path.join(pasta_saida, arquivo.replace(".wav", ".npy"))
        np.save(caminho_saida, embeddings)

        print(f"Convertido: {arquivo} -> {caminho_saida}")

print("Conversão concluída!")
