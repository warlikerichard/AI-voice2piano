from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Carregar o modelo salvo
modelo_carregado = load_model('voice2piano.keras')

# Carregar um novo embedding (substitua pelo caminho do seu arquivo .npy)
novo_embedding = np.load('VoiceInputVec/output1.npy')

# Aplicar padding, se necessário
novo_embedding = pad_sequences([novo_embedding], maxlen=699, padding='post', dtype='float32')

# Remover a dimensão extra (se houver)
if novo_embedding.ndim == 3:
    novo_embedding = np.squeeze(novo_embedding, axis=0)  # Remove a dimensão extra

# Adicionar dimensão de batch (o modelo espera um batch de entradas)
novo_embedding = np.expand_dims(novo_embedding, axis=0)

# Verificar a forma do tensor de entrada
print("Forma do tensor de entrada:", novo_embedding.shape)  # Deve ser (1, 699, 768)

# Gerar o espectrograma
espectrograma_gerado = modelo_carregado.predict(novo_embedding)

# Remover a dimensão de batch (se necessário)
espectrograma_gerado = espectrograma_gerado[0]

# Visualizar o espectrograma puro, sem números, eixos ou títulos
plt.figure(figsize=(10, 4))  # Ajuste o tamanho da figura, se necessário
plt.imshow(espectrograma_gerado, cmap='gray', aspect='auto')
plt.axis('off')  # Desativa os eixos e rótulos
plt.tight_layout()  # Remove espaços extras ao redor da imagem
plt.show()

# Salvar o espectrograma como uma imagem
from PIL import Image

# Converter o espectrograma para uma imagem em escala de cinza
espectrograma_img = Image.fromarray((espectrograma_gerado * 255).astype(np.uint8))

# Remover bordas brancas (se houver)
espectrograma_img = espectrograma_img.crop(espectrograma_img.getbbox())  # Corta as bordas brancas

# Salvar a imagem
espectrograma_img.save('espectrograma_gerado.png')