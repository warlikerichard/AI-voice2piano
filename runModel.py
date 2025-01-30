from tensorflow.keras.models import load_model
import numpy as np

# Carregar o modelo salvo
modelo_carregado = load_model('voice2piano.keras')

# Carregar um novo embedding (substitua pelo caminho do seu arquivo .npy)
novo_embedding = np.load('VoiceInputVec/output1.npy')

# Aplicar padding, se necessário
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

# Visualizar o espectrograma
import matplotlib.pyplot as plt
plt.imshow(espectrograma_gerado, cmap='gray', aspect='auto')
plt.title('Espectrograma Gerado')
plt.show()

# Salvar o espectrograma como uma imagem
from PIL import Image
espectrograma_img = Image.fromarray((espectrograma_gerado * 255).astype(np.uint8))
espectrograma_img.save('espectrograma_gerado.png')