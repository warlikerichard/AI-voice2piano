import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import subprocess

# Caminhos
pasta_midi = "Processed dataset"
pasta_audio = "Processed dataset as wav"
pasta_espectrograma = "Processed dataset as spectrogram"
soundfont = "Soundfont/FluidR3_GM.sf2"

# Criar pastas se não existirem
os.makedirs(pasta_audio, exist_ok=True)
os.makedirs(pasta_espectrograma, exist_ok=True)

def midi_para_wav(arquivo_midi, arquivo_wav):
    """Converte um arquivo MIDI para WAV usando FluidSynth."""
    comando = f'fluidsynth -ni "{soundfont}" "{arquivo_midi}" -F "{arquivo_wav}" -r 44100'
    subprocess.run(comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def gerar_espectrograma(arquivo_wav, arquivo_saida):
    """Gera e salva um espectrograma puro (sem eixos ou títulos) de um arquivo WAV."""
    audio, sr = librosa.load(arquivo_wav, sr=44100)
    espectrograma = np.abs(librosa.stft(audio))

    plt.figure(figsize=(10, 4))
    plt.axis("off")  # Remove os eixos
    plt.xticks([])   # Remove os ticks do eixo X
    plt.yticks([])   # Remove os ticks do eixo Y
    plt.box(False)   # Remove a borda do gráfico

    # Gera e salva o espectrograma puro
    librosa.display.specshow(librosa.amplitude_to_db(espectrograma, ref=np.max), sr=sr, y_axis=None, x_axis=None)
    plt.savefig(arquivo_saida, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()


# Converter todos os arquivos MIDI
for arquivo in os.listdir(pasta_midi):
    if arquivo.endswith(".midi") or arquivo.endswith(".mid"):
        caminho_midi = os.path.join(pasta_midi, arquivo)
        caminho_wav = os.path.join(pasta_audio, arquivo.replace(".midi", ".wav"))
        caminho_spectrograma = os.path.join(pasta_espectrograma, arquivo.replace(".midi", ".png"))

        # Converter MIDI para WAV
        midi_para_wav(caminho_midi, caminho_wav)

        # Gerar espectrograma
        gerar_espectrograma(caminho_wav, caminho_spectrograma)

        print(f"Convertido: {arquivo} -> {caminho_spectrograma}")

print("Processo concluído!")
