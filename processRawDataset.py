import os
import pretty_midi
import random

# Caminho da pasta com os arquivos MIDI
input_folder = "Raw dataset/2004"
output_folder = "New Processed dataset"

# Certifique-se de que a pasta de saída existe
os.makedirs(output_folder, exist_ok=True)

# Duração mínima e máxima em segundos
min_duration = 3
max_duration = 7

# Função para cortar o arquivo MIDI para uma duração específica
def trim_midi(midi, duration):
    # Cortar notas
    for instrument in midi.instruments:
        instrument.notes = [
            note for note in instrument.notes if note.start < duration
        ]
        # Ajustar notas que ultrapassam o limite
        for note in instrument.notes:
            if note.end > duration:
                note.end = duration

    # Cortar eventos de controle
    for instrument in midi.instruments:
        instrument.control_changes = [
            cc for cc in instrument.control_changes if cc.time < duration
        ]
        instrument.pitch_bends = [
            pb for pb in instrument.pitch_bends if pb.time < duration
        ]

    # Cortar mudanças de tempo
    midi.time_signature_changes = [
        ts for ts in midi.time_signature_changes if ts.time < duration
    ]
    midi.key_signature_changes = [
        ks for ks in midi.key_signature_changes if ks.time < duration
    ]

    # Ajustar tempos dos eventos
    original_times = [0, duration]
    new_times = [0, duration]
    midi.adjust_times(original_times, new_times)

    return midi

# Processar arquivos MIDI
def process_midi_files(input_folder, output_folder, min_duration, max_duration):
    midi_files = [f for f in os.listdir(input_folder) if f.endswith(".midi")]
    count = 1  # Contador para renomear arquivos

    for midi_file in midi_files:
        input_path = os.path.join(input_folder, midi_file)

        if count > 7:
            try:
                # Carregar o arquivo MIDI
                midi = pretty_midi.PrettyMIDI(input_path)
                end_time = midi.get_end_time()

                # Ajustar duração do MIDI
                if end_time > max_duration:
                    midi = trim_midi(midi, random.randint(min_duration, max_duration))
                elif end_time < min_duration:
                    print(f"Arquivo {midi_file} ignorado (duração menor que {min_duration}s)")
                    continue

                # Salvar o arquivo MIDI processado com nome padronizado
                output_path = os.path.join(output_folder, f"output{count+93}.midi")
                midi.write(output_path)
                print(f"Arquivo {midi_file} processado e salvo como output{count+93}.midi")
                count += 1

            except Exception as e:
                print(f"Erro ao processar {midi_file}: {e}")
        else:        
            count += 1

# Chamar a função para processar os arquivos
process_midi_files(input_folder, output_folder, min_duration, max_duration)
