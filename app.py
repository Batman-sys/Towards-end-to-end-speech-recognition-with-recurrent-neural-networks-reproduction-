from tkinter import *
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time
import sys
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
sys.path.append('./src')


from model import SpeechRecognitionModel
from test_model import Testing

# Load the model
vocab = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'blank': 10
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = SpeechRecognitionModel(input_dim=128, hidden_dim=256, output_dim=len(vocab) + 1, num_layers=5).to(device)
best_model.load_state_dict(torch.load('.\\outputs\\best_MNIST_model.pth', map_location=device))
tester = Testing(model=best_model, criterion=nn.CTCLoss(blank=10).to(device), device=device)

root = Tk()
root.title("Digit Audio Processing")
root.geometry("400x400")

# status label
status_label = Label(root, text="Welcome! Please Record, Play, and Predict Audio, \n Be quick, it's only 1 second, \n Also, Maybe scream into the microhphone", fg="blue")
status_label.pack()

# record button
def record():
    status_label.config(text="Recording Audio...")
    freq = 44100
    duration = 1
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype='int16')
    sd.wait()
    write("./recordings/recorded_audio.wav", freq, recording)
    status_label.config(text="Audio Recorded Successfully.")

record_button = Button(root, text="Record", command=record)
record_button.pack()

# play button
def play():
    status_label.config(text="Playing Audio...")
    data = wv.read("./recordings/recorded_audio.wav")
    sd.play(data.data, data.rate)
    time.sleep(5)
    status_label.config(text="Audio Playback Finished.")

play_button = Button(root, text="Play", command=play)
play_button.pack()

# Transform function for processing audio
def transform(audio_path, target_sr=16000, length=12000):
    waveform, sr = torchaudio.load(audio_path, normalize=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Ensure waveform is mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = _fix_length(waveform, length)

    mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_fft=500, hop_length=160, n_mels=128)
    mel_spec = mel_transform(waveform).squeeze(0)

    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

    return mel_spec

def _fix_length(waveform, length):
    """Pads or truncates the waveform to the specified length."""
    if waveform.size(1) < length:
        padding = length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :length]
    return waveform

# predict button
def predict():
    status_label.config(text="Predicting Audio...")
    prediction = tester.predict(transform('./recordings/recorded_audio.wav'), vocab)
    status_label.config(text=f"Predicted Transcription: {prediction}")

predict_button = Button(root, text="Predict", command=predict)
predict_button.pack()

root.mainloop()
