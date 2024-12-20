import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from torchaudio.datasets import LIBRISPEECH


class MNISTAudioDataset(Dataset):
    def __init__(self, audio_files, labels, target_sr=16000):
        """
        Args:
            audio_files (list): List of paths to audio files.
            labels (list): List of corresponding labels.
            target_sr (int): Target sampling rate for audio files.
        """
        self.audio_files = audio_files
        self.labels = [int(label) for label in labels]
        self.target_sr = target_sr

        # Transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=500,
            hop_length=160,
            n_mels=128
        )

        # Ensure data integrity
        assert len(self.audio_files) == len(self.labels), "Mismatch between audio and label lengths"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            # Load audio file
            waveform, sr = torchaudio.load(self.audio_files[idx])

            # Resample if needed
            if sr != self.target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Ensure waveform is mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = self._fix_length(waveform, length=12000)

            
            features = self.mel_transform(waveform).squeeze(0)

            # Normalize features
            features = (features - features.mean()) / (features.std() + 1e-6)

            # Return features and label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return features, label

        except Exception as e:
            print(f"Error processing file {self.audio_files[idx]}: {e}")
            return None, None

    @staticmethod
    def _fix_length(waveform, length):
        """Pads or truncates the waveform to the specified length."""
        if waveform.size(1) < length:
            # Pad with zeros
            padding = length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Truncate
            waveform = waveform[:, :length]
        return waveform



class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, subset="train-clean-100", target_sr=16000, max_length=16000):
        """
        Args:
            root_dir (str): Path to the LibriSpeech dataset.
            subset (str): Subset of LibriSpeech to use (e.g., "train-clean-100").
            target_sr (int): Target sampling rate for audio files.
            max_length (int): Max length of waveform in samples for padding/truncation.
        """
        self.dataset = LIBRISPEECH(root=root_dir, url=subset, download=False)
        self.target_sr = target_sr
        self.max_length = max_length

        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=400,
            hop_length=160,
            n_mels=128
        )

    def __len__(self):
        return len(self.dataset)
    
    def get_waveform_and_label(self, idx):
        waveform, sr, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        return waveform, transcript

    def __getitem__(self, idx):
        try:
            # Load audio data
            waveform, sr, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]

            # Resample if needed
            if sr != self.target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Ensure waveform is mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)


            # Extract features
            features = self.mel_transform(waveform).squeeze(0)

            # Normalize features
            features = (features - features.mean()) / (features.std() + 1e-6)

            # Return features and metadata
            return features, transcript

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return None, None

    @staticmethod
    def _fix_length(waveform, length):
        """Pads or truncates the waveform to the specified length."""
        if waveform.size(1) < length:
            padding = length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :length]
        return waveform
