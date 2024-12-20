from dataloader import LibriSpeechDataset, MNISTAudioDataset
import os
from sklearn.model_selection import train_test_split
import torch

path = os.path.dirname(os.path.abspath(__file__))

root = os.path.join(path,"..\\data\\raw\\MNIST\\data")
n = 60
folders = [os.path.join(root,str(i).zfill(2)) for i in range(1,n+1)]
print(folders)
files = []
for folder in folders:
    files += os.listdir(folder)
X = []
Y = []
for file in files:
    label = file.split("_")[0]
    human = file.split("_")[1]
    X.append(os.path.join(root,human,file))
    Y.append(label)



# Split data into training and validation sets
train_audio_files, val_audio_files, train_labels, val_labels = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
test_audio_files, val_audio_files, test_labels, val_labels = train_test_split(
    val_audio_files, val_labels, test_size=0.5, random_state=42
)
# Load datasets
train_dataset = MNISTAudioDataset(
    audio_files=train_audio_files,
    labels=train_labels,
    target_sr=16000
)

val_dataset = MNISTAudioDataset(
    audio_files=val_audio_files,
    labels=val_labels,
    target_sr=16000
)

test_dataset = MNISTAudioDataset(
    audio_files=test_audio_files,
    labels=test_labels,
    target_sr=16000
)

# saving the datasets
torch.save({
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset
}, './data/processed/mnist_audio_dataset.pth')

print("MNIST dataset saved")

# same for librispeech
librispeech_path = os.path.join(path,"..\\data\\raw\\LibriSpeech\\train-clean-100")
train_dataset = LibriSpeechDataset(
    root_dir=librispeech_path, 
    subset="train-clean-100"
)

train_files, val_files = train_test_split(train_dataset, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(val_files, test_size=0.5, random_state=42)



# saving the datasets
torch.save({
    'train_files': train_files,
    'val_files': val_files,
    'test_files': test_files
}, './data/processed/librispeech_dataset.pth')
print("LibriSpeech dataset saved")