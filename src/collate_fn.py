import torch


def collate_fn_Libri(batch, vocab):
    mel_features, transcriptions = zip(*batch)

    # Find max time length in the batch
    max_length = max(feature.shape[1] for feature in mel_features)
    # Pad Mel Spectrogram features
    padded_features = []
    for feature in mel_features:
        pad_amount = max_length - feature.shape[1]
        padded_feature = torch.nn.functional.pad(feature, (0, pad_amount), "constant", 0)
        padded_features.append(padded_feature)

    # Convertir les transcriptions en indices
    targets = [torch.tensor([vocab[char] for char in t.lower()]) for t in transcriptions]
    target_lengths = torch.tensor([len(t) for t in targets])
    
    padded_features = torch.stack(padded_features)
    return padded_features, torch.cat(targets), target_lengths


def collate_fn_MNIST(batch):
    mel_features, labels = zip(*batch)

    # Find max time length in the batch
    max_length = max(feature.shape[1] for feature in mel_features)

    # Pad Mel Spectrogram features
    padded_features = []
    for feature in mel_features:
        pad_amount = max_length - feature.shape[1]
        padded_feature = torch.nn.functional.pad(feature, (0, pad_amount), "constant", 0)
        padded_features.append(padded_feature)

    padded_features = torch.stack(padded_features) 
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_features, labels

