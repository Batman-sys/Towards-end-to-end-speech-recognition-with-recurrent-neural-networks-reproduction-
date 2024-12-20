import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class Testing:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def test(self, test_loader, vocab):
        self.model.eval()
        test_loss = 0.0
        accurate = 0
        total_samples = len(test_loader.dataset)

        # Initialize counters for accuracy per class
        correct_per_class = {k: 0 for k in vocab.keys() if k != 'blank'}
        total_per_class = {k: 0 for k in vocab.keys() if k != 'blank'}

        with torch.no_grad():
            for padded_waveform, targets in tqdm(test_loader):
                # Transfer data to device
                padded_waveform = padded_waveform.to(self.device)
                targets = targets.to(self.device)

                # Prepare features
                features = padded_waveform.permute(0, 2, 1).to(self.device)  # (batch, time, features)

                # Pass features through the model
                output = self.model(features)  # Output: (batch, time, classes)
                output = output.log_softmax(2)  # Log-softmax for CTC
                output = output.permute(1, 0, 2)  # Required format for CTC: (time, batch, classes)

                # Compute input lengths
                input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long).to(self.device)
                target_lengths = torch.tensor([1 for t in targets]).to(self.device)

                # Compute loss
                loss = self.criterion(output, targets, input_lengths, target_lengths)
                test_loss += loss.item()

                # Compute accuracy
                output = output.permute(1, 0, 2)
                output = torch.argmax(output, dim=2)

                for i, target in enumerate(targets):
                    predicted = output[i][-1].item()
                    true_label = target.item()

                    for key, value in vocab.items():
                        if value == true_label:
                            total_per_class[key] += 1
                            if predicted == true_label:
                                accurate += 1
                                correct_per_class[key] += 1

        # Average loss for the epoch
        test_loss /= len(test_loader)
        accuracy = (accurate / total_samples) * 100
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

        self.plot_heatmap(correct_per_class, total_per_class)

    def plot_heatmap(self, correct_per_class, total_per_class):
        class_labels = list(correct_per_class.keys())
        accuracy_per_class = {
            label: (correct_per_class[label] / total_per_class[label] * 100) if total_per_class[label] > 0 else 0
            for label in class_labels
        }
        plt.figure(figsize=(12, 6))

        sns.set(font_scale=1.2)
        sns.heatmap(
            [[accuracy_per_class[label] for label in class_labels]],
            annot=True,
            fmt=".2f",
            xticklabels=class_labels,
            yticklabels=['Accuracy (%)'],
            cmap="Blues"
        )
        plt.title("Per-Class Accuracy Heatmap")
        plt.show()

    def predict(self, mel_spec, vocab):
        """
        Predicts the transcription for a single Mel spectrogram.

        Args:
            mel_spec (torch.Tensor): Mel spectrogram of shape (features, time).
            vocab (dict): Vocabulary mapping characters to indices.

        Returns:
            str: Predicted text transcription.
        """
        self.model.eval()

        features = mel_spec.unsqueeze(0).permute(0, 2, 1).to(self.device)  

        with torch.no_grad():
            output = self.model(features)  
            output = output.log_softmax(2)  
            output = output.permute(1, 0, 2)  

            # Decode predictions
            output = torch.argmax(output, dim=2).squeeze(1)
            predicted_text = []

            for t in range(output.size(0)):
                predicted_index = output[t].item()
                for key, value in vocab.items():
                    if value == predicted_index:
                        predicted_text.append(key)
                        break

        # Remove consecutive duplicates and blanks (if any)
        filtered_text = []
        prev_char = None
        blank_token = vocab.get('blank', None)

        for char in predicted_text:
            if char != prev_char and char != 'blank':
                filtered_text.append(char)
            prev_char = char

        return ''.join(filtered_text)


