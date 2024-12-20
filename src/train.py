import torch
from tqdm import tqdm
import os

class Training:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.val_losses = []
        self.sys_path = os.path.dirname(os.path.abspath(__file__))

    def train(self, train_loader, val_loader, num_epochs, dataset_name):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            for mel_spec, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                mel_spec, targets = mel_spec.to(self.device), targets.to(self.device)
                target_lengths = torch.tensor([1 for t in targets]).to(self.device)
                features = mel_spec.permute(0, 2, 1).to(self.device)  # (batch, time, features)

                # Forward pass
                output = self.model(features)
                output = output.log_softmax(2)  # Log-softmax for CTC
                output = output.permute(1, 0, 2)  # (T, N, C)
                
                # Input lengths (time dimension of output)
                input_lengths = torch.full(
                    size=(output.size(1),), fill_value=output.size(0), dtype=torch.long, device=self.device
                )

                # Compute loss
                loss = self.criterion(output, targets, input_lengths, target_lengths)
                train_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Scheduler step
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(train_loader):.4f}")

            # Validation Phase
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = os.path.join(self.sys_path, f"../outputs/best_{dataset_name}_model.pth")
                torch.save(self.model.state_dict(), path)
                print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel_spec, targets in val_loader:
                mel_spec, targets = mel_spec.to(self.device), targets.to(self.device)
                target_lengths = torch.tensor([1 for t in targets]).to(self.device)
                features = mel_spec.permute(0, 2, 1).to(self.device)  # (batch, time, features)

                # Forward pass
                output = self.model(features)
                output = output.log_softmax(2)
                output = output.permute(1, 0, 2)

                # Input lengths (time dimension of output)
                input_lengths = torch.full(
                    size=(output.size(1),), fill_value=output.size(0), dtype=torch.long, device=self.device
                )

                # Compute loss
                loss = self.criterion(output, targets, input_lengths, target_lengths)
                val_loss += loss.item()
        return val_loss / len(val_loader)
