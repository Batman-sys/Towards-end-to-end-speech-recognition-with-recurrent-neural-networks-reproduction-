import torch.nn as nn


# Bidirectional LSTM Model
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout = 0.0):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        

        output = self.fc(lstm_out)
        return output

