import torch
import torch.nn as nn


class LSTMIntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        dropout=0.2
    ):

        super().__init__()

        # ----------------------------------------------------
        # WORD EMBEDDING
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length]

        embedded = self.embedding(x)

        # embedded:
        # [batch_size, sequence_length, embedding_dim]

        output, (hidden, cell) = self.lstm(
            embedded
        )

        # hidden:
        # [num_layers, batch_size, hidden_dim]

        # Take the final LSTM hidden state
        last_hidden = hidden[-1]

        # Apply dropout
        last_hidden = self.dropout(
            last_hidden
        )

        # Classification
        output = self.fc(
            last_hidden
        )

        return output
