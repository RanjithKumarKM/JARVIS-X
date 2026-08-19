import torch
import torch.nn as nn


class IntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(self, x):

        embeddings = self.embedding(x)

        mask = (x != 0).unsqueeze(-1)

        masked_embeddings = embeddings * mask

        summed = masked_embeddings.sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        representation = summed / lengths

        output = self.network(representation)

        return output
