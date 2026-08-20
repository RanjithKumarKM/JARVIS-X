import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.text_preprocessor import TextPreprocessor
from training.label_encoder import LabelEncoder
from ai_engine.intent_model import IntentClassifier


# -----------------------------
# CONFIGURATION
# -----------------------------
TRAIN_PATH = "data/train_v3.csv"
VAL_PATH = "data/validation_v3.csv"

BATCH_SIZE = 8
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 30
MAX_LENGTH = 8


# -----------------------------
# DATASET CLASS
# -----------------------------

class JarvisDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        text_processor,
        label_encoder
    ):

        self.texts = texts
        self.labels = labels

        self.text_processor = text_processor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoded = self.text_processor.encode_text(text)

        # Truncate long sequences
        encoded = encoded[:MAX_LENGTH]

        # Padding
        while len(encoded) < MAX_LENGTH:
            encoded.append(0)

        encoded = torch.tensor(
            encoded,
            dtype=torch.long
        )

        label = torch.tensor(
            self.label_encoder.encode(label),
            dtype=torch.long
        )

        return encoded, label


# -----------------------------
# MAIN TRAINING FUNCTION
# -----------------------------

def main():

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # -----------------------------
    # TEXT PREPROCESSOR
    # -----------------------------

    text_processor = TextPreprocessor()

    text_processor.build_vocabulary(
        train_df["text"].tolist()
    )

    # -----------------------------
    # LABEL ENCODER
    # -----------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        train_df["intent"].tolist()
    )

    # -----------------------------
    # DATASETS
    # -----------------------------

    train_dataset = JarvisDataset(
        train_df["text"].tolist(),
        train_df["intent"].tolist(),
        text_processor,
        label_encoder
    )

    val_dataset = JarvisDataset(
        val_df["text"].tolist(),
        val_df["intent"].tolist(),
        text_processor,
        label_encoder
    )

    # -----------------------------
    # DATA LOADERS
    # -----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    # -----------------------------
    # DEVICE
    # -----------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    # -----------------------------
    # MODEL
    # -----------------------------

    model = IntentClassifier(
        vocab_size=len(text_processor.word_to_index),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(label_encoder.label_to_index)
    )

    model = model.to(device)

    print("\n===== MODEL =====")
    print(model)

    # -----------------------------
    # LOSS
    # -----------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # OPTIMIZER
    # -----------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------
    # TRAINING
    # -----------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = correct / total

        # -----------------------------
        # VALIDATION
        # -----------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Train Acc: {train_accuracy:.2%} "
            f"Val Acc: {val_accuracy:.2%}"
        )
    # -----------------------------
# SAVE MODEL
# -----------------------------

    # -----------------------------
# SAVE MODEL
# -----------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word_to_index": text_processor.word_to_index,
            "label_to_index": label_encoder.label_to_index,
            "index_to_label": label_encoder.index_to_label,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_length": MAX_LENGTH
        },
        "models/jarvis_intent_model.pth"
    )

    print("\n===== MODEL SAVED =====")
    print("models/jarvis_intent_model.pth")


if __name__ == "__main__":
    main()
