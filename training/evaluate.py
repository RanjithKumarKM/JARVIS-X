import pandas as pd
import torch
from torch.utils.data import DataLoader

from training.text_preprocessor import TextPreprocessor
from training.label_encoder import LabelEncoder
from training.train import JarvisDataset
from ai_engine.intent_model import IntentClassifier


TEST_PATH = "data/test.csv"

BATCH_SIZE = 8


def main():

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    checkpoint = torch.load(
        "models/jarvis_intent_model.pth",
        map_location="cpu"
    )

    # -----------------------------
    # RESTORE PREPROCESSOR
    # -----------------------------

    text_processor = TextPreprocessor()

    text_processor.word_to_index = checkpoint["word_to_index"]

    text_processor.index_to_word = {
        index: word
        for word, index in text_processor.word_to_index.items()
    }

    # -----------------------------
    # RESTORE LABEL ENCODER
    # -----------------------------

    label_encoder = LabelEncoder()

    label_encoder.label_to_index = checkpoint["label_to_index"]

    label_encoder.index_to_label = {
        int(index): label
        for label, index
        in label_encoder.label_to_index.items()
    }

    # -----------------------------
    # TEST DATASET
    # -----------------------------

    test_dataset = JarvisDataset(
        test_df["text"].tolist(),
        test_df["intent"].tolist(),
        text_processor,
        label_encoder
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = IntentClassifier(
        vocab_size=len(text_processor.word_to_index),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=len(label_encoder.label_to_index)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------
    # EVALUATION
    # -----------------------------

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = model(inputs)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    accuracy = correct / total

    print("\n===== TEST RESULTS =====")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Test accuracy: {accuracy:.2%}")

    print("\n===== PREDICTIONS =====")

    for i in range(len(test_df)):

        actual_index = all_labels[i]
        predicted_index = all_predictions[i]

        actual = label_encoder.decode(
            actual_index
        )

        predicted = label_encoder.decode(
            predicted_index
        )

        status = "✓" if actual == predicted else "✗"

        print(
            f"{status} "
            f"{test_df.iloc[i]['text']} "
            f"→ Actual: {actual} "
            f"| Predicted: {predicted}"
        )


if __name__ == "__main__":
    main()
