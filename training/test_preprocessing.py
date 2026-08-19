from text_preprocessor import TextPreprocessor
from label_encoder import LabelEncoder
import pandas as pd


def main():

    # Load training dataset
    df = pd.read_csv("data/train.csv")

    texts = df["text"].tolist()
    intents = df["intent"].tolist()

    # -----------------------------
    # TEXT PREPROCESSING
    # -----------------------------

    processor = TextPreprocessor()

    # Build vocabulary using training data only
    processor.build_vocabulary(texts)

    print("===== VOCABULARY =====")

    print("Vocabulary size:", len(processor.word_to_index))

    print("\nFirst 20 vocabulary entries:")

    for word, index in list(processor.word_to_index.items())[:20]:
        print(f"{word} -> {index}")

    # -----------------------------
    # TEST COMMAND
    # -----------------------------

    command = "Could you launch YouTube for me?"

    print("\n===== TEST COMMAND =====")
    print("Original:", command)

    cleaned = processor.clean_text(command)

    print("Cleaned:", cleaned)

    words = cleaned.split()

    print("Tokens:", words)

    encoded = processor.encode_text(command)

    print("Encoded:", encoded)

    # -----------------------------
    # LABEL ENCODING
    # -----------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(intents)

    print("\n===== INTENT LABELS =====")

    for label, index in label_encoder.label_to_index.items():
        print(f"{label} -> {index}")

    print("\nOPEN_WEBSITE encoded as:",
          label_encoder.encode("OPEN_WEBSITE"))


if __name__ == "__main__":
    main()
