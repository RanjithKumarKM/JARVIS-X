import torch

from ai_engine.intent_model import IntentClassifier
from training.text_preprocessor import TextPreprocessor
from voice.speech_recognizer import listen


MODEL_PATH = "models/jarvis_intent_model.pth"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("          JARVIS-X VOICE → DEEP LEARNING TEST")
print("=" * 70)

print("\nLoading JARVIS-X V3 model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

word_to_index = checkpoint["word_to_index"]

label_to_index = checkpoint["label_to_index"]

index_to_label = {
    int(index): label
    for index, label in checkpoint["index_to_label"].items()
}

embedding_dim = checkpoint["embedding_dim"]
hidden_dim = checkpoint["hidden_dim"]
max_length = checkpoint["max_length"]


# ============================================================
# CREATE MODEL
# ============================================================

model = IntentClassifier(
    vocab_size=len(word_to_index),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_classes=len(label_to_index)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# TEXT PREPROCESSOR
# ============================================================

text_processor = TextPreprocessor()

text_processor.word_to_index = word_to_index


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_intent(text):

    encoded = text_processor.encode_text(text)

    encoded = encoded[:max_length]

    while len(encoded) < max_length:
        encoded.append(0)

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    predicted_intent = index_to_label[
        predicted_index
    ]

    confidence = confidence.item() * 100

    return predicted_intent, confidence


# ============================================================
# LISTEN
# ============================================================

print(
    f"\nDevice: {device}"
)

print("\n🎤 JARVIS is ready.")

text = listen()


# ============================================================
# CLASSIFY
# ============================================================

if text:

    print("\n📝 Recognized command:")
    print(text)

    intent, confidence = predict_intent(
        text
    )

    print("\n🧠 JARVIS-X V3 RESULT")

    print(
        f"Intent:     {intent}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

else:

    print(
        "\n❌ No command was recognized."
    )
