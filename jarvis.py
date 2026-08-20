import torch
import time

from ai_engine.intent_model import IntentClassifier
from training.text_preprocessor import TextPreprocessor

from voice.speech_recognizer import listen

from actions.action_engine import execute_action
from voice.speech_output import speak


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/jarvis_intent_model.pth"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    word_to_index = checkpoint[
        "word_to_index"
    ]

    label_to_index = checkpoint[
        "label_to_index"
    ]

    index_to_label = {
        int(index): label
        for index, label
        in checkpoint[
            "index_to_label"
        ].items()
    }

    model = IntentClassifier(
        vocab_size=len(
            word_to_index
        ),
        embedding_dim=checkpoint[
            "embedding_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
        num_classes=len(
            label_to_index
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()

    text_processor = TextPreprocessor()

    text_processor.word_to_index = (
        word_to_index
    )

    return (
        model,
        text_processor,
        index_to_label,
        device,
        checkpoint["max_length"]
    )


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(
    text,
    model,
    text_processor,
    index_to_label,
    device,
    max_length
):

    encoded = (
        text_processor.encode_text(
            text
        )
    )

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

    predicted_index = (
        prediction.item()
    )

    intent = index_to_label[
        predicted_index
    ]

    confidence = (
        confidence.item() * 100
    )

    return intent, confidence


# ============================================================
# MAIN JARVIS LOOP
# ============================================================

def main():

    print("=" * 70)
    print("                 JARVIS-X")
    print("=" * 70)

    print("\nLoading Deep Learning model...")

    (
        model,
        text_processor,
        index_to_label,
        device,
        max_length
    ) = load_model()

    print(
        f"Model loaded successfully."
    )

    print(
        f"Device: {device}"
    )

    print(
        "\n🎤 JARVIS is ready."
    )

    print(
        "Say a command."
    )

    print(
        "Press Ctrl+C to stop."
    )

    while True:

        try:

            # ------------------------------------------------
            # LISTEN
            # ------------------------------------------------

            text = listen()

            if not text:

                continue

            print(
                f"\n📝 Command: {text}"
            )

            # ------------------------------------------------
            # INTENT
            # ------------------------------------------------

            intent, confidence = (
                predict_intent(
                    text,
                    model,
                    text_processor,
                    index_to_label,
                    device,
                    max_length
                )
            )

            print(
                f"🧠 Intent: {intent}"
            )

            print(
                f"📊 Confidence: "
                f"{confidence:.2f}%"
            )

            # ------------------------------------------------
            # ACTION
            # ------------------------------------------------

            if intent in [
                "OPEN_WEBSITE",
                "OPEN_APPLICATION"
            ]:

                success = execute_action(
                    intent,
                    text
                )

                if success:

                    if intent == "OPEN_WEBSITE":

                        speak(
                            "Opening the website."
                        )

                    elif intent == "OPEN_APPLICATION":

                        speak(
                            "Opening the application."
                        )

                else:

                    speak(
                        "Sorry. I couldn't complete that action."
                    )

                print(
                    "\n⏳ Ready for your next command..."
                )

                time.sleep(1.5)
            else:

                print(
                    f"⚠️ Action for "
                    f"{intent} "
                    f"is not implemented yet."
                )

        except KeyboardInterrupt:

            print(
                "\n\n👋 JARVIS shutting down."
            )

            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
