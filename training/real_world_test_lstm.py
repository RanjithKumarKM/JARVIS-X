import torch

from ai_engine.lstm_intent_model import LSTMIntentClassifier
from training.text_preprocessor import TextPreprocessor


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/jarvis_lstm_model_v4.pth"


# ============================================================
# SAME 50 REAL-WORLD COMMANDS
# ============================================================

test_commands = [

    # OPEN WEBSITE
    ("Hey Jarvis, get me onto YouTube.", "OPEN_WEBSITE"),
    ("Take me over to GitHub.", "OPEN_WEBSITE"),
    ("I want Google open in the browser.", "OPEN_WEBSITE"),
    ("Bring up ChatGPT for me.", "OPEN_WEBSITE"),
    ("Get me to Reddit.", "OPEN_WEBSITE"),

    # OPEN APPLICATION
    ("Get the calculator up.", "OPEN_APPLICATION"),
    ("I need VS Code running.", "OPEN_APPLICATION"),
    ("Open the notepad program.", "OPEN_APPLICATION"),
    ("Can you bring up Paint?", "OPEN_APPLICATION"),
    ("Start File Explorer for me.", "OPEN_APPLICATION"),

    # SEARCH WEB
    ("I want to know how backpropagation works.", "SEARCH_WEB"),
    ("Look for information about transformers.", "SEARCH_WEB"),
    ("Find something about Python decorators.", "SEARCH_WEB"),
    ("Can you investigate CNNs?", "SEARCH_WEB"),
    ("I need information on computer vision.", "SEARCH_WEB"),

    # PLAY MEDIA
    ("I feel like listening to something.", "PLAY_MEDIA"),
    ("Put something relaxing on.", "PLAY_MEDIA"),
    ("I want some music in the background.", "PLAY_MEDIA"),
    ("Play something for studying.", "PLAY_MEDIA"),
    ("Give me something energetic to listen to.", "PLAY_MEDIA"),

    # STOP MEDIA
    ("That's enough music for now.", "STOP_MEDIA"),
    ("I don't want to hear this anymore.", "STOP_MEDIA"),
    ("Can you end what's playing?", "STOP_MEDIA"),
    ("Make the current audio stop.", "STOP_MEDIA"),
    ("I think we should pause this.", "STOP_MEDIA"),

    # SCREENSHOT
    ("Save a picture of my current display.", "TAKE_SCREENSHOT"),
    ("I need an image of what's on my desktop.", "TAKE_SCREENSHOT"),
    ("Capture everything I'm seeing right now.", "TAKE_SCREENSHOT"),
    ("Can you save what is currently visible?", "TAKE_SCREENSHOT"),
    ("Grab my current display.", "TAKE_SCREENSHOT"),

    # SYSTEM INFO
    ("How much power does my laptop have left?", "SYSTEM_INFO"),
    ("Tell me how much RAM I'm using.", "SYSTEM_INFO"),
    ("How busy is my processor?", "SYSTEM_INFO"),
    ("Give me details about this computer.", "SYSTEM_INFO"),
    ("How is my system doing right now?", "SYSTEM_INFO"),

    # VOLUME
    ("Make everything louder.", "VOLUME_CONTROL"),
    ("The sound is too high, bring it down.", "VOLUME_CONTROL"),
    ("I can't hear anything, turn the audio up.", "VOLUME_CONTROL"),
    ("Keep the computer silent.", "VOLUME_CONTROL"),
    ("Bring the audio back after muting it.", "VOLUME_CONTROL"),

    # TIMER
    ("Wake me after ten minutes.", "SET_TIMER"),
    ("Start counting down from half an hour.", "SET_TIMER"),
    ("I need a reminder in five minutes.", "SET_TIMER"),
    ("Give me a one-hour countdown.", "SET_TIMER"),
    ("Start timing my study session.", "SET_TIMER"),

    # EXIT
    ("We're finished here, Jarvis.", "EXIT_JARVIS"),
    ("You can shut down now.", "EXIT_JARVIS"),
    ("That's all I need from you.", "EXIT_JARVIS"),
    ("I'm done talking to you.", "EXIT_JARVIS"),
    ("We can end this session.", "EXIT_JARVIS"),
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("        JARVIS-X V4 LSTM REAL-WORLD TEST")
print("=" * 70)

print("\nLoading trained LSTM model...")

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
# LOAD TRAINING INFORMATION
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

num_layers = checkpoint.get(
    "num_layers",
    1
)

dropout = checkpoint.get(
    "dropout",
    0.2
)


# ============================================================
# CREATE MODEL
# ============================================================

model = LSTMIntentClassifier(

    vocab_size=len(word_to_index),

    embedding_dim=embedding_dim,

    hidden_dim=hidden_dim,

    num_classes=len(label_to_index),

    num_layers=num_layers,

    dropout=dropout
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

def predict(text):

    encoded = text_processor.encode_text(
        text
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

    predicted_index = prediction.item()

    predicted_intent = index_to_label[
        predicted_index
    ]

    confidence = confidence.item() * 100

    return predicted_intent, confidence


# ============================================================
# RUN TEST
# ============================================================

correct = 0

print(
    f"\nDevice: {device}"
)

print(
    f"Total commands: {len(test_commands)}"
)

print()


for text, actual in test_commands:

    predicted, confidence = predict(
        text
    )

    if predicted == actual:

        correct += 1
        symbol = "✓"

    else:

        symbol = "✗"

    print(
        f"{symbol} {text}"
    )

    print(
        f"   Actual:    {actual}"
    )

    print(
        f"   Predicted: {predicted}"
    )

    print(
        f"   Confidence: {confidence:.2f}%"
    )

    print()


# ============================================================
# FINAL RESULT
# ============================================================

accuracy = (
    correct /
    len(test_commands)
) * 100


print("=" * 70)

print(
    f"Correct predictions: "
    f"{correct}/{len(test_commands)}"
)

print(
    f"LSTM real-world accuracy: "
    f"{accuracy:.2f}%"
)

print("=" * 70)
