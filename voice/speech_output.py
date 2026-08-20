import pyttsx3


# ============================================================
# INITIALIZE SPEECH ENGINE
# ============================================================

engine = pyttsx3.init()


# ============================================================
# CONFIGURATION
# ============================================================

engine.setProperty(
    "rate",
    170
)

engine.setProperty(
    "volume",
    1.0
)


# ============================================================
# SPEAK
# ============================================================

def speak(text):

    print(f"🔊 JARVIS: {text}")

    engine.say(text)

    engine.runAndWait()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       JARVIS-X TEXT-TO-SPEECH TEST")
    print("=" * 60)

    speak(
        "Hello. I am Jarvis. "
        "Text to speech is working successfully."
    )
