import time
import speech_recognition as sr


# ============================================================
# CONFIGURATION
# ============================================================

AMBIENT_NOISE_DURATION = 1.0

# Maximum time to wait for you to START speaking
LISTEN_TIMEOUT = 10

# Maximum duration of one spoken command
PHRASE_TIME_LIMIT = 10

# How long silence must occur before the phrase is considered finished
PAUSE_THRESHOLD = 1.2

# Minimum audio energy considered speech
ENERGY_THRESHOLD = 300

# Delay after finishing one command
COMMAND_DELAY = 1.0


# ============================================================
# LISTEN
# ============================================================

def listen():

    recognizer = sr.Recognizer()

    # --------------------------------------------------------
    # Voice sensitivity
    # --------------------------------------------------------

    recognizer.pause_threshold = PAUSE_THRESHOLD

    recognizer.energy_threshold = ENERGY_THRESHOLD

    recognizer.dynamic_energy_threshold = True

    # --------------------------------------------------------
    # Microphone
    # --------------------------------------------------------

    with sr.Microphone() as source:

        print("\n🎤 Listening...")
        print("   You can speak now.")

        # Adjust microphone for background noise
        recognizer.adjust_for_ambient_noise(
            source,
            duration=AMBIENT_NOISE_DURATION
        )

        try:

            # ------------------------------------------------
            # Wait for user to start speaking
            # ------------------------------------------------

            audio = recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )

        except sr.WaitTimeoutError:

            print(
                "⏱️ No speech detected."
            )

            return None

    # --------------------------------------------------------
    # Speech recognition
    # --------------------------------------------------------

    print(
        "🔄 Converting speech to text..."
    )

    try:

        text = recognizer.recognize_google(
            audio
        )

        return text

    except sr.UnknownValueError:

        print(
            "❌ Sorry, I couldn't understand that."
        )

        return None

    except sr.RequestError as error:

        print(
            f"❌ Speech recognition service error: "
            f"{error}"
        )

        return None
