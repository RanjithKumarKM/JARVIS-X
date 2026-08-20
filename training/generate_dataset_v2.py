import pandas as pd
import random
from pathlib import Path


random.seed(42)

SAMPLES_PER_INTENT = 100


# ============================================================
# INTENT-SPECIFIC EXAMPLES
# ============================================================

examples = {

    # --------------------------------------------------------
    # OPEN WEBSITE
    # --------------------------------------------------------

    "OPEN_WEBSITE": [
        "open youtube in my browser",
        "please open youtube website",
        "navigate to youtube",
        "go over to youtube",
        "show me youtube",

        "open google in my browser",
        "please open google website",
        "navigate to google",
        "go over to google",
        "show me google",

        "open github in my browser",
        "please open github website",
        "navigate to github",
        "go over to github",
        "show me github",

        "open chatgpt in my browser",
        "please open chatgpt website",
        "navigate to chatgpt",
        "go over to chatgpt",
        "show me chatgpt",

        "open youtube",
        "launch youtube",
        "start youtube",
        "bring up youtube",
        "get youtube running",
        "take me to youtube",
        "go to youtube",
        "open the youtube website",
        "can you open youtube",
        "please launch youtube",

        "open google",
        "launch google",
        "start google",
        "bring up google",
        "get google running",
        "take me to google",
        "go to google",
        "open the google website",
        "can you open google",
        "please launch google",

        "open github",
        "launch github",
        "start github",
        "bring up github",
        "get github running",
        "take me to github",
        "go to github",
        "open the github website",
        "can you open github",
        "please launch github",

        "open chatgpt",
        "launch chatgpt",
        "start chatgpt",
        "bring up chatgpt",
        "get chatgpt running",
        "take me to chatgpt",
        "go to chatgpt",
        "open the chatgpt website",
        "can you open chatgpt",
        "please launch chatgpt",

        "open gmail",
        "launch gmail",
        "start gmail",
        "bring up gmail",
        "get gmail running",
        "take me to gmail",
        "go to gmail",
        "open the gmail website",
        "can you open gmail",
        "please launch gmail",

        "open wikipedia",
        "launch wikipedia",
        "start wikipedia",
        "bring up wikipedia",
        "get wikipedia running",
        "take me to wikipedia",
        "go to wikipedia",
        "open the wikipedia website",
        "can you open wikipedia",
        "please launch wikipedia",

        "open reddit",
        "launch reddit",
        "start reddit",
        "bring up reddit",
        "get reddit running",
        "take me to reddit",
        "go to reddit",
        "open the reddit website",
        "can you open reddit",
        "please launch reddit",

        "open linkedin",
        "launch linkedin",
        "start linkedin",
        "bring up linkedin",
        "get linkedin running",
        "take me to linkedin",
        "go to linkedin",
        "open the linkedin website",
        "can you open linkedin",
        "please launch linkedin"
    ],


    # --------------------------------------------------------
    # OPEN APPLICATION
    # --------------------------------------------------------

    "OPEN_APPLICATION": [

        "open calculator",
        "launch calculator",
        "start calculator",
        "run calculator",
        "bring up calculator",
        "get calculator running",
        "open the calculator application",
        "can you open calculator",
        "please launch calculator",
        "start the calculator program",

        "open notepad",
        "launch notepad",
        "start notepad",
        "run notepad",
        "bring up notepad",
        "get notepad running",
        "open the notepad application",
        "can you open notepad",
        "please launch notepad",
        "start the notepad program",

        "open paint",
        "launch paint",
        "start paint",
        "run paint",
        "bring up paint",
        "get paint running",
        "open the paint application",
        "can you open paint",
        "please launch paint",
        "start the paint program",

        "open visual studio code",
        "launch visual studio code",
        "start visual studio code",
        "run visual studio code",
        "bring up visual studio code",
        "get visual studio code running",
        "open the visual studio code application",
        "can you open visual studio code",
        "please launch visual studio code",
        "start the visual studio code program",

        "open file explorer",
        "launch file explorer",
        "start file explorer",
        "run file explorer",
        "bring up file explorer",
        "get file explorer running",
        "open the file explorer application",
        "can you open file explorer",
        "please launch file explorer",
        "start the file explorer program"
    ],


    # --------------------------------------------------------
    # SEARCH WEB
    # --------------------------------------------------------

    "SEARCH_WEB": [

        "search for python tutorials",
        "search python tutorials",
        "look up python tutorials",
        "find information about python tutorials",
        "find python tutorials",

        "search for machine learning",
        "search machine learning",
        "look up machine learning",
        "find information about machine learning",
        "find machine learning",

        "search for deep learning",
        "search deep learning",
        "look up deep learning",
        "find information about deep learning",
        "find deep learning",

        "search for computer vision",
        "search computer vision",
        "look up computer vision",
        "find information about computer vision",
        "find computer vision",

        "search for neural networks",
        "search neural networks",
        "look up neural networks",
        "find information about neural networks",
        "find neural networks",

        "search for cnn",
        "search cnn",
        "look up cnn",
        "find information about cnn",
        "find cnn",

        "search for lstm",
        "search lstm",
        "look up lstm",
        "find information about lstm",
        "find lstm",

        "search for artificial intelligence",
        "search artificial intelligence",
        "look up artificial intelligence",
        "find information about artificial intelligence",
        "find artificial intelligence",

        "search for data science",
        "search data science",
        "look up data science",
        "find information about data science",
        "find data science",

        "search for transformers",
        "search transformers",
        "look up transformers",
        "find information about transformers",
        "find transformers",

        "search for natural language processing",
        "search natural language processing",
        "look up natural language processing",
        "find information about natural language processing",
        "find natural language processing",

        "search for cybersecurity",
        "search cybersecurity",
        "look up cybersecurity",
        "find information about cybersecurity",
        "find cybersecurity",

        "search for cloud computing",
        "search cloud computing",
        "look up cloud computing",
        "find information about cloud computing",
        "find cloud computing",

        "search for software engineering",
        "search software engineering",
        "look up software engineering",
        "find information about software engineering",
        "find software engineering",

        "search for python programming",
        "search python programming",
        "look up python programming",
        "find information about python programming",
        "find python programming",

        "search for convolutional neural networks",
        "search convolutional neural networks",
        "look up convolutional neural networks",
        "find information about convolutional neural networks",
        "find convolutional neural networks",

        "search for reinforcement learning",
        "search reinforcement learning",
        "look up reinforcement learning",
        "find information about reinforcement learning",
        "find reinforcement learning",

        "search for generative ai",
        "search generative ai",
        "look up generative ai",
        "find information about generative ai",
        "find generative ai",

        "search for pytorch",
        "search pytorch",
        "look up pytorch",
        "find information about pytorch",
        "find pytorch",

        "search for tensorflow",
        "search tensorflow",
        "look up tensorflow",
        "find information about tensorflow",
        "find tensorflow"
    ],


    # --------------------------------------------------------
    # PLAY MEDIA
    # --------------------------------------------------------

    "PLAY_MEDIA": [

        "play music",
        "play some music",
        "start playing music",
        "play relaxing music",
        "play study music",
        "play lofi music",
        "play classical music",
        "play workout music",
        "play some songs",
        "play a song",

        "start music",
        "start some music",
        "start playing songs",
        "start a song",
        "start playing relaxing music",
        "start study music",
        "start lofi music",
        "start classical music",
        "start workout music",
        "start a podcast",

        "play a podcast",
        "play some podcasts",
        "start a podcast",
        "start some podcasts",
        "play a video",
        "play some videos",
        "start a video",
        "start playing a video",
        "play a playlist",
        "start my playlist",

        "put on some music",
        "put on a song",
        "put on relaxing music",
        "put on some lofi music",
        "put on a podcast",
        "put on a playlist",
        "I want to hear some music",
        "I want to listen to music",
        "I want to hear a song",
        "I want to listen to a podcast",

        "can you play music",
        "can you play a song",
        "can you play relaxing music",
        "can you start the music",
        "can you play a podcast",
        "can you start a playlist",
        "please play music",
        "please play a song",
        "please start the music",
        "please play a video",

        "play something for studying",
        "play something relaxing",
        "play something energetic",
        "play something calm",
        "play something for working",
        "play something for exercising",
        "play something to relax",
        "play something to study",
        "play something in the background",
        "play something for me"
    ],


    # --------------------------------------------------------
    # STOP MEDIA
    # --------------------------------------------------------

    "STOP_MEDIA": [

        "stop the music",
        "stop playing",
        "stop playback",
        "pause the music",
        "pause the song",
        "pause playback",
        "stop the song",
        "stop the video",
        "pause the video",
        "stop what's playing",

        "stop the current song",
        "pause what is playing",
        "end music playback",
        "stop audio playback",
        "end the music",
        "stop the current media",
        "pause the current media",
        "please stop the music",
        "please pause the song",
        "can you stop playback",

        "can you pause the music",
        "stop whatever is playing",
        "end the current playback",
        "halt the music",
        "pause what is currently playing",
        "stop the audio",
        "pause the audio",
        "stop the current video",
        "pause the current video",
        "stop listening",

        "stop the song now",
        "pause the song now",
        "stop my music",
        "pause my music",
        "stop the playlist",
        "pause the playlist",
        "stop the podcast",
        "pause the podcast",
        "stop the video now",
        "pause the video now",

        "please stop playback",
        "please pause playback",
        "please stop the video",
        "please pause the video",
        "please stop the audio",
        "please pause the audio",
        "could you stop the music",
        "could you pause the music",
        "could you stop the song",
        "could you pause the song",

        "end the song",
        "end the video",
        "end the podcast",
        "end the playlist",
        "finish the music",
        "finish playback",
        "stop what is playing",
        "pause what is playing",
        "stop everything playing",
        "pause everything playing",

        "I want to stop the music",
        "I want to pause the music",
        "I want to stop the song",
        "I want to pause the song",
        "I want to stop the video",
        "I want to pause the video",
        "stop the media",
        "pause the media",
        "stop playing now",
        "pause playing now",

        "stop music playback",
        "pause music playback",
        "stop video playback",
        "pause video playback",
        "stop audio playback now",
        "pause audio playback now",
        "stop the current playback",
        "pause the current playback",
        "stop whatever is playing now",
        "pause whatever is playing now"
    ],


    # --------------------------------------------------------
    # SCREENSHOT
    # --------------------------------------------------------

    "TAKE_SCREENSHOT": [

        "take a screenshot",
        "capture the screen",
        "take a screen capture",
        "capture my screen",
        "capture the display",
        "take a picture of the screen",
        "save a screenshot",
        "take a screenshot of my screen",
        "screenshot this",
        "capture what's on my screen",

        "grab a screenshot",
        "save the current screen",
        "capture the current display",
        "take a screen shot",
        "please take a screenshot",
        "can you capture the screen",
        "take a snapshot of the screen",
        "capture what is displayed",
        "save what is on the screen",
        "get a screenshot",

        "capture my display",
        "take a picture of my display",
        "save the screen",
        "save my screen",
        "capture this screen",
        "capture this display",
        "take a screenshot now",
        "capture the screen now",
        "get a screen capture",
        "create a screenshot",

        "please capture my screen",
        "please save a screenshot",
        "please take a screen capture",
        "can you take a screenshot",
        "can you save my screen",
        "can you capture this screen",
        "could you take a screenshot",
        "could you capture the screen",
        "take a screenshot for me",
        "capture the screen for me",

        "I need a screenshot",
        "I need a screen capture",
        "I want a screenshot",
        "I want a screen capture",
        "take an image of the screen",
        "save an image of the screen",
        "capture everything on screen",
        "capture the current screen",
        "capture the visible screen",
        "save the current display",

        "screenshot my computer",
        "capture my computer screen",
        "take a screenshot of the computer",
        "capture my desktop",
        "take a picture of my desktop",
        "save my desktop",
        "capture the desktop screen",
        "take a desktop screenshot",
        "get a picture of the screen",
        "get an image of the screen",

        "screenshot now",
        "capture now",
        "capture screen now",
        "take screen capture now",
        "save screen now",
        "take the screenshot",
        "get the screenshot",
        "make a screenshot",
        "make a screen capture",
        "capture the screen please",

        "take my screenshot",
        "take the screen shot",
        "capture my current screen",
        "save my current screen",
        "capture what I see",
        "save what I see",
        "take a picture of what is on screen",
        "capture what is on screen",
        "take a snapshot of my display",
        "save a snapshot of my display"
    ],


    # --------------------------------------------------------
    # SYSTEM INFO
    # --------------------------------------------------------

    "SYSTEM_INFO": [

        "check my battery",
        "show my battery percentage",
        "what is my battery level",
        "check the battery level",
        "how much battery do I have",
        "show battery status",
        "tell me my battery percentage",
        "what percentage is my battery",
        "check battery status",
        "show my battery",

        "check CPU usage",
        "show CPU usage",
        "what is my CPU usage",
        "check processor usage",
        "show processor usage",
        "how much CPU am I using",
        "check my processor",
        "show CPU status",
        "tell me CPU usage",
        "what is the processor usage",

        "show memory usage",
        "check RAM usage",
        "how much memory am I using",
        "check my RAM",
        "show RAM status",
        "tell me memory usage",
        "what is my RAM usage",
        "check memory status",
        "show available memory",
        "how much RAM is available",

        "show system information",
        "give me system information",
        "show my computer information",
        "check system status",
        "what are my system specifications",
        "show computer status",
        "give me my system details",
        "show system details",
        "check my computer",
        "show computer information",

        "what are my computer specs",
        "show my computer specs",
        "tell me my system specs",
        "what is my system status",
        "check computer status",
        "give me computer details",
        "show device information",
        "check device information",
        "show device status",
        "tell me about my computer",

        "check battery and CPU",
        "show battery and CPU",
        "show system status",
        "check system information",
        "show hardware information",
        "show hardware status",
        "check hardware information",
        "tell me system information",
        "tell me computer information",
        "tell me my system details"
    ],


    # --------------------------------------------------------
    # VOLUME CONTROL
    # --------------------------------------------------------

    "VOLUME_CONTROL": [

        "increase the volume",
        "turn the volume up",
        "make it louder",
        "raise the volume",
        "increase the sound",
        "turn up the audio",
        "make the sound louder",
        "increase audio level",
        "raise the audio level",
        "turn the sound up",

        "lower the volume",
        "turn the volume down",
        "make it quieter",
        "decrease the volume",
        "lower the sound",
        "turn down the audio",
        "decrease audio level",
        "lower the audio level",
        "turn the sound down",
        "make the audio quieter",

        "mute the sound",
        "mute my laptop",
        "silence the audio",
        "mute the computer",
        "mute the audio",
        "silence my laptop",
        "turn the sound off",
        "turn off the audio",
        "mute my computer",
        "silence the computer",

        "unmute the sound",
        "turn the sound back on",
        "restore the audio",
        "unmute my laptop",
        "bring the sound back",
        "unmute the audio",
        "turn audio back on",
        "restore sound",
        "turn the volume back on",
        "enable the sound",

        "change the volume",
        "adjust the volume",
        "control the volume",
        "change audio level",
        "adjust audio",
        "control audio",
        "set the volume",
        "modify the volume",
        "change sound level",
        "adjust sound level",

        "please increase the volume",
        "please decrease the volume",
        "please mute the sound",
        "please unmute the sound",
        "can you increase the volume",
        "can you decrease the volume",
        "can you mute the sound",
        "can you unmute the sound",
        "make the volume louder",
        "make the volume quieter",

        "I want more volume",
        "I want less volume",
        "I want to mute the sound",
        "I want to unmute the sound",
        "make my computer louder",
        "make my computer quieter",
        "turn up my computer volume",
        "turn down my computer volume",
        "silence my computer",
        "restore my computer audio"
    ],


    # --------------------------------------------------------
    # SET TIMER
    # --------------------------------------------------------

    "SET_TIMER": [

        "set a timer",
        "start a timer",
        "set a countdown",
        "start a countdown",
        "set a timer for 10 seconds",
        "set a timer for 30 seconds",
        "set a timer for 1 minute",
        "set a timer for 5 minutes",
        "set a timer for 10 minutes",
        "set a timer for 15 minutes",

        "start a timer for 30 minutes",
        "start a timer for 1 hour",
        "set a countdown for 5 minutes",
        "set a countdown for 10 minutes",
        "start a countdown for 15 minutes",
        "start a countdown for 30 minutes",
        "create a timer for 1 minute",
        "create a timer for 10 minutes",
        "set my timer to 20 minutes",
        "set my timer to 30 minutes",

        "please set a timer",
        "can you set a timer",
        "I need a timer",
        "I need a timer for 5 minutes",
        "I need a timer for 10 minutes",
        "set a study timer",
        "set a study timer for 30 minutes",
        "start my study timer",
        "start my timer",
        "begin a countdown",

        "begin a countdown for 10 minutes",
        "begin a countdown for 30 minutes",
        "count down for 1 minute",
        "count down for 5 minutes",
        "count down for 10 minutes",
        "count down for 30 minutes",
        "start my countdown",
        "start my countdown for 10 minutes",
        "set a countdown timer",
        "create a countdown",

        "set a 1 minute timer",
        "set a 5 minute timer",
        "set a 10 minute timer",
        "set a 15 minute timer",
        "set a 30 minute timer",
        "set a 1 hour timer",
        "start a 5 minute timer",
        "start a 10 minute timer",
        "start a 30 minute timer",
        "start a 1 hour timer",

        "remind me after 5 minutes",
        "remind me after 10 minutes",
        "remind me after 15 minutes",
        "remind me after 30 minutes",
        "remind me after 1 hour",
        "countdown for 5 minutes",
        "countdown for 10 minutes",
        "countdown for 15 minutes",
        "countdown for 30 minutes",
        "countdown for 1 hour",

        "set a timer for my study session",
        "set a timer for cooking",
        "set a timer for exercise",
        "set a timer for work",
        "start a study countdown",
        "start a cooking timer",
        "start an exercise timer",
        "start a work timer",
        "set my study countdown",
        "set my work countdown",

        "please start a countdown",
        "please start a timer",
        "please create a timer",
        "please create a countdown",
        "could you set a timer",
        "could you start a countdown",
        "could you create a timer",
        "can you start a countdown",
        "can you create a timer",
        "set my countdown"
    ],


    # --------------------------------------------------------
    # EXIT JARVIS
    # --------------------------------------------------------

    "EXIT_JARVIS": [

        "exit",
        "quit",
        "stop jarvis",
        "close jarvis",
        "exit jarvis",
        "quit jarvis",
        "goodbye jarvis",
        "end the session",
        "close the assistant",
        "shut down jarvis",

        "I want to quit",
        "I want to exit",
        "stop the assistant",
        "end jarvis",
        "terminate the assistant",
        "close the program",
        "please exit",
        "please quit",
        "you can stop now",
        "I'm done",

        "that's all jarvis",
        "goodbye",
        "end the assistant",
        "shut down the assistant",
        "finish the session",
        "stop the program",
        "close the program now",
        "end the program",
        "terminate jarvis",
        "turn off jarvis",

        "jarvis stop",
        "jarvis quit",
        "jarvis exit",
        "jarvis close",
        "jarvis goodbye",
        "stop the assistant now",
        "quit the assistant now",
        "exit the assistant now",
        "close jarvis now",
        "shut jarvis down",

        "I am finished",
        "I'm finished",
        "I am done",
        "I'm done with jarvis",
        "we are done",
        "end this",
        "end everything",
        "stop now",
        "close everything",
        "terminate the session",

        "please stop jarvis",
        "please close jarvis",
        "please exit jarvis",
        "please quit jarvis",
        "please shut down jarvis",
        "can you stop jarvis",
        "can you close jarvis",
        "can you exit jarvis",
        "can you quit jarvis",
        "can you shut down jarvis",

        "I want to stop",
        "I want to close jarvis",
        "I want to end the session",
        "I want to shut down jarvis",
        "stop this session",
        "close this session",
        "end this session",
        "finish this session",
        "we can stop now",
        "you can close now",

        "goodbye",
        "see you later jarvis",
        "bye jarvis",
        "bye assistant",
        "goodbye assistant",
        "that's enough",
        "that's all",
        "I'm finished here",
        "we're finished",
        "session complete"
    ]
}


# ============================================================
# DATASET AUGMENTATION
# ============================================================

def generate_variations(text):
    """
    Generate natural variations of an existing command.
    Used only when an intent has fewer than 100 examples.
    """

    variations = set()

    prefixes = [
        "please",
        "can you",
        "could you",
        "would you",
        "hey jarvis",
        "jarvis",
        "I want you to",
        "I need you to",
        "I would like you to",
        "help me",
        "please help me",
        "I want to",
        "I need to",
        "could you please",
        "can you please",
        "would you please",
    ]

    suffixes = [
        "please",
        "for me",
        "right now",
        "now",
        "when you can",
        "for me please",
        "right away",
        "immediately",
    ]

    # Original text
    variations.add(text.strip())

    # Prefix variations
    for prefix in prefixes:
        variations.add(f"{prefix} {text}".strip())

    # Suffix variations
    for suffix in suffixes:
        variations.add(f"{text} {suffix}".strip())

    # Prefix + suffix combinations
    for prefix in prefixes:
        for suffix in suffixes:
            variations.add(
                f"{prefix} {text} {suffix}".strip()
            )

    # Capitalization is NOT used because the dataset should
    # contain genuinely different utterances rather than
    # meaningless capitalization changes.

    return variations


def expand_intent(intent, intent_list, target_count=100):
    """
    Expand an intent until it contains target_count unique
    examples.

    Existing examples are always preserved.
    """

    # Remove duplicates
    unique_examples = set(
        text.strip()
        for text in intent_list
        if text.strip()
    )

    # Already enough examples
    if len(unique_examples) >= target_count:
        return list(unique_examples)

    original_examples = list(unique_examples)

    # Generate variations from existing examples
    for text in original_examples:

        generated = generate_variations(text)

        for variation in generated:
            if variation not in unique_examples:
                unique_examples.add(variation)

            if len(unique_examples) >= target_count:
                break

        if len(unique_examples) >= target_count:
            break

    # Safety check
    if len(unique_examples) < target_count:
        raise ValueError(
            f"{intent} could only generate "
            f"{len(unique_examples)} unique examples. "
            f"Need {target_count}."
        )

    return list(unique_examples)


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset():

    rows = []

    print("\n===== DATASET GENERATION =====")

    for intent, intent_list in examples.items():

        # ----------------------------------------------------
        # Expand intent to exactly 100 unique examples
        # ----------------------------------------------------

        unique_examples = expand_intent(
            intent,
            intent_list,
            SAMPLES_PER_INTENT
        )

        # Randomly select exactly 100
        selected = random.sample(
            unique_examples,
            SAMPLES_PER_INTENT
        )

        # Add rows
        for text in selected:

            rows.append({
                "text": text,
                "intent": intent
            })

        print(
            f"{intent:<20} "
            f"{len(unique_examples):>3} unique → "
            f"{len(selected):>3} selected"
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Shuffle entire dataset
    # --------------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df

# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(df):

    print("\n===== QUALITY CHECK =====")

    print(
        "Missing values:",
        df.isnull().sum().sum()
    )

    print(
        "Duplicate rows:",
        df.duplicated().sum()
    )

    print(
        "Unique texts:",
        df["text"].nunique()
    )

    print("\nExamples per intent:")
    print(df["intent"].value_counts())

    print("\n===== SAMPLE =====")
    print(
        df.sample(
            20,
            random_state=42
        ).to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Creating JARVIS-X Dataset V2...")

    df = create_dataset()

    output_path = Path(
        "data/intents_v2.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n===== DATASET V2 CREATED =====")

    print(
        f"Total examples: {len(df)}"
    )

    print(
        f"Total intents: {df['intent'].nunique()}"
    )

    quality_check(df)

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()
