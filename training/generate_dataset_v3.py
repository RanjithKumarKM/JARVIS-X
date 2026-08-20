import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

V2_PATH = "data/intents_v2.csv"
V3_PATH = "data/intents_v3.csv"


# ============================================================
# HARD EXAMPLES
# ============================================================

hard_examples = [

    # --------------------------------------------------------
    # SEARCH_WEB
    # --------------------------------------------------------

    ("I want to know how backpropagation works", "SEARCH_WEB"),
    ("Can you investigate CNNs for me", "SEARCH_WEB"),
    ("I need information on computer vision", "SEARCH_WEB"),
    ("Tell me how transformers work", "SEARCH_WEB"),
    ("I'm curious about neural networks", "SEARCH_WEB"),
    ("I'd like to learn about reinforcement learning", "SEARCH_WEB"),
    ("Can you tell me about convolutional networks", "SEARCH_WEB"),
    ("I want to find out how LSTMs work", "SEARCH_WEB"),
    ("Explain what attention mechanisms are", "SEARCH_WEB"),
    ("I need some information about deep learning", "SEARCH_WEB"),


    # --------------------------------------------------------
    # STOP_MEDIA
    # --------------------------------------------------------

    ("That's enough music for now", "STOP_MEDIA"),
    ("I don't want to hear this anymore", "STOP_MEDIA"),
    ("I think we should pause this", "STOP_MEDIA"),
    ("I've heard enough of this song", "STOP_MEDIA"),
    ("I don't want this playing anymore", "STOP_MEDIA"),
    ("Make the music stop", "STOP_MEDIA"),
    ("I want the current audio to end", "STOP_MEDIA"),
    ("Can you stop what's playing", "STOP_MEDIA"),
    ("I've had enough of this audio", "STOP_MEDIA"),
    ("Please end the music", "STOP_MEDIA"),


    # --------------------------------------------------------
    # TAKE_SCREENSHOT
    # --------------------------------------------------------

    ("Save what's currently visible", "TAKE_SCREENSHOT"),
    ("I need an image of my desktop", "TAKE_SCREENSHOT"),
    ("Capture what I'm looking at", "TAKE_SCREENSHOT"),
    ("Save the current display", "TAKE_SCREENSHOT"),
    ("Give me a picture of what's on screen", "TAKE_SCREENSHOT"),
    ("I want a picture of my screen", "TAKE_SCREENSHOT"),
    ("Save an image of what I'm seeing", "TAKE_SCREENSHOT"),
    ("Get a copy of my current display", "TAKE_SCREENSHOT"),
    ("Record what's visible on my screen", "TAKE_SCREENSHOT"),
    ("Make an image of the screen right now", "TAKE_SCREENSHOT"),


    # --------------------------------------------------------
    # OPEN_WEBSITE
    # --------------------------------------------------------

    ("Hey Jarvis get me onto YouTube", "OPEN_WEBSITE"),
    ("I want to go to YouTube", "OPEN_WEBSITE"),
    ("Take me over to Google", "OPEN_WEBSITE"),
    ("Can you get me onto GitHub", "OPEN_WEBSITE"),
    ("I want to visit ChatGPT", "OPEN_WEBSITE"),
    ("Bring me to Reddit", "OPEN_WEBSITE"),
    ("Get the YouTube website up", "OPEN_WEBSITE"),
    ("I'd like to visit LinkedIn", "OPEN_WEBSITE"),
    ("Take me to the Google website", "OPEN_WEBSITE"),
    ("I want the GitHub site open", "OPEN_WEBSITE"),


    # --------------------------------------------------------
    # VOLUME_CONTROL
    # --------------------------------------------------------

    ("Keep the computer silent", "VOLUME_CONTROL"),
    ("Silence my laptop", "VOLUME_CONTROL"),
    ("Make the computer quiet", "VOLUME_CONTROL"),
    ("Keep the audio turned off", "VOLUME_CONTROL"),
    ("I want the laptop muted", "VOLUME_CONTROL"),
    ("Make everything quieter", "VOLUME_CONTROL"),
    ("The computer is too loud", "VOLUME_CONTROL"),
    ("Turn the sound down a little", "VOLUME_CONTROL"),
    ("Bring the volume back up", "VOLUME_CONTROL"),
    ("Make the audio louder", "VOLUME_CONTROL"),


    # --------------------------------------------------------
    # SET_TIMER
    # --------------------------------------------------------

    ("Start counting down from half an hour", "SET_TIMER"),
    ("Set a countdown for thirty minutes", "SET_TIMER"),
    ("Start a timer for half an hour", "SET_TIMER"),
    ("I need a countdown of thirty minutes", "SET_TIMER"),
    ("Count down for twenty minutes", "SET_TIMER"),
    ("Start timing twenty minutes from now", "SET_TIMER"),
    ("Give me a fifteen minute countdown", "SET_TIMER"),
    ("Start a timer for half an hour please", "SET_TIMER"),
    ("Begin a thirty minute countdown", "SET_TIMER"),
    ("Time the next forty five minutes", "SET_TIMER"),
]


# ============================================================
# LOAD V2
# ============================================================

print("=" * 70)
print("           JARVIS-X DATASET V3 GENERATION")
print("=" * 70)

v2_df = pd.read_csv(V2_PATH)

print(f"\nV2 examples: {len(v2_df)}")


# ============================================================
# CREATE HARD-EXAMPLE DATAFRAME
# ============================================================

hard_df = pd.DataFrame(
    hard_examples,
    columns=["text", "intent"]
)

print(
    f"Hard examples: {len(hard_df)}"
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

existing_texts = set(
    v2_df["text"].str.lower().str.strip()
)

hard_df["normalized"] = (
    hard_df["text"]
    .str.lower()
    .str.strip()
)

hard_df = hard_df[
    ~hard_df["normalized"].isin(existing_texts)
]

hard_df = hard_df.drop(
    columns=["normalized"]
)


# ============================================================
# COMBINE V2 + HARD EXAMPLES
# ============================================================

v3_df = pd.concat(
    [v2_df, hard_df],
    ignore_index=True
)


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

v3_df = v3_df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

v3_df.to_csv(
    V3_PATH,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n===== DATASET V3 CREATED =====")

print(
    f"Original V2 examples: {len(v2_df)}"
)

print(
    f"New hard examples:    {len(hard_df)}"
)

print(
    f"Final V3 examples:    {len(v3_df)}"
)

print(
    f"Total intents:        {v3_df['intent'].nunique()}"
)

print("\n===== EXAMPLES PER INTENT =====")

print(
    v3_df["intent"].value_counts()
)

print("\n===== DUPLICATES =====")

print(
    v3_df["text"].duplicated().sum()
)

print("\n===== HARD EXAMPLES ADDED =====")

print(
    hard_df.to_string(index=False)
)

print("\n===== FILE SAVED =====")

print(V3_PATH)
