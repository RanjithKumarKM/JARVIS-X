import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/intents_v3.csv"

TRAIN_PATH = "data/train_v3.csv"
VAL_PATH = "data/validation_v3.csv"
TEST_PATH = "data/test_v3.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("           JARVIS-X DATASET V3 PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal examples: {len(df)}")
    print(f"Total intents: {df['intent'].nunique()}")

    # --------------------------------------------------------
    # First split
    # 70% training
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["intent"]
    )

    # --------------------------------------------------------
    # Second split
    # 15% validation
    # 15% testing
    # --------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["intent"]
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    val_df.to_csv(
        VAL_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n===== DATA SPLIT =====")

    print(
        f"Training examples:   {len(train_df)}"
    )

    print(
        f"Validation examples: {len(val_df)}"
    )

    print(
        f"Testing examples:    {len(test_df)}"
    )

    print("\n===== TRAIN DISTRIBUTION =====")

    print(
        train_df["intent"].value_counts()
    )

    print("\n===== VALIDATION DISTRIBUTION =====")

    print(
        val_df["intent"].value_counts()
    )

    print("\n===== TEST DISTRIBUTION =====")

    print(
        test_df["intent"].value_counts()
    )

    print("\n===== FILES SAVED =====")

    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()
