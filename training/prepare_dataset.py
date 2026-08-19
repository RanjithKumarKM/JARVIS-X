import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH = "data/intents.csv"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    print("===== ORIGINAL DATASET =====")
    print(f"Total examples: {len(df)}")
    print(f"Total intents: {df['intent'].nunique()}")

    return df


def clean_dataset(df):
    # Remove rows with missing values
    df = df.dropna(subset=["text", "intent"])

    # Convert text to lowercase and remove extra spaces
    df["text"] = df["text"].str.lower().str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def analyze_dataset(df):
    print("\n===== CLEAN DATASET =====")
    print(f"Total examples: {len(df)}")
    print(f"Total intents: {df['intent'].nunique()}")

    print("\n===== EXAMPLES PER INTENT =====")
    print(df["intent"].value_counts())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATES =====")
    print(df.duplicated().sum())


def split_dataset(df):
    X = df["text"]
    y = df["intent"]

    # 70% training, 15% validation, 15% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\n===== DATA SPLIT =====")
    print(f"Training examples:   {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Testing examples:    {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test):

    train_df = pd.DataFrame({
        "text": X_train,
        "intent": y_train
    })

    val_df = pd.DataFrame({
        "text": X_val,
        "intent": y_val
    })

    test_df = pd.DataFrame({
        "text": X_test,
        "intent": y_test
    })

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/validation.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\n===== FILES SAVED =====")
    print("data/train.csv")
    print("data/validation.csv")
    print("data/test.csv")


def main():

    df = load_dataset()

    df = clean_dataset(df)

    analyze_dataset(df)

    splits = split_dataset(df)

    save_splits(*splits)


if __name__ == "__main__":
    main()
