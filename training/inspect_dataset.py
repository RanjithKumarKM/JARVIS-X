import pandas as pd


DATASET_PATH = "data/intents.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    print("\n===== DATASET INFORMATION =====")
    print(f"Total examples: {len(df)}")
    print(f"Total intents: {df['intent'].nunique()}")

    print("\n===== INTENTS =====")
    print(df["intent"].unique())

    print("\n===== EXAMPLES PER INTENT =====")
    print(df["intent"].value_counts())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    print("\n===== SAMPLE DATA =====")
    print(df.head(10))


if __name__ == "__main__":
    main()
