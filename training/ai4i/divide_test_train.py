import pandas as pd
from sklearn.model_selection import train_test_split

# dividing dataset for train and test

df = pd.read_csv("training/data/raw/ai4i2020.csv")

df = df.drop(["UDI", "Product ID", "Type"], axis=1)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Machine failure"]
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.to_csv("training/data/raw/train.csv", index=False)
test_df.to_csv("training/data/raw/test.csv", index=False)