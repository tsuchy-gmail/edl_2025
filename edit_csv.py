import pandas as pd

train_df = pd.read_csv("csv/train_data.csv")
test_df = pd.read_csv("csv/test_data.csv")

def edit_csv(df, filename):
    df["case"] = df["img_path"].str.extract(r'(JMR\d{4})')
    df.to_csv(filename, index=False)

edit_csv(train_df, "csv/train_data.csv")
edit_csv(test_df, "csv/test_data.csv")
