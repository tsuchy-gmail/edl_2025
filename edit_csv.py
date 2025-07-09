import pandas as pd

def edit_csv(typ, region):
    df = pd.read_csv(f"csv/{typ}_data.csv")
    df = df[df["region"]==region]

    filename = f"csv/{typ}_data_{region}.csv"
    df.to_csv(filename, index=False)


typs = ["train", "test"]
regions = ["inside", "outside"]
for typ in typs:
    for region in regions:
        edit_csv(typ, region)
