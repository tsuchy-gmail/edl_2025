import pandas as pd
patch_info_path = "csv/patch_info.csv"

patch_df = pd.read_csv(patch_info_path)
subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]

Reactive_patch_count = sum(patch_df[patch_df["subtype"] == "Reactive"].value_counts())
FL_all_patch_count = sum(patch_df[patch_df["subtype"].str.contains("FL")].value_counts())
FL_all_patch_count = sum(patch_df[patch_df["subtype"] == ("FL/G1")].value_counts())
FL_all_patch_count = sum(patch_df[patch_df["subtype"] == ("FL/G2")].value_counts())
FL_all_patch_count = sum(patch_df[patch_df["subtype"] == ("FL/G3a")].value_counts())

rows = []

def count_by_subtype():
    for subtype in subtypes:
        patch_count_by_subtype = sum(patch_df[patch_df["subtype"] == subtype].value_counts())
        row = {"subtype": subtype, "patch_count": patch_count_by_subtype}
        rows.append(row)

    patch_count_df = pd.DataFrame(rows)
    patch_count_df.to_csv("csv/patch_count_by_subtype.csv", index=False)

def count_by_case():
    count = patch_df.groupby("case").agg(count=("case", "size"), subtype=("subtype", "first"))
    sorted_count = count.sort_values(by="count", ascending=False)
    sorted_count.to_csv("csv/patch_count_by_case.csv")

def count_case():
    case_count = patch_df.groupby("subtype")["case"].nunique()
    case_count.to_csv("csv/case_count_by_subtype.csv")
    print(case_count)

def count_more_200():
    df_by_case = patch_df.groupby("case").agg(count=("case", "size"), subtype=("subtype", "first")).query("count > 200")
    print(df_by_case)


count_case()
     
