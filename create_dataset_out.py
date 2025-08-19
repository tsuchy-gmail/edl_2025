import os
import pandas as pd

patch_size = 896
stride = 896
data_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size{patch_size}_stride{stride}/" 
subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]
csv_path = f"csv/size{patch_size}_stride{stride}/img_path_outside.csv"


def create_path_csv():
    img_path_list = []
    case_list = []
    subtype_list = []
    region_list = []

    for subtype in subtypes:
        case_dir = os.path.join(data_dir, subtype)
        cases = os.listdir(case_dir)
        for case in cases:
            if not case.startswith("JMR"):
                continue
            
            fol_dir = os.path.join(case_dir, case)
            follicles = os.listdir(fol_dir)
            for follicle in follicles:
                if not follicle.startswith("outside_follicles"):
                    continue

                patch_dir = os.path.join(fol_dir, follicle)
                patchs = os.listdir(patch_dir)
                for patch in patchs:
                    if not patch.startswith("img"):
                        continue
                    
                    img_path = os.path.join(patch_dir, patch)
                    img_path_list.append(img_path) 
                    case_list.append(case)
                    subtype_list.append(subtype)
                    region_list.append("outside")

    csv_data = {
            "img_path": img_path_list,
            "case": case_list,
            "subtype": subtype_list,
            "region": region_list,
            }

    pd.DataFrame(csv_data).to_csv(csv_path, index=False)


def create_train_data(df):
    df_F = df[df["subtype"].str.contains("FL")]
    df_R = df[df["subtype"] == "Reactive"]
    n_min = min(len(df_F), len(df_R))

    print("n_min", n_min)
    seed = 42
    df_F_sampled = df_F.sample(n=n_min, random_state=seed)
    df_R_sampled = df_R.sample(n=n_min, random_state=seed)

    df_cat = pd.concat([df_F_sampled, df_R_sampled])
    save_dir = os.path.dirname(csv_path)
    save_path = os.path.join(save_dir, "train_data_outside.csv")
    df_cat.to_csv(save_path, index=False)

def create_test_data(df):
    save_dir = os.path.dirname(csv_path)
    save_path = os.path.join(save_dir, "test_data_outside.csv")
    df.to_csv(save_path, index=False)


def separate_train_test():
    df = pd.read_csv(csv_path)
    test_cases_R = ["JMR1364", "JMR2302", "JMR2205"]
    test_cases_F = ["JMR0398", "JMR0025", "JMR0011"] #G1, G2, G3a

    test_cases = test_cases_R + test_cases_F

    is_test = df["case"].isin(test_cases)
    test_df = df[is_test]
    train_df = df[~is_test]

    return train_df, test_df


train_df, test_df = separate_train_test()
create_train_data(train_df)
create_test_data(test_df)
