import os
import random
import pandas as pd

REACTIVE = "Reactive"
FL = "FL"
INSIDE = "inside"
OUTSIDE = "outside"

def random_sample_fullpath_inside(max_n_patch=500):
    patch_size = 512
    stride = 256
    JMR = "JMR"
    FOLLICLE = "follicle"
    IMG = "img"

    patch_root_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size{patch_size}_stride{stride}/"
    subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]

    sampled_fullpath_list = []
    subtype_list = []
    case_list = []
    patch_count_list = []
    for subtype in subtypes:
        case_dir = os.path.join(patch_root_dir, subtype)
        cases = [case for case in os.listdir(case_dir) if case.startswith(JMR)]
        for case in cases:
            follicle_id_dir = os.path.join(case_dir, case)
            follicle_id_list = [follicle_id for follicle_id in os.listdir(follicle_id_dir) if follicle_id.startswith(FOLLICLE)]
            tmp_fullpath_list = []
            for follicle_id in follicle_id_list:
                patch_dir = os.path.join(follicle_id_dir, follicle_id)
                patch_list = [patch for patch in os.listdir(patch_dir) if patch.startswith(IMG)]
                for patch in patch_list:
                    fullpath = os.path.join(patch_dir, patch)
                    tmp_fullpath_list.append(fullpath)  
            n_patch = min(max_n_patch, len(tmp_fullpath_list))
            sampled_fullpath = random.sample(tmp_fullpath_list, n_patch)

            patch_count_list.extend([n_patch] * n_patch)
            sampled_fullpath_list.extend(sampled_fullpath)
            subtype_list.extend([subtype] * n_patch)
            case_list.extend([case] * n_patch)

    train_data = {"img_path": sampled_fullpath_list, "subtype": subtype_list, "case": case_list, "patch_count": patch_count_list, "region": INSIDE}
    train_data_df = pd.DataFrame(train_data)
    print(train_data_df["subtype"].value_counts())
    train_data_df.to_csv("csv/sampled_patchs_inside_max500.csv", index=False)
    print("created csv inside")

random_sample_fullpath_inside()

def random_sample_fullpath_outside(max_n_patch=400):
    patch_size = 512
    stride = 256
    JMR = "JMR"
    FOLLICLE = "follicle"
    OUTSIDE_FOLLICLES = "outside_follicles_n1000"
    IMG = "img"

    patch_root_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size{patch_size}_stride{stride}/"
    subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]

    sampled_fullpath_list = []
    subtype_list = []
    case_list = []
    patch_count_list = []
    for subtype in subtypes:
        case_dir = os.path.join(patch_root_dir, subtype)
        cases = [case for case in os.listdir(case_dir) if case.startswith(JMR)]
        for case in cases:
            outside_patch_dir = os.path.join(case_dir, case, OUTSIDE_FOLLICLES)
            outside_patch_list = [patch for patch in os.listdir(outside_patch_dir) if patch.startswith(IMG)]
            tmp_fullpath_list = []
            for patch in outside_patch_list:
                fullpath = os.path.join(outside_patch_dir, patch)
                tmp_fullpath_list.append(fullpath)  
            n_patch = min(max_n_patch, len(tmp_fullpath_list))
            sampled_fullpath = random.sample(tmp_fullpath_list, n_patch)
            patch_count_list.extend([n_patch] * n_patch)
            sampled_fullpath_list.extend(sampled_fullpath)
            subtype_list.extend([subtype] * n_patch)
            case_list.extend([case] * n_patch)

    train_data = {"img_path": sampled_fullpath_list, "subtype": subtype_list, "case": case_list, "patch_count": patch_count_list, "region": OUTSIDE}
    train_data_df = pd.DataFrame(train_data)
    print(train_data_df["subtype"].value_counts())
    train_data_df.to_csv("csv/sampled_patchs_outside_max400.csv", index=False)
    print("created csv outside")

random_sample_fullpath_outside()

def subtype_to_patch_count(df, subtype):
    return df[df["subtype"].str.contains(subtype)].groupby("case").first()["patch_count"].sum()

def create_dataset(random_state=42):
    inside_patchs_df = pd.read_csv("csv/sampled_patchs_inside_max500.csv")
    outside_patchs_df = pd.read_csv("csv/sampled_patchs_outside_max400.csv")
    dataset_df = pd.concat([inside_patchs_df, outside_patchs_df])

    test_case_list_Reactive = ["JMR0077", "JMR0299", "JMR2518"] #一旦パッチ数300くらいの症例で適当に選択
    test_case_list_FL = ["JMR0020", "JMR0465", "JMR2499"] 
    test_case_list = test_case_list_Reactive + test_case_list_FL

    train_inside_patchs_df = inside_patchs_df[~inside_patchs_df["case"].isin(test_case_list)]
    test_inside_patchs_df = inside_patchs_df[inside_patchs_df["case"].isin(test_case_list)]

    train_outside_patchs_df = outside_patchs_df[~outside_patchs_df["case"].isin(test_case_list)]
    test_outside_patchs_df = outside_patchs_df[outside_patchs_df["case"].isin(test_case_list)]

    #for train
    train_inside_patch_count_Reactive = subtype_to_patch_count(train_inside_patchs_df, REACTIVE)
    train_inside_patch_count_FL = subtype_to_patch_count(train_inside_patchs_df, FL)

    n_train_patch_by_subtype = min(train_inside_patch_count_Reactive, train_inside_patch_count_FL)

    Reactive_train_inside_patch_df = train_inside_patchs_df[train_inside_patchs_df["subtype"] == REACTIVE].sample(n=n_train_patch_by_subtype, random_state=random_state)
    FL_train_inside_patch_df = train_inside_patchs_df[train_inside_patchs_df["subtype"].str.contains(FL)].sample(n=n_train_patch_by_subtype, random_state=random_state)
    Reactive_train_outside_patch_df = train_outside_patchs_df[train_outside_patchs_df["subtype"] == REACTIVE].sample(n=n_train_patch_by_subtype, random_state=random_state)
    FL_train_outside_patch_df = train_outside_patchs_df[train_outside_patchs_df["subtype"].str.contains(FL)].sample(n=n_train_patch_by_subtype, random_state=random_state)


    #for test
    test_inside_patch_count_Reactive = subtype_to_patch_count(test_inside_patchs_df, REACTIVE)
    test_inside_patch_count_FL = subtype_to_patch_count(test_inside_patchs_df, FL)

    n_test_patch_by_subtype = min(test_inside_patch_count_Reactive, test_inside_patch_count_FL)

    Reactive_test_inside_patch_df = test_inside_patchs_df[test_inside_patchs_df["subtype"] == REACTIVE].sample(n=n_test_patch_by_subtype, random_state=random_state)
    FL_test_inside_patch_df = test_inside_patchs_df[test_inside_patchs_df["subtype"].str.contains(FL)].sample(n=n_test_patch_by_subtype, random_state=random_state)
    Reactive_test_outside_patch_df = test_outside_patchs_df[test_outside_patchs_df["subtype"] == REACTIVE].sample(n=n_test_patch_by_subtype, random_state=random_state)
    FL_test_outside_patch_df = test_outside_patchs_df[test_outside_patchs_df["subtype"].str.contains(FL)].sample(n=n_test_patch_by_subtype, random_state=random_state)

    train_data_df = pd.concat([Reactive_train_inside_patch_df, FL_train_inside_patch_df, Reactive_train_outside_patch_df, FL_train_outside_patch_df])
    test_data_df = pd.concat([Reactive_test_inside_patch_df, FL_test_inside_patch_df, Reactive_test_outside_patch_df, FL_test_outside_patch_df])

    train_data_df.drop(columns=["case", "patch_count"], inplace=True)
    test_data_df.drop(columns=["case", "patch_count"], inplace=True)
    print("n_patch_train", n_train_patch_by_subtype)
    print("n_patch_test", n_test_patch_by_subtype)
    train_data_df.to_csv("csv/train_data.csv", index=False)
    test_data_df.to_csv("csv/test_data.csv", index=False)


create_dataset()

