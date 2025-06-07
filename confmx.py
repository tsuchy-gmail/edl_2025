def confmx(alpha_dict, save_root_dir, region):
    subtypes = ["R", "F"]
    row_R = []
    row_F = []
    n_all = 0
    correct = 0
    for subtype in subtypes:
        alpha = alpha_dict[f"{subtype}_{region}"]
        n_all += alpha.shape[0]
        R_count = np.sum(alpha[:, 0] > alpha[:, 1])
        F_count = np.sum(alpha[:, 0] < alpha[:, 1])

        print("R_count+F_count", R_count+F_count)
        print("shape[0]", alpha.shape[0])
        
        if subtype == "R":
            row_R.extend([R_count, F_count])
            correct += R_count
        elif subtype == "F":
            row_F.extend([R_count, F_count])
            correct += F_count
        else:
            raise ValueError("subtypeに予期せぬ値")
    
    col_R, col_F = map(list, zip(row_R, row_F))
    confmx_dict = {"R": col_R, "F": col_F}
    confmx_save_path = os.path.join(save_root_dir, f"confmx_{region}.csv")
    df = pd.DataFrame(confmx_dict, index=subtypes)
    df.to_csv(confmx_save_path, index=True, index_label="true")

    acc_save_path = os.path.join(save_root_dir, f"acc_{region}.txt")
    acc = correct / n_all
    with open(acc_save_path, "w") as f:
        f.write(f"accuracy:{acc}")
