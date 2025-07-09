import os

root = "/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size512_stride256/"

dirs = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]
all_follicle_count = 0
all_case_count = 0
for tar_dir in dirs:
    case_dir = os.path.join(root, tar_dir)
    for tar_case in os.listdir(case_dir):
        if not tar_case.startswith("JMR"):
            continue
        all_case_count += 1
        fol_dir = os.path.join(case_dir, tar_case)
        follicles = os.listdir(fol_dir)
        for tar_fol in follicles:
            if not tar_fol.startswith("follicle"):
                continue
            fol_path = os.path.join(fol_dir, tar_fol)
            fol = os.listdir(fol_path)
            if len(fol) == 0:
                continue
            all_follicle_count += 1

print("patch", all_follicle_count)
print("wsi", all_case_count)
print("avg_patch", all_follicle_count // all_case_count)

