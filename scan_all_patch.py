import os

def scan_all_patch():
    patch_size = 512
    stride = 256
    JMR = "JMR"
    FOLLICLE = "follicle"
    IMG = "img"

    patch_root_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size{patch_size}_stride{stride}/"
    subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a"]

    rows = []
    for subtype in subtypes:
        case_dir = os.path.join(patch_root_dir, subtype)
        case_list = [case for case in os.listdir(case_dir) if case.startswith(JMR)]
        for case in case_list:
            follicle_id_dir = os.path.join(case_dir, case)
            follicle_id_list = [follicle_id for follicle_id in os.listdir(follicle_id_dir) if follicle_id.startswith(FOLLICLE)]
            for follicle_id in follicle_id_list:
                patch_dir = os.path.join(follicle_id_dir, follicle_id)
                patch_list = [patch for patch in os.listdir(patch_dir) if patch.startswith(IMG)]
                for patch in patch_list:
                    fullpath = os.path.join(patch_dir, patch)
                    print(fullpath)

