import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import json
import random
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
from edl_pytorch import Dirichlet, evidential_classification
from loss_func import my_evidential_classification
import csv
import os
from datetime import datetime

from mixed_model_test import model_test

len_Reactive = 2286
REACTIVE = "Reactive"
FL = "FL"
batch_size = 128

lambda_denom = 5

round_size = 5

num_workers = 5

class CustomDataset(Dataset):
    def __init__(self, files_list, transform=None, file_label_map=None):
        self.transform = transform
        Reactive, FL = files_list
        Reactive_inside, Reactive_outside = Reactive
        FL_inside, FL_outside = FL
        self.files = Reactive_inside + Reactive_outside + FL_inside + FL_outside
        self.file_label_map = file_label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        image = Image.open(img_path).convert("RGB")

        label_str = self.file_label_map[img_path]

        label_mapping = {"Reactive": 0, "FL": 1}

        label = label_mapping[label_str]
        """
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        """

        return np.array(image), label, img_path


class CustomDataset2(Dataset):
    def __init__(self, files_list, transform=None, file_label_map=None):
        self.transform = transform
        """
        self.files = Reactive + FL
        """
        # self.file_label_map = file_label_map
        total_len = len(files_list[0][0]) + len(files_list[0][1])+len(files_list[1][0])+len(files_list[1][1])
        self.imgs = np.zeros(
            (total_len, 512, 512, 3), dtype=np.uint8
        )
        self.targets = np.zeros(
            (total_len), dtype=np.uint8
        )
        dataset = CustomDataset(
            files_list=files_list, transform=transform, file_label_map=file_label_map
        )
        self.filename_list = dataset.files
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )

        for i, (img, label, _) in enumerate(data_loader):
            print(i, "/", len(data_loader))
            self.imgs[i * batch_size : i * batch_size + img.shape[0], :, :, :] = img
            self.targets[i * batch_size : i * batch_size + img.shape[0]] = label

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):

        if self.transform:
            image = self.transform(Image.fromarray(self.imgs[idx]))

        label = self.targets[idx]

        return image, label, self.filename_list[idx]


class Random90Rotation:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)

        return transforms.functional.rotate(img, angle)


crop_size = 224 
json_file_inside = "filepath_subtype_cut_table.json"
json_file_outside = "filepath_subtype_cut_table_outside.json"
json_file_mixed = "filepath_subtype_cut_table_mixed.json"
len_train_files = 2000
len_test_files = len_Reactive - len_train_files

def n_zeros(a):
    return str(a).count("0")

def get_file_label_map(json_file):
    file_label_map = {}
    with open(json_file, "r") as f:
        file_label_map = json.load(f)

    return file_label_map

def load_json(json_file):
    with open(json_file, mode="r") as f:
        return json.load(f)


def get_train_test_files(json_file, len_train_files):
    file_label_map = get_file_label_map(json_file)

    Reactive_files = [
        file for file, label in file_label_map.items() if label == "Reactive"
    ]
    FL_files = [file for file, label in file_label_map.items() if label == "FL"]

    permuted_Reactive_files = list(np.random.permutation(Reactive_files))
    permuted_FL_files = list(np.random.permutation(FL_files))

    train_Reactive_files = permuted_Reactive_files[:len_train_files]
    test_Reactive_files = permuted_Reactive_files[len_train_files:]
    train_FL_files = permuted_FL_files[:len_train_files]
    test_FL_files = permuted_FL_files[len_train_files:]

    return {
        "Reactive": {"train": train_Reactive_files, "test": test_Reactive_files},
        "FL": {"train": train_FL_files, "test": test_FL_files},
    }


filename_patchs_count = "patchs_count.json"

    

def get_specified_files(json_file, excluded_cases_Reactive, excluded_cases_FL):
    file_label_map = get_file_label_map(json_file)

    Reactive_files_train = []
    Reactive_files_test = []
    FL_files_train = []
    FL_files_test = []
    for file, label in file_label_map.items():
        if label == REACTIVE:
            if file.split("/")[4] in list(excluded_cases_Reactive):
                Reactive_files_test.append(file)
            else:
                Reactive_files_train.append(file)
        elif label == FL:
            if file.split("/")[4] in list(excluded_cases_FL):
                FL_files_test.append(file)
            else:
                FL_files_train.append(file)
        else:
            print("想定外のラベル")
            exit(0)

    len_test = get_len_cases(excluded_cases_Reactive)
    len_train = len_Reactive - len_test

    print("len_test", len_test)
    print("len_train", len_train)

    permuted_Reactive_files_train = list(np.random.permutation(Reactive_files_train))[:len_train]
    permuted_Reactive_files_test = list(np.random.permutation(Reactive_files_test))[:len_test]
    permuted_FL_files_train = list(np.random.permutation(FL_files_train))[:len_train]
    permuted_FL_files_test = list(np.random.permutation(FL_files_test))[:len_test]

    return {
        "Reactive": {"train": permuted_Reactive_files_train, "test": permuted_Reactive_files_test},
        "FL": {"train": permuted_FL_files_train, "test": permuted_FL_files_test},
    }


def get_len_cases(excluded_cases):
    patchs_count_table = load_json(filename_patchs_count)
    sum_patchs = 0
    for case in excluded_cases:
        sum_patchs += patchs_count_table[REACTIVE][case] #Reactiveの数に合わせているから変数じゃなくていい

    return sum_patchs


def dump_train_test_files(train_test_files, results_dirname, side):

    os.makedirs(f"{results_dirname}/{side}", exist_ok=True)
    with open(f"{results_dirname}/{side}/train_files_Reactive.json", mode="w") as f:
        json.dump(train_test_files["Reactive"]["train"], f, indent=4)
    with open(f"{results_dirname}/{side}/test_files_Reactive.json", mode="w") as f:
        json.dump(train_test_files["Reactive"]["test"], f, indent=4)
    with open(f"{results_dirname}/{side}/train_files_FL.json", mode="w") as f:
        json.dump(train_test_files["FL"]["train"], f, indent=4)
    with open(f"{results_dirname}/{side}/test_files_FL.json", mode="w") as f:
        json.dump(train_test_files["FL"]["test"], f, indent=4)

def set_transforms():
    return transforms.Compose(
        [
            Random90Rotation(),
            #transforms.ColorJitter(
            #    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            #),
            transforms.RandomCrop(size=(crop_size, crop_size)),
            #transforms.RandomApply(
            #    [transforms.GaussianBlur(kernel_size=5, sigma=(1, 2))], p=0.2
            #),
            
            #transforms.CenterCrop(size=(crop_size, crop_size)),
            transforms.ToTensor(),
        ]
    )


def print_by_epoch(epoch, train_loss, test_loss, acc, lam):
    print(
        "epoch:",
        epoch,
        ",",
        "train_loss:",
        train_loss,
        ",",
        "test_loss:",
        test_loss,
        ",",
        "accuracy:",
        acc,
        ",",
        "lambda:",
        lam
    )


def print_to_file(results_dirname, epoch, train_loss, test_loss, acc, lam):
    with open(f"{results_dirname}/result.txt", mode="a") as f:
        print(
            "epoch:",
            epoch,
            ",",
            "train_loss:",
            train_loss,
            ",",
            "test_loss:",
            test_loss,
            ",",
            "accuracy:",
            acc,
            ",",
            "lambda:",
            lam,
            file=f,
        )

def get_dumped_files_inout(in_json, out_json):
    return [load_json(in_json), load_json(out_json)]

def get_dumped_files(re_in, re_out, fl_in, fl_out):
    re = get_dumped_files_inout(re_in, re_out)
    fl = get_dumped_files_inout(fl_in, fl_out)

    return [re, fl]

def get_dumpfile_path(specified_root, date, lam, case):
    lams_path = f"{specified_root}/{date}"
    lambda_list = os.listdir(lams_path)

    for lamb in lambda_list:
        if lamb == f"lambda_denom{lam}":
            models_path = f"{lams_path}/{lamb}"
            models = os.listdir(models_path)

            for model in models:
                if case in model:
                    print(f"{models_path}/{model}")
                    return f"{models_path}/{model}"

def get_dumpfile_fullpaths_inout(specified_root, date, lam, case, tr_or_te, subtype):
    model_path = get_dumpfile_path(specified_root, date, lam, case)

    def get_fullpath(in_out):
       return f"{model_path}/{tr_or_te}_files/{subtype}/{in_out}.json" 

    fullpath_in = get_fullpath("inside")
    fullpath_out = get_fullpath("outside")

    return [fullpath_in, fullpath_out]
def dump_inout(lists, results_dirname, subtype, tr_or_te):
    in_list, out_list = lists

    save_dir_in = f"{results_dirname}/{tr_or_te}_files/{subtype}"
    save_dir_out = f"{results_dirname}/{tr_or_te}_files/{subtype}"

    os.makedirs(save_dir_in, exist_ok=True)
    os.makedirs(save_dir_out, exist_ok=True)

    with open(f"{save_dir_in}/inside.json", mode="w") as f:
        json.dump(in_list, f, indent=4)

    with open(f"{save_dir_out}/outside.json", mode="w") as f:
        json.dump(out_list, f, indent=4)

def dump(files_list, results_dirname, tr_or_te):
    re_inout, fl_inout = files_list

    dump_inout(re_inout, results_dirname, REACTIVE, tr_or_te)
    dump_inout(fl_inout, results_dirname, FL, tr_or_te)

def set_seed(seed):
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True

def train(excluded_cases_Reactive, excluded_cases_FL, cuda, transforms, epochs, specified_date, specified_lam, lam_th, lam_start, lam_end, crop_size, lr, dir_suffix):
    #torch.manual_seed(0)
    set_seed(0)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    
    '''
    train_test_files_inside = get_specified_files(json_file_inside, excluded_cases_Reactive, excluded_cases_FL)

    train_files_Reactive_inside = train_test_files_inside["Reactive"]["train"]
    test_files_Reactive_inside = train_test_files_inside["Reactive"]["test"]
    train_files_FL_inside = train_test_files_inside["FL"]["train"]
    test_files_FL_inside = train_test_files_inside["FL"]["test"]

    train_test_files_outside = get_specified_files(json_file_outside, excluded_cases_Reactive, excluded_cases_FL)
    train_files_Reactive_outside = train_test_files_outside["Reactive"]["train"]
    test_files_Reactive_outside = train_test_files_outside["Reactive"]["test"]
    train_files_FL_outside = train_test_files_outside["FL"]["train"]
    test_files_FL_outside = train_test_files_outside["FL"]["test"]
    '''

    date = datetime.now().date()
    #results_dirname = f"results/mixed/{date}/lambda_denom{lambda_denom}/excluded_Re-{'-'.join(excluded_cases_Reactive)}_FL-{'-'.join(excluded_cases_FL)}"
    results_dirname = f"results/mixed/{date}/lam_th{lam_th}-lam_start{lam_start}-lam_end{lam_end}_crop{crop_size}_lr{n_zeros(lr)}_epoch{epochs}{dir_suffix}/excluded_Re-{'-'.join(excluded_cases_Reactive)}_FL-{'-'.join(excluded_cases_FL)}"
    os.makedirs(results_dirname, exist_ok=True)
    print(f"results_dirname: {results_dirname}")

    '''
    dump_train_test_files(train_test_files_inside, results_dirname, "inside")
    dump_train_test_files(train_test_files_outside, results_dirname, "outside")

    train_files_list = [[train_files_Reactive_inside, train_files_Reactive_outside], [train_files_FL_inside, train_files_FL_outside]]
    test_files_list = [[test_files_Reactive_inside, test_files_Reactive_outside], [test_files_FL_inside, test_files_FL_outside]]
    '''

    excluded_case_re = excluded_cases_Reactive[0]
    specified_root_dir = f"results/mixed"
    paths_train_files_re = get_dumpfile_fullpaths_inout(specified_root_dir, specified_date, specified_lam, excluded_case_re, "train", "Reactive")
    paths_train_files_fl = get_dumpfile_fullpaths_inout(specified_root_dir, specified_date, specified_lam, excluded_case_re, "train", "FL")

    paths_test_files_re = get_dumpfile_fullpaths_inout(specified_root_dir, specified_date, specified_lam, excluded_case_re, "test", "Reactive")
    paths_test_files_fl = get_dumpfile_fullpaths_inout(specified_root_dir, specified_date, specified_lam, excluded_case_re, "test", "FL")

    train_files_list = get_dumped_files(*paths_train_files_re, *paths_train_files_fl)
    test_files_list = get_dumped_files(*paths_test_files_re, *paths_test_files_fl)

    dump(train_files_list, results_dirname, "train")
    dump(test_files_list, results_dirname, "test")
    
    file_label_map_mixed = get_file_label_map(json_file_mixed)

    train_ds = CustomDataset2(
        files_list=train_files_list,
        transform=transforms,
        file_label_map=file_label_map_mixed,
    )

    test_ds = CustomDataset2(
        files_list=test_files_list,
        transform=transforms,
        file_label_map=file_label_map_mixed,
    )

    resnet = models.resnet18(pretrained=False)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 64),
        nn.ReLU(),
        Dirichlet(64, 2),
    )
    resnet.to(device)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    print("training start")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    out_list = [["epoch", "train_loss", "test_loss", "acc", "lambda"]]
    D_train = {}
    D_test = {}
    min_trainloss = float("inf")
    min_testloss = float("inf")
    max_acc = -1
    for epoch in range(1, epochs + 1):
        resnet.train()
        train_loss = 0
        test_loss = 0
        lam = min((epoch - 1) / lambda_denom, 1)
        #lam = 0 if epoch <= lam_th else 1
        #lam = lam_start if epoch <= lam_th else lam_end
        #lam = 1
        #lam = max(lam_start - (epoch-1) * ((lam_start - 1) / lam_th), lam_end)
        #lam = lam_start - (lam_start / epochs) * epoch
        for index, (x, y, img_path_list) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device).to(torch.long)
            with torch.cuda.amp.autocast():
                pred = resnet(x)

            loss = evidential_classification(pred, y, lamb=lam)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            L_list, KL_list = my_evidential_classification(pred, y)
            se_list, var_list = L_list
            #monitoring
            for i in range(x.shape[0]):
                img_path = img_path_list[i]
                se = round(se_list[i].item(), round_size)
                var = round(var_list[i].item(), round_size)
                kl = round(KL_list[i].item(), round_size)

                loss_sum = round((se+ var) + lam * kl, round_size)

                if img_path in D_train:
                    D_train[img_path]["se"].append(se)
                    D_train[img_path]["var"].append(var)
                    D_train[img_path]["se+var"].append(se + var)
                    D_train[img_path]["kl"].append(kl)
                    D_train[img_path]["loss_sum"].append(loss_sum)
                else:
                    D_train[img_path] = {"se": [se], "var": [var], "se+var": [se + var], "kl": [kl], "loss_sum": [loss_sum]}


        with torch.no_grad():
            resnet.eval()
            correct, total = 0, 0

            for index, (x, y, img_path_list) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device).to(torch.long)

                pred = resnet(x)

                loss = evidential_classification(pred, y, lamb=lam)

                L_list, KL_list = my_evidential_classification(pred, y)
                se_list, var_list = L_list

                test_loss += loss.item()
                evidence = torch.relu(pred)

                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                belief = evidence / S

                uncertainty = 2 / S

                correct += (pred.argmax(-1) == y).sum()
                total += y.shape[0]

                L_list, KL_list = my_evidential_classification(pred, y)
                se_list, var_list = L_list
                #monitoring
                for i in range(x.shape[0]):
                    img_path = img_path_list[i]
                    se = round(se_list[i].item(), round_size)
                    var = round(var_list[i].item(), round_size)
                    kl = round(KL_list[i].item(), round_size)

                    loss_sum = round((se+ var) + lam * kl, round_size)

                    if img_path in D_test:
                        D_test[img_path]["se"].append(se)
                        D_test[img_path]["var"].append(var)
                        D_test[img_path]["se+var"].append(se + var)
                        D_test[img_path]["kl"].append(kl)
                        D_test[img_path]["loss_sum"].append(loss_sum)
                    else:
                        D_test[img_path] = {"se": [se], "var": [var], "se+var": [se + var], "kl": [kl], "loss_sum": [loss_sum]}


            acc = (correct / total).item()
            out = [epoch, train_loss, test_loss, acc, lam]
            out_list.append(out)
            print_by_epoch(epoch, train_loss, test_loss, acc, lam)
            print_to_file(results_dirname, epoch, train_loss, test_loss, acc, lam)

            if train_loss < min_trainloss:
                min_trainloss = train_loss
                torch.save(
                    resnet.state_dict(), f"{results_dirname}/saved_model_min_trainloss.pth"
                )

            if test_loss < min_testloss:
                min_testloss = test_loss
                torch.save(
                    resnet.state_dict(), f"{results_dirname}/saved_model_min_testloss.pth"
                )

            if acc > max_acc:
                max_acc = acc
                torch.save(
                    resnet.state_dict(), f"{results_dirname}/saved_model_max_acc.pth"
                )



        if epoch % 100 == 0:
            torch.save(
                resnet.state_dict(), f"{results_dirname}/saved_model{epoch}.pth"
            )
            with open(f"{results_dirname}/result.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(out_list)
            with open(f"{results_dirname}/monitor_loss_train.json", mode="w") as file:
                json.dump(D_train, file, indent=4)
            with open(f"{results_dirname}/monitor_loss_test.json", mode="w") as file:
                json.dump(D_test, file, indent=4)
        
        if epoch == epochs:
            model_test(results_dirname)
        


    '''
    with open(f"{results_dirname}/result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(out_list)

    with open(f"{results_dirname}/monitor_loss_train.json", mode="w") as file:
        json.dump(D_train, file, indent=4)
    with open(f"{results_dirname}/monitor_loss_test.json", mode="w") as file:
        json.dump(D_test, file, indent=4)
    '''




"""
with open(f"{results_dirname}/result_detail_{epochs}epochs.txt", mode="a") as f:
    for i in range(x.shape[0]):
        if uncertainty[i].item() > 0.3 or uncertainty[i].item() < 0.1:
            print("Image:", img_path[i], file=f)
            print(f"True Label: {y[i].item()}", file=f)
            print(f"Uncertainty: {uncertainty[i].item()}", file=f)
            print(f"Belief: {belief[i].tolist()}", file=f)
            print(f"Evidence: {evidence[i].tolist()}", file=f)
            print(file=f)
"""

def main(pair_idx, cuda, epochs, specified_date, specified_lam, lam_th, lam_start, lam_end, crop_size, lr, dir_suffix):
    print(pair_idx, cuda, epochs, specified_date, specified_lam, lam_th, lam_start, lam_end, crop_size, lr)
    excluded_cases_Reactive_list = ["JMR0007", "JMR0045", "JMR0057", "JMR0077", "JMR0089", "JMR0090", "JMR1022", "JMR1050", "JMR1077"]
    excluded_cases_FL_list = ["JMR0011", "JMR0020", "JMR0022", "JMR0025", "JMR0068", "JMR0054", "JMR0072", "JMR2515", "JMR0206"] 

#pair_idx: 0~8

    case_Reactive = excluded_cases_Reactive_list[pair_idx]
    case_FL = excluded_cases_FL_list[pair_idx]

    train([case_Reactive], [case_FL], cuda=cuda, transforms=set_transforms(), epochs=epochs, specified_date=specified_date, specified_lam=specified_lam, lam_th=lam_th, lam_start=lam_start, lam_end=lam_end, crop_size=crop_size, lr=lr, dir_suffix=dir_suffix)



main(pair_idx=6, cuda=0, epochs=50, specified_date="2024-11-24", specified_lam=500, lam_th=5, lam_start=0, lam_end=1, crop_size=crop_size, lr=1e-4, dir_suffix="Apply-ColorJitter-Off")

#メモ
    # excluded_files_Reactive = ["JMR0007", "JMR0045", "JMR0057", "JMR0090", "JMR0117"]
    # excluded_files_FL = ["JMR0022", "JMR0223", "JMR2506"] #1回目

    # excluded_files_Reactive = ["JMR1077", "JMR0057"]
    # excluded_files_FL = ["JMR0020", "JMR0025", "JMR0011", "JMR1018"]  #2回目

    #excluded_cases_Reactive = ["JMR0077"]
    #excluded_cases_FL = ["JMR0206", "JMR1031", "JMR2551", "JMR0068", "JMR2515"]  # 3回目
