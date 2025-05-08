from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import pandas as pd
import torch

from pandas import read_csv
from datetime import datetime
from edl_pytorch import Dirichlet, evidential_classification
import os
from my_loss_function import my_evidential_classification
import random

REACTIVE = "Reactive"
FL = "FL"
INSIDE = "inside"
OUTSIDE = "outside"
batch_size = 128
n_classes = 2
learning_rate = 1e-4
epochs = 100
num_workers = 8

def encode_subtype(subtype):
    if subtype == REACTIVE:
        return 0
    elif FL in subtype:
        return 1
    else:
        raise ValueError("subtypeがReactiveでもFLでもない")

class CustomDataset(Dataset):
    def __init__(self, img_path_list, subtype_list, region_list, transform=None):
        self.img_path_list = img_path_list
        self.subtype_list = subtype_list
        self.region_list = region_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert("RGB")

        subtype = self.subtype_list[idx]
        label = encode_subtype(subtype)

        region = self.region_list[idx]

        if self.transform:
            img = self.transform(img)
            
        return img, label, region


def test_CustomDataset():
    train_data_df = read_csv("csv/train_data.csv")
    img_path_list = train_data_df["img_path"].tolist()
    subtype_list = train_data_df["subtype"].tolist()

    transform = transforms.ToTensor()
    dataset = CustomDataset(img_path_list, subtype_list, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


    for img_batch, label_batch in dataloader:
        print(img_batch.shape, label_batch)
        exit(0)


#test_CustomDataset()

def get_list_data(csv_path):
    data_df = read_csv(csv_path)

    img_path_list = data_df["img_path"].tolist()
    subtype_list = data_df["subtype"].tolist()
    region_list = data_df["region"].tolist()

    return img_path_list, subtype_list, region_list


class RandomRotation90:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)

        return transforms.functional.rotate(img, angle)

def get_transforms():
    crop_size = (224, 224)

    return transforms.Compose(
            [
                RandomRotation90(),
                transforms.CenterCrop(size=crop_size),
                transforms.ToTensor(),
            ]
    )

def train(cuda, transforms, save_dir):
    print("taining start")
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
    result_dir_path = f"result/{save_dir}/{timestamp}/"
    os.makedirs(result_dir_path, exist_ok=True)


    device = torch.device(f"cuda:{cuda}")  

    train_img_path_list, train_subtype_list, train_region_list = get_list_data("csv/train_data.csv")
    test_img_path_list, test_subtype_list, test_region_list = get_list_data("csv/test_data.csv")

    train_ds = CustomDataset(train_img_path_list, train_subtype_list, train_region_list, transforms)
    test_ds = CustomDataset(test_img_path_list, test_subtype_list, test_region_list, transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    print("set loader")

    model = models.resnet18(pretrained=False)
    n_features = model.fc.in_features

    model.fc = Dirichlet(n_features, n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #for csv
    loss_list_train = []
    loss_inside_list_train = []
    loss_outside_list_train = []
    mse_inside_list_train = []
    mse_outside_list_train = []
    kl_inside_list_train = []
    kl_outside_list_train = []

    loss_list_test = []
    loss_inside_list_test = []
    loss_outside_list_test = []
    mse_inside_list_test = []
    mse_outside_list_test = []
    kl_inside_list_test = []
    kl_outside_list_test = []
    #

    for epoch in range(epochs):
        print(f"epoch{epoch + 1}")
        model.train()


        loss_total = 0.0

        loss_inside = 0.0
        mse_inside_total = 0.0
        kl_inside_total = 0.0

        loss_outside = 0.0
        mse_outside_total = 0.0
        kl_outside_total = 0.0

        lamb = 5 - ((epoch + 1) * 0.04)

        for img_batch, label_batch, region_batch in train_loader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            alpha_batch = model(img_batch).to(device)
            loss = evidential_classification(alpha_batch, label_batch, lamb)
            loss_total += loss.item() * img_batch.size(0)
            
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            print("every patch")
            for i in range(img_batch.size(0)):
                mse = mse_batch[i].item()
                kl = kl_batch[i].item()
                region = region_batch[i]

                if region == INSIDE:
                    mse_inside_total += mse
                    kl_inside_total += kl
                    loss_inside += mse + lamb*kl
                elif region == OUTSIDE:
                    mse_outside_total += mse
                    kl_outside_total += kl
                    loss_outside += mse + lamb*kl
                else:
                    raise ValueError("regionに想定外の値")
            print("every patch done")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("step")
        
        print("loss", loss_total)
        print("my_loss", loss_inside + loss_outside)
        n_data = len(train_loader.dataset)
        avg_loss = loss_total / n_data
        avg_loss_inside = loss_inside / n_data
        avg_loss_outside = loss_outside / n_data

        avg_mse_inside = mse_inside_total / n_data
        avg_mse_outside = mse_outside_total / n_data
        
        avg_kl_inside = kl_inside_total / n_data
        avg_kl_outside = kl_outside_total / n_data

        loss_list_train.extend([avg_loss])
        loss_inside_list_train.extend([avg_loss_inside])
        loss_outside_list_train.extend([avg_loss_outside])
        mse_inside_list_train.extend([avg_mse_inside])
        mse_outside_list_train.extend([avg_mse_outside])
        kl_inside_list_train.extend([avg_kl_inside])
        kl_outside_list_train.extend([avg_kl_outside])

        loss_data_for_csv = {
                "epoch": epoch + 1,
                "loss": loss_list_train,
                "loss_inside": loss_inside_list_train,
                "loss_outside": loss_outside_list_train,
                "mse_inside": mse_inside_list_train,
                "mse_outside": mse_outside_list_train,
                "kl_inside": kl_inside_list_train,
                "kl_outside": kl_outside_list_train,
                }
        loss_df = pd.DataFrame(loss_data_for_csv)
        csv_path = os.path.join(result_dir_path, "train_loss_status")
        loss_df.to_csv(csv_path, index=False)
        print("created csv")

        loss_total = 0.0

        loss_inside = 0.0
        mse_inside_total = 0.0
        kl_inside_total = 0.0

        loss_outside = 0.0
        mse_outside_total = 0.0
        kl_outside_total = 0.0

        with torch.no_grad():
            model.eval()
            correct = 0
            correct_inside = 0
            correct_outside = 0
            total = 0

            for img_batch, label_batch, region_batch in test_loader:
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
                alpha_batch = model(img_batch).to(device)
                loss = evidential_classification(alpha_batch, label_batch, lamb)
                loss_total += loss.item() * img_batch.size(0)

                mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

                is_correct = alpha_batch.argmax(-1) == label_batch
                is_inside = torch.tensor([regn == INSIDE for regn in region_batch], device=alpha_batch.device).to(device)
                is_outside = torch.tensor([regn == OUTSIDE for regn in region_batch], device=alpha_batch.device).to(device)


                #correct = is_correct.sum().item()
                #correct_inside = (is_correct & is_inside).sum().item()
                #correct_outside = (is_correct & is_outside).sum().item()
                #total += img_batch.size(0)
                #print("correct", correct)
                #print("correct_inside", correct_inside)
                #print("correct_outside", correct_outside)

                print("every patch")
                mse_inside_total = mse_batch[is_inside].sum().item()
                kl_inside_total = kl_batch[is_inside].sum().item()
                for i in range(img_batch.size(0)):
                    mse = mse_batch[i].item()
                    kl = kl_batch[i].item()
                    region = region_batch[i]

                    if region == INSIDE:
                        mse_inside_total += mse
                        kl_inside_total += kl
                        loss_inside += mse + lamb*kl
                    elif region == OUTSIDE:
                        mse_outside_total += mse
                        kl_outside_total += kl
                        loss_outside += mse + lamb*kl
                    else:
                        raise ValueError("regionに想定外の値")

        n_data = len(test_loader.dataset)
        avg_loss = loss_total / n_data
        avg_loss_inside = loss_inside / n_data
        avg_loss_outside = loss_outside / n_data

        avg_mse_inside = mse_inside_total / n_data
        avg_mse_outside = mse_outside_total / n_data
        
        avg_kl_inside = kl_inside_total / n_data
        avg_kl_outside = kl_outside_total / n_data

        loss_list_test.extend([avg_loss])
        loss_inside_list_test.extend([avg_loss_inside])
        loss_outside_list_test.extend([avg_loss_outside])
        mse_inside_list_test.extend([avg_mse_inside])
        mse_outside_list_test.extend([avg_mse_outside])
        kl_inside_list_test.extend([avg_kl_inside])
        kl_outside_list_test.extend([avg_kl_outside])

        loss_data_for_csv = {
                "epoch": epoch + 1,
                "loss": loss_list_test,
                "loss_inside": loss_inside_list_test,
                "loss_outside": loss_outside_list_test,
                "mse_inside": mse_inside_list_test,
                "mse_outside": mse_outside_list_test,
                "kl_inside": kl_inside_list_test,
                "kl_outside": kl_outside_list_test,
                }

        loss_df = pd.DataFrame(loss_data_for_csv)
        csv_path = os.path.join(result_dir_path, "test_loss_status")
        loss_df.to_csv(csv_path, index=False)
        print("created csv")

        

train(3, get_transforms(), "test")
            

