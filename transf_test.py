from matplotlib import pyplot
from torchvision import transforms as T
from PIL import Image
import pandas as pd

def get_img_paths(csv_path):
    df = pd.read_csv(csv_path)
    img_paths = df["img_path"].tolist()
    return img_paths

paths = get_img_paths("csv/T_test_data.csv")
img = Image.open(paths[0])
#resize = T.Resize((224, 224))
crop = T.CenterCrop(14)
#resized = resize(img)

pyplot.imshow(crop(img))
pyplot.savefig("figure/crop14.png")

