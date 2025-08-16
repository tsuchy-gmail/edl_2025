from matplotlib import pyplot
from torchvision import transforms as T
from PIL import Image
import pandas as pd

def get_img_paths(csv_path):
    df = pd.read_csv(csv_path)
    img_paths = df["img_path"].tolist()
    return img_paths

csv_path = f"csv/size896_stride896/train_data.csv"
paths = get_img_paths(csv_path)
img = Image.open(paths[0])

size = 224
size_t = (size, size)
c_crop = T.CenterCrop(size=size_t)
resize = T.Resize(size_t)
cropped = c_crop(img)
resized = resize(cropped)

pyplot.imshow(resized)
pyplot.savefig(f"figure/896to{size}_dpi300.png", dpi=300)

