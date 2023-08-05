import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B,'A_paths': self.files_A[index % len(self.files_A)], 'B_paths': self.files_B[index % len(self.files_B)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset_pix2pix(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        if mode == "train":
            self.files_A = sorted(glob.glob(os.path.join(root, "train/A") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "train/B") + "/*.*"))
        elif mode =='test':
            self.files_A = sorted(glob.glob(os.path.join(root, "test/A") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "test/B") + "/*.*"))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

        # self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        # self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        # 确保输入文件夹和目标文件夹中图像文件数量一致
        assert len(self.files_A) == len(self.files_B), "Number of input and target images doesn't match."

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index]) # 从文件夹中读取图像

        image_B = Image.open(self.files_B[index])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

