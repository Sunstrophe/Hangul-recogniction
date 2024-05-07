import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder


class HGImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = []

        if train:
            img_names = os.listdir(self.img_dir)
            for img_name in img_names:
                if int(img_name.split('_')[1]) not in range(0, 30):
                    self.img_names.append(img_name)
        else:
            img_names = os.listdir(self.img_dir)
            for img_name in img_names:
                if int(img_name.split('_')[1]) in range(0, 30):
                    self.img_names.append(img_name)

        self.label_encoder = LabelEncoder()
        labels = [img_name.split('_')[0] for img_name in self.img_names]
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = Image.open(img_path).convert("L")
        label_str = self.img_names[index].split('_')[0]
        label = self.label_encoder.transform([label_str])[0]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_len_unique_labels(self):
        unique_labels = []
        for img_name in self.img_names:
            label = img_name.split('_')[0]
            if label not in unique_labels:
                unique_labels.append(label)
        return len(unique_labels)


if __name__ == "__main__":
    test_ds = HGImageDataset(ann_file='data/characters.csv',
                             img_dir='data/images/DosGothic/characters', transform=transforms.ToTensor())

    print(test_ds[0])
