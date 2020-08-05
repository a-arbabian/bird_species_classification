from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
import matplotlib.pyplot as plt
from PIL import Image


class BirdsDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        if self.transforms:
            img = self.transforms(img)

        return {'image': img,
                'label': label}




## Split train val randomly with stratification
## CAN"T HAVE SEPARATE TRANSFORMS FOR TRAIN AND VAL
# dataset = BirdsDataset(DATA_DIR, transforms=train_transforms, debug=False)
# train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
# train_dataset = Subset(dataset, train_idx)
# val_dataset = Subset(dataset, val_idx)
# print(len(train_dataset), len(val_dataset))


## Split train val randomly
## CAN"T BE STRATIFIED
## CAN'T HAVE SEPARATE TRANSFORMS FOR TRAIN AND VAL
# VAL_SIZE = 0.2
# train_size = int((1 - VAL_SIZE) * len(dataset))
# val_size = int(len(dataset) - train_size)
# split_lengths = [train_size, val_size]
# train, val = random_split(dataset, split_lengths)
# print(len(train), len(val))


## Check class distributions of train and val sets
# train_classes = []
# val_classes = []
# for i, (t, v) in enumerate(zip(train_dataset, val_dataset)):
#     train_classes.append(t['label'])
#     val_classes.append(v['label'])
#
# fig, ax = plt.subplots()
# ax.hist(train_classes, bins=range(len(dataset.classes)), label='train')
# ax.hist(val_classes, bins=range(len(dataset.classes)), label='val')
#
# plt.show()

