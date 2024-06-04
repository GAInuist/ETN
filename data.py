import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


def get_loader(args, ROOT, files, labels_based0, transform, is_sample=False, count_labels=None):
    data = DataLoader(ROOT, files, labels_based0, transform=transform)
    if is_sample:
        weights_ = 1. / count_labels
        weights = weights_[labels_based0]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=labels_based0.shape[0],
                                                               replacement=True)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=train_sampler,
                                                  num_workers=args.num_workers)
        return data_loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch, shuffle=False,
                                              num_workers=args.num_workers)
    return data_loader
