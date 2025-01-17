from torch import utils
import os

class CustomDataset(utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.endswith(".jpg") or fname.endswith(".png")]

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)