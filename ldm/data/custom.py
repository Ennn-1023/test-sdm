from torch import utils
import os
from PIL import Image
import numpy as np
from torchvision import transforms


class CustomDataset(utils.data.Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform 
        else:
            self.transform  = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            ])
        self.data = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.endswith(".jpg") or fname.endswith(".png")]

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.open(image)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image.resize(size=(512, 512))
        image = np.array(image).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
        example = {}
        example['image'] = image
        return example

    def __len__(self):
        return len(self.data)