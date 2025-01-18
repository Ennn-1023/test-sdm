from torch import utils
import os
from PIL import Image
import numpy as np
from torchvision import transforms

import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, data_dir):
        super().__init__()
        paths = os.listdir(data_dir)
        # with open(image_list, "r") as f:
        #     paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, data_dir):
        super().__init__()
        paths = os.listdir(data_dir)
        # with open(image_list, "r") as f:
        #     paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)




# class CustomDataset(utils.data.Dataset):
    
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         if transform is not None:
#             self.transform = transform 
#         else:
#             self.transform  = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             ])
#         self.data = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
#                             if fname.endswith(".jpg") or fname.endswith(".png")]

#     def __getitem__(self, index):
#         image = self.data[index]
#         image = Image.open(image).resize((512, 512))

#         if not image.mode == "RGB":
#             image = image.convert("RGB")
        
#         image = np.array(image).astype(np.uint8)
#         image = (image/127.5 - 1.0).astype(np.float32) 
#         example = {}
#         example['image'] = image
#         return example

#     def __len__(self):
#         return len(self.data)