import torch
from torch.utils.data import Dataset
from torchvision import transforms
from path import Path
from sklearn.model_selection import train_test_split
import os
from PIL import Image

class ClassificationDS(Dataset):
    """
    Create a Dataset for image classification where the filename
    contains the class label and all files are in the same dir.
    assumes one dataset per directory.
    Currently specific to CatsDogs dataset.
    """
    def __init__(self, path, filenames, tfms, class2idx, idx2class, mode):
        """
        Constructor is called by factory method, don't call directly
        """
        if tfms is not None and not isinstance(tfms, transforms.transforms.Compose): 
            raise TypeError("""Expected transforms to be passed in 
        as torchvision.transforms.transforms.Compose""")

        self.path = Path(path)
        self.filenames = filenames
        self.num_data = len(self.filenames)
        self.num_classes = len(class2idx)
        self.transforms = tfms
        self.class2idx = class2idx
        self.idx2class = idx2class
        self.mode = mode
        # check if items will be converted to tensors
        # if so, convert labels to tensor as well
        # currently assumes multiple transforms as Compose object
        self.to_tensor = False
        if tfms:
            for transform in self.transforms.transforms:
                if isinstance(transform, transforms.transforms.ToTensor):
                    self.to_tensor = True
                    break


    @staticmethod
    def _create_label_mapping(filenames):
        """
        Helper function for factory method
        Extract class information from filenames and store mapping as state
        """
        class_names = set([fname.split('.')[0] for fname in filenames])
        class2idx = {}
        idx2class = {}

        for idx, cls_name in enumerate(class_names):
            class2idx[cls_name] = idx
            idx2class[idx] = cls_name

        return class2idx, idx2class


    @classmethod
    def from_directory(cls, ds_path, val_split=0, tfms=None, mode="train"):
        """
        Factory method to create seperate instances for training and validation
        set or a testset
        """
        ds_path = Path(ds_path)
        filenames = os.listdir(ds_path)
        class2idx, idx2class = cls._create_label_mapping(filenames)

        # if the validation set needs to be created from the 
        # train directory
        if val_split > 0:
            train_fnames, val_fnames = train_test_split(filenames, test_size=val_split)
            return cls(ds_path, train_fnames, tfms, class2idx, idx2class, mode="train"), \
                cls(ds_path, val_fnames, tfms, class2idx, idx2class, mode="validation")

        # to create a dataset without splitting (e.g. only train or test)
        return cls(ds_path, filenames, tfms, class2idx, idx2class, mode)

    
    def __getitem__(self, idx):
        """
        Interface for the dataloader. Loads an image an its label
        and returns them as tensors
        """
        filename = self.filenames[idx]
        label = self.class2idx[filename.split('.')[0]]
        img = Image.open(self.path/filename)

        if self.transforms:
            img = self.transforms(img)
            if self.to_tensor:
                # one-hot encoding of the labels is not necessary with nn.CrossEntropyLoss
                label_tensor = torch.tensor(label)
        return img, label_tensor


    def __len__(self):
        return self.num_data
