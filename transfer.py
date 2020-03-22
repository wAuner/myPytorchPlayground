import torch
from torchvision import models, transforms
from path import Path
from mydataset import ClassificationDS

from myresnet import MyResnet

DATAPATH = Path("datasets/CatsDogs")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                std=imagenet_std)]
            )
 
train_set, val_set = ClassificationDS.from_directory(DATAPATH/"train", 0.2, train_transforms)
# testing
loader = torch.utils.data.DataLoader(train_set)
img, label = next(iter(loader))
model = MyResnet(2, train_set, val_set)
output = model(img)
loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(output, label)

model.freeze()