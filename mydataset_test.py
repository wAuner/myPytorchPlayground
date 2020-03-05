from mydataset import ClassificationDS
from torchvision import transforms

files = "datasets/CatsDogs/train"
testset = ClassificationDS.from_directory(files)

tfms = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])]
)
trainset, valset = ClassificationDS.from_directory(files, 0.25, tfms)

img, label = trainset[0]