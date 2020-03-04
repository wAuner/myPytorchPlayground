from mydataset import ClassificationDS

files = "datasets/CatsDogs/train"
testset = ClassificationDS.from_directory(files)

trainset, valset = ClassificationDS.from_directory(files, 0.25)

img, label = trainset[0]