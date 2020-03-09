from mydataset import ClassificationDS
from torchvision import transforms
import torch
import os
import unittest
from PIL import Image
import PIL


class TestClassificationDS(unittest.TestCase):
    def setUp(self):
        self.train_path = "datasets/CatsDogs/train"
        self.test_path = "datasets/CatsDogs/test"
        self.train_len = len(os.listdir(self.train_path))
        self.val_split = 0.2

        self.testset_pure = ClassificationDS.from_directory(self.test_path, mode="test")

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_tfms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                std=imagenet_std)]
        )

        self.trainset, self.valset = ClassificationDS.from_directory(self.train_path, self.val_split, train_tfms)

        inf_tfms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=imagenet_mean,
                                            std=imagenet_std)
                    ])

        self.testset_inf = ClassificationDS.from_directory(self.test_path, tfms=inf_tfms, mode="test")

    def tearDown(self):
        pass


    def test_len(self):
        self.assertEqual(len(self.testset_pure), 12500)
        self.assertAlmostEqual(len(self.valset), self.val_split * self.train_len, delta=1)


    def test_get_item(self):
        # test the non-tensor version
        pillow_image, int_label = self.testset_pure[0]
        self.assertIsInstance(pillow_image, PIL.JpegImagePlugin.JpegImageFile)
        self.assertIsInstance(int_label, int)

        # test tensor version
        tensor_img, tensor_lbl = self.trainset[0]
        # type test
        self.assertIsInstance(tensor_img, torch.Tensor)
        self.assertIsInstance(tensor_lbl, torch.Tensor)
        # resize test
        self.assertEqual(tensor_img.size()[1], 224)
        self.assertEqual(tensor_img.size()[2], 224)


    def test_init(self):
        # test different transform types
        tfm = transforms.transforms.ToTensor()
        # test if a TypeError is raised if the transform is not a compose object
        with self.assertRaises(TypeError):
            a = ClassificationDS.from_directory(self.train_path, tfms=tfm)



if __name__ == "__main__":
    unittest.main()
