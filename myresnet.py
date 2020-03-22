import torch
from torchvision import models
from torch import nn
import sys
from mydataset import ClassificationDS


class MyResnet:
    def __init__(self,
                 n_classes,
                 train_set: ClassificationDS,
                 val_set: ClassificationDS = None,
                 architecture: models.ResNet = models.resnet50):
        self.model = architecture(pretrained=True)
        n_input_feature = self.model.fc.in_features
        new_head = nn.Linear(n_input_feature, n_classes)
        self.model.fc = new_head
        self.train_set = train_set
        self.val_set = val_set
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def freeze(self) -> None:
        """freezes all layers but the last one"""
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False

    def __call__(self, arg: torch.Tensor):
        return self.model(arg)

    def train(
            self,
            bs: int = 32,
            lr: float = 1e-3,
            epochs: int = 1
    ) -> None:
        train_dataloader = torch.utils.data.DataLoader(self.train_set,
                                                       batch_size=bs,
                                                       shuffle=True,
                                                       num_workers=8)
        if self.val_set:
            val_dataloader = torch.utils.data.DataLoader(self.val_set,
                                                         batch_size=bs,
                                                         shuffle=False,
                                                         num_workers=8)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # actual training loop
        for epoch in range(epochs):
            print(f"Start training epoch {epoch}...")
            for images, labels in train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                # feed forward pass
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                loss = loss_func(outputs, labels)

                # calculate gradients and optimize
                loss.backward()
                optimizer.step()

                accuracy = (predictions == labels).sum().double().div(len(images))
                sys.stdout.write(f"\rLoss of current batch: {loss.item():.2f}")
                sys.stdout.write(f"\rAccuracy of current batch: {accuracy.item():.2f}")
                sys.stdout.flush()
                
