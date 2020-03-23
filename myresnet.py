import torch
from torchvision import models
from torch import nn
import sys
from typing import Union
from path import Path
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
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def freeze(self) -> None:
        """freezes all layers but the last one"""
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False

    def __call__(self, arg: torch.Tensor) -> torch.Tensor:
        return self.model(arg)

    def train(self, bs: int = 32, lr: float = 1e-3, epochs: int = 1) -> None:
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
        print(f"Training on {self.device.type}")
        for epoch in range(epochs):
            print(f"Start training epoch {epoch}...")
            # train metrics per epoch
            correct_predictions = 0.
            total_predictions = 0.
            cum_train_loss = 0.
            for images, labels in train_dataloader:
                # move batch to gpu if necessary
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                # feed forward pass
                outputs = self.model(images)                
                loss = loss_func(outputs, labels)
                cum_train_loss += loss.data.item()

                # calculate gradients and optimize
                loss.backward()
                optimizer.step()

                # calculate performance metrics
                with torch.set_grad_enabled(False):
                    _, predictions = torch.max(outputs, 1)
                    correct_batch = (predictions == labels).sum()
                    total_batch = len(images)
                    accuracy = correct_batch.double().div(total_batch)
                sys.stdout.write(f"""\rcurrent batch loss: {loss.item():.4f}\taccuracy: {accuracy.item()*100:.2f} %""")
                sys.stdout.flush()

                correct_predictions += correct_batch
                total_predictions += total_batch

            # avg batch loss / num_batches
            avg_epoch_loss = cum_train_loss / len(train_dataloader)
            print("Avg-Train error for epoch {epoch}: {avg_epoch_loss:.4f}")
            avg_epoch_acc = correct_predictions / total_predictions * 100
            print("Avg-Train accuracy for epoch {epoch}: {avg_epoch_acc:.2f} %")

            #TODO: implement validation


        #TODO: implement loading and saving of checkpoints
        def save_model(self, filepath: Union[str, Path]="./checkpoints") -> None:
            pass

        def load_model(self, filepath: Union[str, Path]) -> None:
            pass

