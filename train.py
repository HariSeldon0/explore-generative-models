import torch
from torch import nn
from utils.dataloader import MNISTDataLoader
from utils.trainer import Trainer
from models.ARMs.mlp import MNIST_MLP

batch_size = 36
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.01

MNIST_dataset = MNISTDataLoader(batch_size, transform_mode="discrete")
MNIST_training_set = MNIST_dataset.get_dataloader(train=True, number=1)

model = MNIST_MLP()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(
    dataloader=MNIST_training_set,
    model=model,
    epochs=epochs,
    device=device,
    optimizer=optimizer,
    loss_fn=loss_fn,
)

trainer.train_ar()

trainer.save_model("./save_models/MLP_MNIST_1.pth")
