import torch
from torch import nn
from utils.dataloader import MNISTDataLoader
from utils.trainer import Trainer
from models.ARMs.mlp import MNIST_MLP
from models.ARMs.rnn import MNIST_LSTM

batch_size = 32
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001

MNIST_dataset = MNISTDataLoader(batch_size, transform_mode="discrete")
MNIST_training_set = MNIST_dataset.get_dataloader(train=True)  # , number=0)

hidden_size = 300
num_layers = 3
model = MNIST_LSTM(hidden_size=hidden_size, num_layers=num_layers)

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

trainer.train_ar_MNIST()

trainer.save_model("./save_models/LSTM_MNIST_all.pth")
