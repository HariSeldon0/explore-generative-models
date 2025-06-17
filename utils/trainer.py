import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        dataloader,
        model,
        epochs,
        device,
        optimizer,
        loss_fn,
        log_file="loss_log.txt",
    ):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.log_file = log_file

        # Clear the log file at the beginning
        with open(self.log_file, "w") as f:
            f.write("Epoch,Iteration,Loss\n")

    def autoregressive_transform(self, data):
        data = torch.flatten(data, start_dim=1)
        batch_size, seq_len = data.shape
        labels = torch.flatten(data).long()
        data = (
            data.unsqueeze(1).expand(batch_size, seq_len, seq_len).reshape(-1, seq_len)
        )
        mask = (
            torch.tril(torch.ones(seq_len, seq_len), diagonal=-1)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * seq_len, seq_len)
        )
        return data, mask, labels

    def train(self):
        for epoch in range(self.epochs):
            print(f"### epoch {epoch} ###")
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, (inputs, labels) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(inputs)

            loss = self.loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            print(f"loss:{loss_value}")

            # Save loss to file
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{idx},{loss_value}\n")

    def train_ar_MNIST(self):
        for epoch in range(self.epochs):
            print(f"### epoch {epoch} ###")
            self._train_ar_MNIST_epoch(epoch)

    def _train_ar_MNIST_epoch(self, epoch):
        self.model.train()
        for idx, (inputs, labels) in enumerate(self.dataloader):

            inputs = inputs.to(self.device).reshape(inputs.shape[0], -1)
            labels = inputs.clone().long()
            inputs[:, 1:] = inputs[:, :-1].clone()
            inputs[:, 0] = 0
            inputs = inputs.reshape(inputs.shape[0], -1, 1).float()
            (h_0, c_0) = self.model.get_init_hidden(inputs.shape[0])
            h_0 = h_0.to(self.device)
            c_0 = c_0.to(self.device)

            self.optimizer.zero_grad()
            preds, _ = self.model(inputs, (h_0, c_0))
            preds = preds.permute(0, 2, 1)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            if idx % 10 == 0:
                print(f"{idx} loss:{loss_value}")

            # Save loss to file
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{idx},{loss_value}\n")

    def save_model(self, path):
        torch.save(self.model, path)
