import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


class ToDiscreteTensor:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.int64)
        return torch.tensor(image_array, dtype=torch.int)  # 保持离散整数类型张量


class MNISTDataLoader:

    def __init__(self, batch_size=32, transform_mode="[0,1]", shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if transform_mode == "[0,1]":
            self.transform = (
                transforms.ToTensor()
            )  # Convert to FloatTensor of shape (C*H*W) in range [0.0, 1.0]
        elif transform_mode == "[-1,1]":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # Convert to FloatTensor of shape (C*H*W) in range [0.0, 1.0]
                    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1.0, 1.0]
                ]
            )
        elif transform_mode == "discrete":
            self.transform = ToDiscreteTensor()
        else:
            raise "invalid transform mode"

    def get_dataloader(self, train=True, number=None):
        dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=self.transform
        )

        if number is not None:
            indices = torch.where(dataset.targets == number)[0]  # 选择指定类别的索引
            dataset = Subset(dataset, indices)  # 只保留指定类别的数据

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        return dataloader


class Wikitext2DataLoader:

    def __init__(self, batch_size=32, model_name="bert-base-uncased", shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",  # padding all sentence to max length
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def collate_batch(self, batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings

    def get_dataloader(self, split="train"):
        tokenized_dataset = self.dataset[split].map(
            self.tokenize_function, batched=True
        )
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_batch,
        )
        return dataloader
