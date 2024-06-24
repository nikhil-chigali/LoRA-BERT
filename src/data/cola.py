from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset


class CoLADataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def prepare_data(self):
        # Load CoLA dataset
        self.dataset = load_dataset("glue", "cola", cache_dir="cache/data")

    def setup(self, stage=None):
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["sentence"],
                truncation=True,
                padding=True,
            )

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
