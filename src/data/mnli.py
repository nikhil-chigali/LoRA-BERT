from typing import Tuple
from functools import partial
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset


class MNLIDataModule(pl.LightningDataModule):
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, cache_dir="cache/tokenizer"
        )
        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def prepare_data(self):
        # Load MNLI dataset
        self.dataset = load_dataset("glue", "mnli", cache_dir="cache/data")

    def setup(self, stage=None):
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        # MNLI has two validation sets: matched and mismatched
        matched_val = DataLoader(
            self.dataset["validation_matched"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        mismatched_val = DataLoader(
            self.dataset["validation_mismatched"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return [matched_val, mismatched_val]

    def test_dataloader(self):
        # MNLI has two test sets: matched and mismatched
        matched_test = DataLoader(
            self.dataset["test_matched"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        mismatched_test = DataLoader(
            self.dataset["test_mismatched"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return [matched_test, mismatched_test]
