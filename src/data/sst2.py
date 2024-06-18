from functools import partial
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset


class SST2DataModule(pl.LightningDataModule):
    """
    Sequence classification data module.
    :param model_name: The model name.
    :param dataset_name: The dataset name.
    :param batch_size: The batch size.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = ("glue", "sst2")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None
        self.classes = None
        self.tokenizer = None
        self.tokenized = None
        self.collate_fn = None

    def prepare_data(self):
        # Download the dataset
        load_dataset(
            *self.dataset_name,
            cache_dir="cache/data",
        )

    def setup(self, stage=None):
        # Load the dataset
        self.dataset = load_dataset(*self.dataset_name, cache_dir="cache/data")
        self.classes = self.dataset["train"].features["label"].names
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            cache_dir="cache/tokenizer",
        )
        # Tokenize the dataset
        self.tokenized = _get_tokens_for_seq_classification(
            self.tokenizer, self.dataset, self.batch_size
        )
        # Data collate function
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized["train"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized["test"],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
        )


def __tokenize_for_seq_classification(tokenizer: AutoTokenizer, batch: dict) -> dict:
    """
    Tokenize a batch for sequence classification.
    :param tokenizer: The tokenizer.
    :param batch: The batch.
    :return: The tokenized batch.
    """
    return tokenizer(
        batch["sentence"],
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
    )


def _get_tokens_for_seq_classification(
    tokenizer: AutoTokenizer, dataset: dict, batch_size: int
) -> dict:
    """
    Get tokens for sequence classification.
    :param tokenizer: The tokenizer.
    :param dataset: The dataset.
    :param batch_size: The batch size.
    :return: The tokenized dataset.
    """

    partial_tokenize_for_seq_classification = partial(
        __tokenize_for_seq_classification, tokenizer
    )

    tokenized = dataset.map(
        partial_tokenize_for_seq_classification, batch_size=batch_size, batched=True
    )
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_columns({"label": "labels"})
    tokenized.set_format("torch")
    return tokenized
