from .mnli import MNLIDataModule
from .sst2 import SST2DataModule
from .cola import CoLADataModule

__all__ = ["MNLIDataModule", "SST2DataModule", "CoLADataModule"]


def get_datamodule(task: str):
    """
    Get the data module for the given task.
    :param task: The task name.
    :return: The data module.
    """
    if task == "sst2":
        return SST2DataModule
    elif task == "mnli":
        return MNLIDataModule
    elif task == "cola":
        return CoLADataModule
    else:
        raise ValueError(f"Task {task} not supported")
