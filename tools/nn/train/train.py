# from yolov5 import train
from pathlib import Path

from utils.logger import Logger


class FSOCOTrainer:
    def __init__(
        self,
        sly_project_folder_train: str,
        working_folder: str,
        test_split: float,
        train_kwargs: dict,
    ) -> None:

        self._working_dir = self._init_working_folder(working_folder)

    def _init_working_folder(self, working_folder: str) -> None:
        work_dir = Path(working_folder)
        work_dir.mkdir(parents=True, exist_ok=True)

        Logger.log_info(f"Using working directory - {work_dir}")

    def train(self):
        pass


def main(sly_project_folder_train, working_folder, test_split, train_kwargs):
    trainer = FSOCOTrainer(
        sly_project_folder_train, working_folder, test_split, train_kwargs
    )
    trainer.train()
