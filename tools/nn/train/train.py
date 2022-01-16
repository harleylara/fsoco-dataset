# from yolov5 import train
from pathlib import Path
import yaml
from yolov5 import train
import os
import shutil

from utils.logger import Logger
from label_converters.sly2yolo.sly2yolo import main as sly2yolo


class FSOCOTrainer:
    def __init__(
        self,
        sly_project_folder_train: str,
        sly_project_folder_val: str,
        working_folder: str,
        train_kwargs: dict,
    ) -> None:

        self._working_directory = None
        self._data_dir = None
        self._weights_dir = None

        self._train_kwargs = train_kwargs

        self._train_classes = {}
        self._val_classes = {}

        self._data_config_file = None

        self._init_working_folder(working_folder)
        self._prepare_training_data(sly_project_folder_train, sly_project_folder_val)
        self._collect_meta_data()
        self._create_data_file()

    def _init_working_folder(self, working_folder: str) -> None:
        self._working_directory = Path(working_folder)
        self._data_dir = self._working_directory / "data"
        self._weights_dir = self._working_directory / "weights"

        self._train_data_dir = self._data_dir / "train"
        self._val_data_dir = self._data_dir / "val"

        self._working_directory.mkdir(parents=True, exist_ok=True)

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._train_data_dir.mkdir(parents=True, exist_ok=True)
        self._val_data_dir.mkdir(parents=True, exist_ok=True)

        Logger.log_info(f"Using working directory - {self._working_directory}")

    def _prepare_training_data(
        self, sly_project_folder_train: str, sly_project_folder_val: str
    ) -> None:
        Logger.log_info("Converting training dataset ...")

        train_converted_flag_file = (
            self._train_data_dir
            / f"{str(Path(sly_project_folder_train).name)}.converted"
        )
        val_converted_flag_file = (
            self._val_data_dir / f"{str(Path(sly_project_folder_val).name)}.converted"
        )

        if not train_converted_flag_file.exists():
            shutil.rmtree(self._train_data_dir)
            self._train_data_dir.mkdir(parents=True, exist_ok=True)

            sly2yolo(
                sly_project_path=sly_project_folder_train,
                output_path=str(self._train_data_dir.absolute()),
                remove_watermark=True,
                exclude=[],
                keep_image_extension=False,
            )

            with open(train_converted_flag_file, "w") as f:
                f.write("FLAG_FILE")

        else:
            Logger.log_info(f"Using chached train dataset in -> {self._train_data_dir}")

        Logger.log_info("Converting validation dataset ...")

        if not val_converted_flag_file.exists():
            shutil.rmtree(self._val_data_dir)
            self._val_data_dir.mkdir(parents=True, exist_ok=True)

            sly2yolo(
                sly_project_path=sly_project_folder_val,
                output_path=str(self._val_data_dir.absolute()),
                remove_watermark=True,
                exclude=[],
                keep_image_extension=False,
            )

            with open(val_converted_flag_file, "w") as f:
                f.write("FLAG_FILE")
        else:
            Logger.log_info(f"Using chached val dataset in -> {self._val_data_dir}")

    def _collect_meta_data(self):
        train_class_file = self._train_data_dir / "classes.txt"
        val_class_file = self._val_data_dir / "classes.txt"

        with open(train_class_file, "r") as f:
            for n, line in enumerate(f):
                class_name = line.strip()
                self._train_classes[class_name] = n

        with open(val_class_file, "r") as f:
            for n, line in enumerate(f):
                class_name = line.strip()
                self._val_classes[class_name] = n

                if self._train_classes.get(class_name, -1) != n:
                    Logger.log_warn(
                        f"Class '{class_name}' with id {n} not found in training dataset!"
                    )

    def _create_data_file(self):
        self._data_config_file = self._data_dir / "fsoco.yml"
        data = dict(
            path=str(self._data_dir.name),
            train=str(self._train_data_dir.name),
            val=str(self._val_data_dir.name),
            test="",
            nc=len(self._train_classes.keys()),
            names=[class_name for class_name in self._train_classes.keys()],
        )

        with open(self._data_config_file, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=None)

        Logger.log_info(
            f"Stored training data configuration to {str(self._data_config_file)}"
        )

        pass

    def train(self):
        Logger.log_info("Start Training")

        cwd = os.getcwd()
        os.chdir(self._working_directory)

        if "data" in self._train_kwargs.keys():
            Logger.log_warn(
                "Please do not use your own data file. We auto-create it for you. Your value will be overwritten!"
            )

        self._train_kwargs["data"] = str(
            self._data_config_file.relative_to(self._working_directory)
        )

        train.run(**self._train_kwargs)

        os.chdir(cwd)


def main(
    sly_project_folder_train, sly_project_folder_val, working_folder, train_kwargs
):
    trainer = FSOCOTrainer(
        sly_project_folder_train, sly_project_folder_val, working_folder, train_kwargs
    )
    trainer.train()
