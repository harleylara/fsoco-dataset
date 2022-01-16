import click
from typing import Any

from utils.logger import Logger
from .train import main


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_int(element: str) -> bool:
    try:
        int(element)
        return True
    except ValueError:
        return False


def try_convert_string(string: str) -> Any:
    if is_int(string):
        return int(string)
    elif is_float(string):
        return float(string)
    else:
        return string


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("sly_project_folder_train", type=str)
@click.argument("sly_project_folder_val", type=str)
@click.argument("working_folder", type=str)
@click.pass_context
def train(ctx, sly_project_folder_train, sly_project_folder_val, working_folder):
    """

    The "train" command allows you to train a YOLOv5 based network with the FSOCO dataset (or any other available in supervisely Format).

    https://github.com/ultralytics/yolov5

    \b

    The validation set is not automatically extracted from the training dataset to ensure the reproducibility of the experiments. Please split and store the dataset separately.

    The "train" command only needs a reference to the train/validation dataset and a working directory, where the intermediate dataset and artifacts from the YOLOv5 training are stored.

    Any additional arguments will be directly passed through to the YOLOv5 training script. You can use all arguments except "--data" as this will be overwritten by the fsoco CLI tool.

    `fsoco nn train /data/train_set /data/val_set ./work_dir --weights yolov5s.pt --batch-size 8 --epochs 30 --imgsz 640`

    You can use the following commands to show the training logs in tensorboard:

    `tensorboard --logdir ./work_dir/runs/train`

     view at http://localhost:6006/

    \b
    Output:

    work_dir
    ├── data
       ├── train
          └── Training dataset in Darknet format ...
       ├── val
          └── Validation dataset in Darknet format ...
       └── fsoco.yaml
    └── runs / train
       ├── exp
       ├── exp1
       ├── ....
       └── exp<N>
          ├── weights
             ├── best.pt
             └── last.pt
          └── other artifacts and tensorboard logs

    """

    kwargs = {}
    new_key = True
    key = ""
    for item in ctx.args:
        is_key = item.startswith("--")
        if is_key and new_key:
            key = item.strip("--")
            new_key = False
        elif is_key and not new_key:
            # key direct after key -> Flag
            kwargs[key] = True
        elif not is_key and not new_key:
            # value after key
            kwargs[key] = try_convert_string(item)
            new_key = True
        else:
            Logger.log_warn(f"unkown argument '{item}'")

    main(sly_project_folder_train, sly_project_folder_val, working_folder, kwargs)


if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
