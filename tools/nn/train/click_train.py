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
    #TODO add documentation

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
