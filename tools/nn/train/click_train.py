import click

from utils.logger import Logger

from .train import main


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("sly_project_folder_train", type=str)
@click.argument("working_folder", type=str)
@click.option("--test_split", type=click.FloatRange(0, 1.0, clamp=True))
@click.pass_context
def train(ctx, sly_project_folder_train, working_folder, test_split):
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
            kwargs[key] = item
            new_key = True
        else:
            Logger.log_warn(f"unkown argument '{item}'")

    main(sly_project_folder_train, working_folder, test_split, kwargs)


if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
