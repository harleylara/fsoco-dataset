import click

from .detect import main


@click.command()
@click.argument("model", type=str)
@click.argument("image_folder_path", type=str)
@click.option("--config", type=str, default=False)
def detect(model, image_folder_path, config):
    """
    #TODO add documentation

    """
    main(model, image_folder_path, config)


if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
