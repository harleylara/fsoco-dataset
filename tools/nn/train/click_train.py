import click

from .train import main

@click.command()
@click.argument("sly_project_folder_train", type=str)
@click.argument("output_folder", type=str)
@click.option("--config", type=str, default=False)
def train(sly_project_folder_train, output_folder, config):
    """
    #TODO add documentation

    """
    main(sly_project_folder_train, output_folder, config)

if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
