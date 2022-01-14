import click

from nn.train.click_train import train
from nn.detect.click_detect import detect


@click.group()
def nn():
    """
    NN - Neural Network tools

    The commands in this group help you train and run networks on your data.
    """
    pass
    
    
nn.add_command(train)
nn.add_command(detect)


if __name__ == "__main__":
    print(
        "[LOG] This sub-module contains Label Converters and is not meant to be run as a stand-alone script"
    )
