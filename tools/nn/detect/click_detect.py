import click

from .detect import main
from utils.logger import Logger


@click.command()
@click.argument("model", help="path to YOLOv5 weights file", type=str)
@click.argument(
    "input_data",
    help="Path/wildcard for input data. Please don't forget to put your wildcard into quotes '*.xyz', to stop the shell from expanding it.",
    type=str,
)
@click.argument("output_path", help="Path where the output should be stored.", type=str)
@click.argument("catch_wildcard_expansion", nargs=-1)
@click.option("--device", type=str, default="cuda:0")
@click.option(
    "--confidence_threshold", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.4
)
@click.option("--sliced_inference", type=bool, default=True)
@click.option("--slice_height", type=click.IntRange(0), default=256)
@click.option("--slice_width", type=click.IntRange(0), default=256)
@click.option(
    "--overlap_height_ratio", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.2
)
@click.option(
    "--overlap_width_ratio", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.2
)
@click.option(
    "--video_sampling_step",
    help="Videos in your input data will be sampled with this step size.",
    type=click.IntRange(0),
    default=10,
)
def detect(
    model,
    input_data,
    output_path,
    catch_wildcard_expansion,
    device,
    confidence_threshold,
    sliced_inference,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
    video_sampling_step,
):
    """
    #TODO add documentation

    """

    # check for non quoted image glob
    if len(catch_wildcard_expansion):
        Logger.log_error(
            "It looks like you did not put your input glob into quotes and the shell already expanded it!"
        )
        Logger.log_error("Please put your Glob into quotation marks.")
        Logger.log_error("detect yolov5s.pt '*/*.jpeg' out_dir ... ")
        return False

    main(
        model,
        input_data,
        output_path,
        device,
        confidence_threshold,
        sliced_inference,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
        video_sampling_step,
    )


if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
