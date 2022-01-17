# import cv2
# import torch
# from PIL import Image


from utils.logger import Logger
from nn.detect.fsoco_detector import FSOCODetector
from nn.detect.inference_manager import InferenceManager


def main(
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
):
    Logger.log_info("Starting detection")

    detector = FSOCODetector(
        model,
        device,
        confidence_threshold,
        sliced_inference,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
    )

    manager = InferenceManager(
        detector=detector, video_sampling_step=video_sampling_step
    )
    manager.run(input_data, output_path)
