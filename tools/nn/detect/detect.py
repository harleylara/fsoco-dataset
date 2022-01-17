# import cv2
# import torch
# from PIL import Image


class FSOCODetector:
    def __init__(self) -> None:
        pass

    def detect_on_images(self):
        pass

    def detect_on_video(self):
        pass


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
    print("detect")
