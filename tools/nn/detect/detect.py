# import cv2
# import torch
# from PIL import Image


from utils.logger import Logger


class FSOCODetector:
    def __init__(
        self,
        model: str,
        device: str,
        confidence_threshold: float,
        sliced_inference: bool,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
    ) -> None:
        self._model = model
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._sliced_inference = sliced_inference
        self._slice_height = slice_height
        self._slice_width = slice_width
        self._overlap_height_ratio = overlap_height_ratio
        self._overlap_width_ratio = overlap_width_ratio

    def detect(self):
        pass


class InferenceManager:
    def __init__(self, detector: FSOCODetector, video_sampling_step: int = 1) -> None:
        self._detector = detector
        self._video_sampling_step = video_sampling_step

    def run(self, input_data: str, output_path: str):
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
