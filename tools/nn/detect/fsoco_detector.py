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
