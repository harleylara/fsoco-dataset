from pathlib import Path

from nn.detect.fsoco_detector import FSOCODetector
from utils.logger import Logger


class InferenceManager:
    def __init__(self, detector: FSOCODetector, video_sampling_step: int = 1) -> None:
        self._detector = detector
        self._video_sampling_step = video_sampling_step

        self._output_base_directory = None
        self._output_run_directory = None
        self._output_image_directory = None
        self._output_label_directory = None
        self._output_debug_directory = None

    def _init_output_folder(self, output_path: Path):
        self._output_base_directory = output_path

        if self._output_base_directory.exists():
            existing_run_folders = [
                object.name
                for object in self._output_base_directory.glob("run_*")
                if object.is_dir()
            ]
            max_id = 0

            for folder in existing_run_folders:
                try:
                    id = int(folder.split("_")[1])
                except:  # noqa: E722
                    id = 0

                max_id = max(max_id, id)

            self._init_run_directory(max_id + 1)

        else:
            self._output_base_directory.mkdir(parents=True, exist_ok=True)
            self._init_run_directory(0)

    def _init_run_directory(self, id: int):
        self._output_run_directory = self._output_base_directory / f"run_{id}"

        if self._output_run_directory.exists():
            self._init_run_directory(id + 1)
            return

        self._output_run_directory.mkdir(parents=True, exist_ok=False)
        Logger.log_info(f"Using output run folder -> {self._output_run_directory}")

        self._output_image_directory = self._output_run_directory / "images"
        self._output_label_directory = self._output_run_directory / "labels"
        self._output_debug_directory = self._output_run_directory / "debug"

        self._output_image_directory.mkdir(parents=False, exist_ok=False)
        self._output_label_directory.mkdir(parents=False, exist_ok=False)
        self._output_debug_directory.mkdir(parents=False, exist_ok=False)

    def run(self, input_data: str, output_path: str):
        self._init_output_folder(Path(output_path))

        pass
