import abc
import enum
from pathlib import Path
import time
from typing import Union, Optional

import cv2
import decord
import numpy as np

from real_vas.utils.frame_utils import get_resize_dims


class VideoBackend(enum.Enum):
    DECORD = "decord"
    OPENCV = "opencv"


class VideoLoader:
    """
    A context manager for loading and processing video frames.
    """

    def __init__(self, video_path: Union[Path, str], backend: VideoBackend = VideoBackend.DECORD,
                 longest_edge: Optional[int] = None, flip_colour_channel: bool = False, print_stats: bool = False):

        self.video_path = str(video_path)
        self.backend = backend
        self.longest_edge = longest_edge
        self.flip_colour_channel = flip_colour_channel

        # We will create the actual loader inside __enter__ to manage resources correctly
        self._loader = None
        self.orig_frame_size = None
        self.print_stats = print_stats

    def get_num_frames(self):
        return len(self._loader)

    def get_fps(self):
        return self._loader.get_fps()

    def __enter__(self):
        """Opens the video file when entering a 'with' block."""
        # Peek at dimensions first
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, None
        # Get the width and height directly from the video's properties
        self.orig_frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
        del cap  # release the temporary reader

        if self.longest_edge is not None:
            height, width = get_resize_dims(self.orig_frame_size[0], self.orig_frame_size[1],
                                            longest_edge=self.longest_edge)
        else:
            height, width = -1, -1

        # Create and open the main loader
        self._loader = create_video_loader(self.backend, self.video_path, height=height, width=width,
                                           flip_colour_channel=self.flip_colour_channel)
        st = time.time()
        self._loader.open()
        if self.print_stats:
            print("Took {} seconds to create video loader for {}".format(time.time() - st, self.video_path))
        return self  # Return the instance of VideoLoader itself

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the video file when exiting the 'with' block."""
        self._loader.close()

    @staticmethod
    def _calculate_indices(total_frames: int, fps: float, frame_indices: list[int] = None,
                           frame_second_ranges: list[tuple[float, float]] = None,
                           every_x_frames: int = None, start_frame: int = None):
        extracted_indices = set(frame_indices or [])
        end_frame = total_frames
        if frame_second_ranges:
            base_start_frame = start_frame
            for start_sec, end_sec in frame_second_ranges:
                current_start_frame = base_start_frame if base_start_frame is not None else int(start_sec * fps)
                current_end_frame = int(end_sec * fps)
                extracted_indices.update(range(current_start_frame, min(current_end_frame + 1, total_frames)))
                base_start_frame = None
        if every_x_frames:
            effective_start_frame = start_frame or (min(extracted_indices) if extracted_indices else 0)
            temp_indices = set(range(effective_start_frame, end_frame, every_x_frames))
            if extracted_indices:
                extracted_indices.intersection_update(temp_indices)
            else:
                extracted_indices = temp_indices
        return sorted([idx for idx in extracted_indices if 0 <= idx < total_frames])

    def get_frame_generator(self, frame_indices: list[int] = None,
                            frame_second_ranges: list[tuple[float, float]] = None, every_x_frames: int = None,
                            start_frame: int = None, chunk_size: int = -1, just_return: bool = False):
        """
        Calculates frames to load and returns a generator and the total count.
        """
        if self._loader is None:
            raise RuntimeError("VideoLoader must be used within a 'with' statement.")

        fps = self._loader.get_fps()
        total_frames = len(self._loader)

        # --- Index calculation logic remains the same ---
        final_indices = self._calculate_indices(total_frames, fps, frame_indices, frame_second_ranges, every_x_frames,
                                                start_frame)

        if just_return:
            assert chunk_size == -1, "If just_return is True, chunk_size must be -1."
            if not final_indices:
                return np.array([]), []
            return self._loader.get_batch(final_indices), final_indices

        if not final_indices:
            def empty_generator(): yield from ()

            return empty_generator(), 0

        def generator():
            if chunk_size == -1:
                yield self._loader.get_batch(final_indices), final_indices
                return

            for i in range(0, len(final_indices), chunk_size):
                index_chunk = final_indices[i:i + chunk_size]
                if index_chunk:
                    yield self._loader.get_batch(index_chunk), index_chunk

        return generator(), len(final_indices)


class BaseVideoLoader(abc.ABC):
    """Abstract base class for video loaders."""

    def __init__(self, video_path: Union[Path, str], flip_colour_channel, **kwargs):
        self.video_path = str(video_path)
        self.reader = None
        self.flip_colour_channel = flip_colour_channel

    @abc.abstractmethod
    def open(self):
        """Opens the video file and prepares the reader."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Closes the video file and releases resources."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_batch(self, indices: list[int]) -> np.ndarray:
        """Fetches a batch of frames by their indices."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_fps(self) -> float:
        """Returns the video's frames per second."""
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total number of frames in the video."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> np.ndarray:
        """Allows for frame access like `loader[10]`."""
        return self.get_batch([index])[0]

    def __enter__(self):
        """Allows use in `with` statements for automatic resource management."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the video is closed when exiting a `with` block."""
        self.close()


class DecordLoader(BaseVideoLoader):
    """Video loader using the Decord library."""

    def __init__(self, video_path: Union[Path, str], height: int = -1, width: int = -1,
                 flip_colour_channel: bool = False):
        super().__init__(video_path, flip_colour_channel)
        self._height = height
        self._width = width

    def open(self):
        # Use OpenCV to get a more accurate frame count and FPS, as it tends to be more robust.
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self._reliable_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._reliable_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            # Fallback to decord if OpenCV fails for some reason
            self.reader = decord.VideoReader(self.video_path, ctx=decord.cpu(0), height=self._height, width=self._width)
            self._reliable_frame_count = len(self.reader)
            self._reliable_fps = self.reader.get_avg_fps()
            print(
                "Warning: OpenCV could not open the file, falling back to Decord's metadata which might be inaccurate.")
            return
        self.reader = decord.VideoReader(self.video_path, ctx=decord.cpu(0), height=self._height, width=self._width)

    def close(self):
        # Decord manages its own memory well, but clearing the reference is good practice.
        del self.reader

    def get_batch(self, indices: list[int]) -> np.ndarray:
        # Decord is fast and returns frames in standard RGB format.
        if self.flip_colour_channel:
            return self.reader.get_batch(indices).asnumpy()[
                ..., ::-1]  # TODO some videos are BGR, some RGB, so this is a hacky fix for now
        return self.reader.get_batch(indices).asnumpy()  # [..., ::-1]

    def get_fps(self) -> float:
        return self.reader.get_avg_fps()  # self._reliable_fps #

    def __len__(self) -> int:
        return len(self.reader)  # self._reliable_frame_count


class OpenCVLoader(BaseVideoLoader):
    """Video loader using the OpenCV library."""

    def open(self):
        self.reader = cv2.VideoCapture(self.video_path)
        if not self.reader.isOpened():
            raise IOError(f"Could not open video file with OpenCV: {self.video_path}")

    def close(self):
        self.reader.release()

    def get_batch(self, indices: list[int]) -> np.ndarray:
        # 1. Pre-allocate a list of None to hold the results in the requested order
        batch_frames = [None] * len(indices)

        # 2. Pair the requested original index with its position in the output
        #    We treat this as (target_position, frame_index_to_seek)
        indexed_requests = list(enumerate(indices))

        # 3. Sort by frame_index_to_seek to optimise seeking (minimise disk jumping)
        indexed_requests.sort(key=lambda x: x[1])

        for output_pos, frame_idx in indexed_requests:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.reader.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_frames[output_pos] = frame_rgb
            else:
                # Handle the failure.
                # Option A: Raise an error (Recommended for debugging)
                # raise ValueError(f"Could not read frame at index {frame_idx}")

                # Option B: Append a blank frame (black)
                height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                batch_frames[output_pos] = np.zeros((height, width, 3), dtype=np.uint8)

        if self.flip_colour_channel:
            return np.array(batch_frames)[..., ::-1]

        return np.array(batch_frames)

    def get_fps(self) -> float:
        return self.reader.get(cv2.CAP_PROP_FPS)

    def __len__(self) -> int:
        return int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))


def create_video_loader(backend: VideoBackend, video_path: Union[Path, str], **kwargs) -> BaseVideoLoader:
    """
    Factory function to create a video loader instance.
    Args:
        backend (VideoBackend): The backend to use ('decord' or 'opencv').
        video_path (str): Path to the video file.
    """
    if backend == VideoBackend.DECORD:
        return DecordLoader(video_path, **kwargs)
    elif backend == VideoBackend.OPENCV:
        return OpenCVLoader(video_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'decord' or 'opencv'.")
