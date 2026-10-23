# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""GLM-5.3-Flash video frame sampling and preprocessing.

The pixel pipeline is :class:`~.image_processor.Glm5NextPatchifier`'s -- same
``smart_resize``, same resize-then-pad canvas, same patchify -- with a
240,000-token budget instead of 8,000 and a real temporal extent instead of a
still image duplicated into the temporal patch.

:meth:`Glm5NextVideoProcessor.sample_frames` is the new work, and it is not
GLM-4.6V's rule. Four things about it are easy to get subtly wrong, and each
produces a clip that encodes and answers slightly badly rather than failing:

* the second-boundary walk advances by ``1 / target_fps`` but breaks against
  ``int(duration)``, so it yields about ``duration * target_fps`` frames and the
  cap is on the *original* duration, not a clamped one;
* ``np.linspace`` is the fallback in both directions -- too few frames and too
  many -- but over different ranges: the sampled span in the first case and the
  whole clip in the second;
* duplicates are removed *after* the fallback, so the count can drop below
  ``extract_t``;
* the even-count forcing runs *after* the dedup and duplicates the last index,
  so the returned list can legitimately repeat. The patch embedding consumes
  frames in temporal pairs, so an odd count would leave a half-filled temporal
  patch.

Budgets are the shipped ones: ``max_image_tokens`` 240,000 decoder tokens,
``fps`` 2, ``max_frames`` 2048. ``max_image_size`` (``28 * 28 * 2 * 30000``)
exists in the reference's config but its resize path does not read it, so it is
not modelled here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from typing_extensions import Self

from .image_processor import Glm5NextPatchifier

__all__ = ["Glm5NextVideoProcessor"]


class Glm5NextVideoProcessor(Glm5NextPatchifier):
    """Samples frames from a clip and patchifies them for the GLM-5.3 tower."""

    CONFIG_SECTION = "video_processor"

    def __init__(
        self,
        *,
        max_image_tokens: int = 240000,
        fps: float = 2.0,
        max_frames: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_image_tokens=max_image_tokens, **kwargs)
        self.fps = fps
        self.max_frames = max_frames

    @classmethod
    def from_processor_config(
        cls, config: Mapping[str, Any], **overrides: Any
    ) -> Self:
        section = config.get(cls.CONFIG_SECTION, config)
        defaults = cls()
        overrides.setdefault("fps", section.get("fps", defaults.fps))
        overrides.setdefault(
            "max_frames", section.get("max_frames", defaults.max_frames)
        )
        return super().from_processor_config(config, **overrides)

    def sample_frames(
        self,
        total_num_frames: int,
        native_fps: float,
        duration: float | None = None,
        target_fps: float | None = None,
    ) -> npt.NDArray[np.int64]:
        """Picks which frame indices of a clip the tower sees.

        Args:
            total_num_frames: Frames the decoded clip holds.
            native_fps: The clip's own frame rate, from its metadata.
            duration: The clip's duration in seconds, if the container reports
                one. Derived from the frame count and rate when it does not.
            target_fps: Sampling rate. Defaults to the configured ``fps``.

        Returns:
            An even-length, ascending array of frame indices, possibly with the
            last index repeated once.

        Raises:
            ValueError: If ``native_fps`` is missing, which the reference also
                refuses to guess at.
        """
        if not native_fps:
            raise ValueError(
                "Frame sampling needs the clip's native fps; GLM-5.3-Flash "
                "budgets frames in seconds, not in frame counts."
            )
        max_frame_idx = total_num_frames - 1
        duration = duration or round(max_frame_idx / native_fps) + 1
        # The cap is on the original duration. The reference computes it before
        # any `max_duration` clamp, which is disabled in the shipped config.
        max_seconds = int(duration)
        if target_fps is None:
            target_fps = self.fps

        extract_t = min(int(duration * target_fps), self.max_frames)
        duration_per_frame = 1 / native_fps
        timestamps = [i * duration_per_frame for i in range(total_num_frames)]

        frame_indices: list[int]
        if total_num_frames < extract_t:
            frame_indices = np.linspace(
                0, total_num_frames - 1, extract_t, dtype=int
            ).tolist()
        else:
            frame_indices = []
            current_second = 0.0
            inv_fps = 1 / target_fps
            for frame_index in range(total_num_frames):
                if timestamps[frame_index] >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_seconds:
                        break

        if len(frame_indices) < extract_t:
            if not frame_indices:
                start, end = 0, max(total_num_frames - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(
                start, end, extract_t, dtype=int
            ).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(
                0, total_num_frames - 1, extract_t, dtype=int
            ).tolist()

        seen: set[int] = set()
        unique: list[int] = []
        for index in frame_indices:
            if index not in seen:
                seen.add(index)
                unique.append(index)

        # After the dedup, and by duplicating the last index -- so the result
        # can repeat a frame. The patch embedding pairs frames temporally.
        if len(unique) & 1:
            unique.append(unique[-1])

        return np.array(unique, dtype=np.int64)

    def grid_thw(self, frames: Sequence[Image.Image]) -> tuple[int, int, int]:
        """Returns a clip's ``(t, h, w)`` grid without touching the pixels."""
        width, height = frames[0].size
        num_frames = len(frames)
        canvas_height, canvas_width = self.canvas_size(
            height, width, num_frames
        )
        remainder = num_frames % self.temporal_patch_size
        aligned_frames = num_frames + (
            self.temporal_patch_size - remainder if remainder else 0
        )
        return (
            aligned_frames // self.temporal_patch_size,
            canvas_height // self.patch_size,
            canvas_width // self.patch_size,
        )

    def preprocess(
        self, frames: Sequence[Image.Image]
    ) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
        """Patchifies an already-sampled clip.

        Every frame shares one canvas, sized from the frames' common size and
        the clip's frame count, because the tower consumes a single
        ``(t, h, w)`` grid per clip.

        Args:
            frames: The sampled frames, in order.

        Returns:
            The ``(num_patches, 3 * temporal_patch_size * patch_size**2)``
            float32 patch tensor and its ``(t, h, w)`` grid.

        Raises:
            ValueError: If ``frames`` is empty or the frames disagree on size.
        """
        if not frames:
            raise ValueError("A clip needs at least one frame.")
        sizes = {frame.size for frame in frames}
        if len(sizes) != 1:
            raise ValueError(
                f"All frames of a clip must share one size, got {sizes}."
            )

        width, height = frames[0].size
        num_frames = len(frames)
        canvas_height, canvas_width = self.canvas_size(
            height, width, num_frames
        )
        content_height, content_width = self.content_size(
            height, width, canvas_height, canvas_width, num_frames
        )
        normalized = np.stack(
            [
                self.normalize(
                    self.resize_and_pad(
                        frame,
                        canvas_height,
                        canvas_width,
                        content_height,
                        content_width,
                    )
                )
                for frame in frames
            ]
        )
        return self.patchify(normalized)

    def __call__(
        self, clips: Sequence[Sequence[Image.Image]]
    ) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int32]]:
        """Patchifies each clip, returning per-clip patches and the grids."""
        processed = [self.preprocess(clip) for clip in clips]
        return (
            [patches for patches, _ in processed],
            np.array([grid for _, grid in processed], dtype=np.int32).reshape(
                len(processed), 3
            ),
        )
