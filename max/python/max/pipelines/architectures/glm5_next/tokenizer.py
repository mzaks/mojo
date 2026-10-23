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
"""GLM-5.3-Flash tokenizer: the GLM-5.x text tokenizer plus image inputs.

The chat template wraps each image as
``<|begin_of_image|><|image|><|end_of_image|>``, and the reference processor
(``Glm5NextProcessor.replace_image_token``) widens the one ``<|image|>`` to
``t * h * w / merge**2`` copies, one per merged vision token. The tower's rows
are scattered onto exactly those positions; the decoder is NoPE, so there are
no position ids to adjust.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.glm5_1.tokenizer import GlmTokenizer
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import PipelineConfig, max_tokens_to_generate
from max.pipelines.lib.tokenizer import (
    encode_dkv_cache_hint,
    open_image,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.support.image import hash_image
from PIL import Image
from transformers.utils import cached_file

from .context import Glm5NextTextAndVisionContext
from .image_processor import Glm5NextImageProcessor

logger = logging.getLogger("max.pipelines")

__all__ = ["Glm5NextTokenizer"]

_PROCESSOR_CONFIG = "processor_config.json"


def _load_processor_config(
    model_path: str, revision: str | None
) -> dict[str, Any]:
    """Reads ``processor_config.json``, or ``{}`` if the checkpoint has none."""
    try:
        path = cached_file(model_path, _PROCESSOR_CONFIG, revision=revision)
    except OSError:
        path = None
    if path is None:
        logger.warning(
            "%s has no %s; preprocessing images with GLM-5.3-Flash's "
            "published defaults.",
            model_path,
            _PROCESSOR_CONFIG,
        )
        return {}
    with open(path) as f:
        return json.load(f)


class Glm5NextTokenizer(GlmTokenizer):
    """Builds :class:`.Glm5NextTextAndVisionContext` for text and images.

    Text handling -- the chat template, reasoning effort and tool-result
    ordering -- is :class:`GlmTokenizer`'s, unchanged.
    """

    # TODO(GLM53-VISION): video. The placeholder is the same `<|image|>`, one
    # `<|begin_of_image|>...<|end_of_image|>{t:.1f} seconds` run per temporal
    # patch inside a `<|begin_of_video|>` span, so each run maps onto one
    # `grid_t = 1` image entry. Missing are decoding the clip, `sample_frames`
    # (clamp its sub-second empty result, see `arch.py`) and the timestamps.
    max_videos_per_request = 0

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path, pipeline_config, revision=revision, **kwargs
        )
        hf_config = pipeline_config.model.huggingface_config
        self.image_token_id = int(hf_config.image_token_id)
        self._video_start_token_id = int(hf_config.video_start_token_id)
        self._video_end_token_id = int(hf_config.video_end_token_id)
        self.image_processor = Glm5NextImageProcessor.from_processor_config(
            _load_processor_config(model_path, revision)
        )
        # Every image expands to at least `min_image_tokens` placeholders, so
        # more images than this cannot fit; the router rejects them before
        # downloading any.
        self.max_images_per_request = (
            self.max_length // max(1, self.image_processor.min_image_tokens)
            if self.max_length
            else None
        )
        self._hash_images = (
            pipeline_config.model.kv_cache.enable_prefix_caching
            or pipeline_config.runtime.vision_cache_utilization != 0
        )

    @property
    def expects_content_wrapping(self) -> bool:
        """Keeps image parts as placeholders for the chat template."""
        return True

    def _flatten_message(
        self, message: TextGenerationRequestMessage
    ) -> dict[str, Any]:
        """Leaves a message with an image as a list of parts.

        GLM's template renders each image part in place. Text-only messages
        are flattened as before, which joins multiple text parts with
        newlines.
        """
        if message.number_of_images == 0:
            return super()._flatten_message(message)
        return message.model_dump(exclude_none=True)

    def _image_placeholders(
        self, token_ids: npt.NDArray[np.integer[Any]]
    ) -> npt.NDArray[np.intp]:
        """Positions of image placeholders outside any video span.

        A video frame uses the same token; the reference tells them apart by
        whether a ``<|begin_of_video|>`` is open, as done here.
        """
        is_placeholder = token_ids == self.image_token_id
        in_video = np.cumsum(token_ids == self._video_start_token_id) > (
            np.cumsum(token_ids == self._video_end_token_id)
        )
        if np.any(is_placeholder & in_video):
            raise ValueError("GLM-5.3-Flash does not accept video input yet.")
        return np.flatnonzero(is_placeholder)

    def expand_image_placeholders(
        self,
        token_ids: npt.NDArray[np.integer[Any]],
        images: Sequence[bytes | Image.Image],
        image_hashes: Sequence[int | None],
    ) -> tuple[
        npt.NDArray[np.integer[Any]], list[ImageMetadata], npt.NDArray[np.int64]
    ]:
        """Preprocesses the images and widens each placeholder to its rows.

        A request without images passes through unchanged, so a prompt that
        quotes ``<|image|>`` keeps it as text, as the base GLM tokenizer did.

        Args:
            token_ids: The encoded prompt, one placeholder per image.
            images: The request's images, in prompt order.
            image_hashes: One content hash per image, or ``None`` per image
                when no cache needs one.

        Returns:
            The expanded token ids, one :class:`ImageMetadata` per image whose
            span covers exactly its placeholders, and the ``(t, h, w)`` grids.

        Raises:
            ValueError: If the placeholder and image counts differ.
            PromptTooLongError: If the expanded prompt exceeds ``max_length``.
        """
        if not images:
            return token_ids, [], np.empty((0, 3), dtype=np.int64)
        positions = self._image_placeholders(token_ids)
        if len(positions) != len(images):
            raise ValueError(
                f"The prompt has {len(positions)} image placeholders but the "
                f"request carries {len(images)} images."
            )
        opened = [open_image(image) for image in images]
        counts = [
            self.image_processor.num_image_tokens(
                self.image_processor.grid_thw(image)
            )
            for image in opened
        ]
        # Checked before patchifying: images below the token floor are
        # upscaled, so many tiny ones would take gigabytes first.
        length = len(token_ids) + sum(counts) - len(counts)
        if self.max_length and length > self.max_length:
            raise PromptTooLongError(length, self.max_length)
        processed = [self.image_processor.preprocess(image) for image in opened]
        expanded = np.insert(
            token_ids,
            np.repeat(positions, [count - 1 for count in counts]),
            self.image_token_id,
        )
        starts = (
            positions + np.cumsum([0, *counts[:-1]]) - np.arange(len(counts))
        )
        metadata = [
            ImageMetadata(
                start_idx=int(start),
                end_idx=int(start) + count,
                pixel_values=patches,
                image_hash=image_hash,
            )
            for start, count, (patches, _), image_hash in zip(
                starts, counts, processed, image_hashes, strict=True
            )
        ]
        grids = np.array([grid for _, grid in processed], dtype=np.int64)
        return expanded, metadata, grids

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Glm5NextTextAndVisionContext:
        """Creates a context, expanding image placeholders if there are any."""
        if request.videos:
            raise ValueError("GLM-5.3-Flash does not accept video input yet.")
        _prompt, token_ids = await self._generate_prompt_and_token_ids(
            prompt=request.prompt,
            messages=request.messages,
            tools=request.tools,
            **(request.chat_template_options or {}),
        )

        # Keyed on the encoded bytes and the token budget, which together fix
        # the padded canvas the tower sees.
        image_hashes = [
            hash_image(raw, self.image_processor.max_image_tokens)
            if self._hash_images
            else None
            for raw in request.images
        ]
        token_ids, images, grid_thws = await run_with_default_executor(
            self.expand_image_placeholders,
            token_ids,
            request.images_for_processing(),
            image_hashes,
        )
        if self.max_length and len(token_ids) > self.max_length:
            raise PromptTooLongError(len(token_ids), self.max_length)

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )
        grammar = (
            request.response_format.grammar if request.response_format else None
        )
        max_gen_tokens = max_tokens_to_generate(
            len(token_ids),
            self.max_length,
            request.sampling_params.max_new_tokens,
        )
        return Glm5NextTextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=TokenBuffer(array=token_ids.astype(np.int64, copy=False)),
            vocab_size=self.tokenizer_vocab_size,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=GrammarEnforcementState.from_response_format(
                request.response_format
            ),
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            cache_salt=request.cache_salt,
            images=images,
            vision_token_ids=[self.image_token_id],
            grid_thws=grid_thws,
        )
