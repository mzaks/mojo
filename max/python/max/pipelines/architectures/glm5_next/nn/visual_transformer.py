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
"""GLM-5.3-Flash vision tower.

Structurally GLM-4.6V's tower with two modules removed and three changed, which
is exactly how the reference builds it: ``Glm5NextVisionModel`` derives from
``GlmOcrVisionModel``, which derives from ``Glm4vVisionModel`` and whose
``__init__`` is little more than ``del self.embeddings`` and
``del self.post_conv_layernorm``.

Removed, because the checkpoint has no weights for them: the bicubically
interpolated absolute position embedding, and the post-convolution norm.
Changed: attention gains q/k norms and biases, and both SwiGLUs are clamped.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import LayerNorm, RMSNorm

from ..layers.clamped_swiglu import clamped_swiglu
from ..model_config import Glm5NextVisionConfig
from .vision_attention import Glm5NextVisionAttention

__all__ = [
    "Glm5NextVisionBlock",
    "Glm5NextVisionSpatialMerge",
    "Glm5NextVisionTransformer",
]


def _patch_dim(config: Glm5NextVisionConfig) -> int:
    return (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )


def _spatial_merge_unit(config: Glm5NextVisionConfig) -> int:
    return config.spatial_merge_size * config.spatial_merge_size


class Glm5NextVisionMLP(Module):
    """Clamped SwiGLU MLP, used by every block and by the merger."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        has_bias: bool,
        swiglu_limit: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.gate_proj = Linear(
            in_dim=hidden_dim,
            out_dim=intermediate_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=intermediate_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.down_proj = Linear(
            in_dim=intermediate_dim,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.down_proj(
            clamped_swiglu(
                self.gate_proj(x), self.up_proj(x), self.swiglu_limit
            )
        )


class Glm5NextVisionPatchEmbed(Module):
    """Embeds flattened pixel patches.

    The reference is a ``Conv3d`` whose kernel equals its stride and covers a
    whole patch, so it runs here as a matmul over the already-flattened patch
    rows. The weight adapter flattens the checkpoint's 5-D tensor; its memory
    order is ``(channel, temporal, height, width)``, which is the order the
    image processor emits, so no permutation is involved.
    """

    def __init__(self, config: Glm5NextVisionConfig, device: DeviceRef) -> None:
        super().__init__()
        self.proj = Linear(
            in_dim=_patch_dim(config),
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        return self.proj(pixel_values.cast(self.proj.weight.dtype))


class Glm5NextVisionSpatialMerge(Module):
    """The tower's 2x2 strided-conv spatial merge, run as a matmul.

    The convolution sees exactly one ``merge x merge`` window per output token,
    so it is a matmul over the four merged patches -- but only under one
    flattening, and the wrong one is a silent accuracy loss rather than a shape
    error. The weight adapter therefore leaves ``downsample.weight`` in its
    checkpoint form, ``(out, channel, kh, kw)``, and this module owns the
    flattening on both sides:

    * the reference reshapes the token sequence to
      ``(-1, merge, merge, hidden)`` before permuting into NCHW, so within a
      merge block the four patch rows are ordered ``(h_in_block, w_in_block)``
      and a flat row of four concatenated rows is ``(h, w, channel)``;
    * the weight is therefore permuted to ``(out, kh, kw, channel)`` to match,
      which is a constant fold rather than runtime data movement.
    """

    def __init__(self, config: Glm5NextVisionConfig, device: DeviceRef) -> None:
        super().__init__()
        merge = config.spatial_merge_size
        self.out_dim = config.out_hidden_size
        self.weight = Weight(
            "weight",
            config.dtype,
            [config.out_hidden_size, config.hidden_size, merge, merge],
            device=device,
        )
        self.bias = Weight(
            "bias", config.dtype, [config.out_hidden_size], device=device
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Merges ``(num_tokens, merge * merge * hidden)`` rows.

        Args:
            x: Merge windows already flattened as ``(h, w, channel)``.

        Returns:
            ``(num_tokens, out_hidden_size)``.
        """
        weight = ops.permute(self.weight, [0, 2, 3, 1]).reshape(
            (self.out_dim, -1)
        )
        return x @ weight.transpose(0, 1).cast(x.dtype) + self.bias.to(
            x.device
        ).cast(x.dtype)


class Glm5NextVisionRotaryEmbedding(Module):
    """2D vision RoPE tables, built from per-patch ``(h, w)`` grid coordinates.

    There is no temporal axis: the two halves of the ``head_dim``-wide table
    carry the height and width angles respectively.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = head_dim // 2
        self.theta = theta

    def __call__(
        self, rot_pos_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Builds ``(cos, sin)`` of shape ``(seq_len, head_dim)``.

        Args:
            rot_pos_ids: ``(seq_len, 2)`` integer ``(h, w)`` grid coordinates.
        """
        device = rot_pos_ids.device
        # float64 keeps the exponent exact before the pow, matching the
        # reference's float32 `inv_freq`.
        iota = ops.range(
            0,
            self.dim,
            2,
            out_dim=self.dim // 2,
            device=device,
            dtype=DType.float64,
        )
        inv_freq = ops.cast(
            1.0 / (self.theta ** (iota / self.dim)), DType.float32
        )
        positions = rot_pos_ids.cast(DType.float32)
        # (seq_len, 2, 1) * (dim // 2,) flattened to (seq_len, dim), i.e. the
        # height angles followed by the width angles.
        freqs = ops.unsqueeze(positions, -1) * inv_freq
        freqs = freqs.reshape((freqs.shape[0], self.dim))
        emb = ops.concat((freqs, freqs), -1)
        return ops.cos(emb), ops.sin(emb)


class Glm5NextVisionBlock(Module):
    """Pre-norm block: RMSNorm, full attention, RMSNorm, clamped SwiGLU MLP."""

    def __init__(self, config: Glm5NextVisionConfig, device: DeviceRef) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            dim=config.hidden_size,
            dtype=config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.norm2 = RMSNorm(
            dim=config.hidden_size,
            dtype=config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.attn = Glm5NextVisionAttention(
            dtype=config.dtype,
            device=device,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            rms_norm_eps=config.rms_norm_eps,
            has_bias=config.attention_bias,
        )
        self.mlp = Glm5NextVisionMLP(
            hidden_dim=config.hidden_size,
            # The blocks' width, not the merger's. This is the field whose
            # meaning changed between GLM-4.6V and GLM-5.3-Flash.
            intermediate_dim=config.intermediate_size,
            # The reference passes `bias=config.attention_bias` here, so one
            # config field gates the biases on attention and on the block MLP.
            has_bias=config.attention_bias,
            swiglu_limit=config.swiglu_limit,
            dtype=config.dtype,
            device=device,
        )

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
    ) -> TensorValue:
        x = x + self.attn(
            self.norm1(x),
            position_embeddings=position_embeddings,
            input_row_offsets=input_row_offsets,
            max_seqlen=max_seqlen,
        )
        return x + self.mlp(self.norm2(x))


class Glm5NextVisionPatchMerger(Module):
    """Projects merged patches into the language model's embedding space."""

    def __init__(self, config: Glm5NextVisionConfig, device: DeviceRef) -> None:
        super().__init__()
        dim = config.out_hidden_size
        self.proj = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )
        # The tower's only LayerNorm; every other norm here is an RMSNorm, and
        # this one carries a bias.
        self.post_projection_norm = LayerNorm(
            dims=dim,
            devices=[device],
            dtype=config.dtype,
            eps=1e-5,
            use_bias=True,
        )
        self.swiglu_limit = config.swiglu_limit
        # GLM stores these as siblings of `merger.proj`, with no `mlp.` level,
        # so they are declared here rather than in a nested MLP module. Nesting
        # them would leave `merger.gate_proj.weight` unconsumed.
        self.gate_proj = Linear(
            in_dim=dim,
            out_dim=config.projection_intermediate_size,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )
        self.up_proj = Linear(
            in_dim=dim,
            out_dim=config.projection_intermediate_size,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )
        self.down_proj = Linear(
            in_dim=config.projection_intermediate_size,
            out_dim=dim,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.proj(x)
        # Exact GELU, not the tanh approximation.
        x = ops.gelu(self.post_projection_norm(x))
        return self.down_proj(
            clamped_swiglu(
                self.gate_proj(x), self.up_proj(x), self.swiglu_limit
            )
        )


class Glm5NextVisionTransformer(Module):
    """The GLM-5.3-Flash vision tower.

    Attention is segmented per frame by ``input_row_offsets``, exactly as the
    reference builds ``cu_seqlens``. A clip is never merged into a single
    attention span, which is what keeps per-frame vision-cache granularity
    bit-equivalent to encoding a whole clip at once.

    Tensor parallelism is not supported: the vision encoder cache assumes each
    device holds a full replica of the encoder output, so a sharded tower would
    need an all-gather before returning. At 0.56B parameters that replication is
    cheap.
    """

    def __init__(
        self,
        config: Glm5NextVisionConfig,
        device: DeviceRef,
        llm_dtype: DType | None = None,
    ) -> None:
        """
        Args:
            config: The tower's shapes and dtypes.
            device: Device every weight and activation lives on.
            llm_dtype: dtype the merger output is cast to before it is
                scattered onto the decoder's embeddings. Defaults to the
                tower's own dtype, which is what the real checkpoint wants:
                the tower and the decoder embeddings are both BF16.
        """
        super().__init__()
        self.config = config
        self.llm_dtype = llm_dtype if llm_dtype is not None else config.dtype
        self.patch_embed = Glm5NextVisionPatchEmbed(config, device)
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(
            head_dim=config.head_dim
        )
        self.blocks = LayerList(
            [Glm5NextVisionBlock(config, device) for _ in range(config.depth)]
        )
        self.post_layernorm = RMSNorm(
            dim=config.hidden_size,
            dtype=config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.downsample = Glm5NextVisionSpatialMerge(config, device)
        self.merger = Glm5NextVisionPatchMerger(config, device)

    def __call__(
        self,
        pixel_values: TensorValue,
        rot_pos_ids: TensorValue,
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
        stages: list[TensorValue] | None = None,
    ) -> TensorValue:
        """Encodes patches into language-model embeddings.

        Args:
            pixel_values: ``(total_patches, patch_dim)`` flattened patches, in
                the image processor's ``(channel, temporal, height, width)``
                order.
            rot_pos_ids: ``(total_patches, 2)`` ``(h, w)`` grid coordinates.
            input_row_offsets: ``(num_frames + 1,)`` uint32 prefix sum of
                per-frame patch counts -- one attention segment per frame.
            max_seqlen: Longest frame length, as a ``(1,)`` uint32 tensor.
            stages: If given, per-stage activations are appended for
                tensor-level comparison against the reference tower.

        Returns:
            ``(total_patches // spatial_merge_unit, out_hidden_size)``
            embeddings in the language model's dtype.
        """
        h = self.patch_embed(pixel_values)
        if stages is not None:
            stages.append(h)

        position_embeddings = self.rotary_pos_emb(rot_pos_ids)
        for block in self.blocks:
            h = block(
                h,
                position_embeddings=position_embeddings,
                input_row_offsets=input_row_offsets,
                max_seqlen=max_seqlen,
            )
            if stages is not None:
                stages.append(h)

        h = self.post_layernorm(h)
        merge_unit = _spatial_merge_unit(self.config)
        seq_len, hidden = h.shape
        # The patch count is a multiple of the merge unit by construction, but
        # a dynamic dimension does not carry that fact into the reshape.
        h = h.rebind([(seq_len // merge_unit) * merge_unit, hidden])
        h = self.downsample(h.reshape((-1, merge_unit * hidden)))
        if stages is not None:
            stages.append(h)

        out = self.merger(h)
        if stages is not None:
            stages.append(out)
        return out.cast(self.llm_dtype)
