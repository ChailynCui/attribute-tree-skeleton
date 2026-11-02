# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from einops import rearrange, pack
from x_transformers.autoregressive_wrapper import top_k

from tqdm import tqdm

from tree_skeleton._tensor_ops import divisible_by, exists, default


class SkeletonSampler:
    """Autoregressive sampling and segment-complete termination."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        prompt=None,
        batch_size=None,
        species_id=None,
        attribute_ids=None,
        attribute_values=None,
        text_embeddings=None,
        filter_logits_fn=top_k,
        filter_kwargs=dict(),
        temperature=1.0,
        return_codes=False,
        cond_scale=1.0,
        cache_kv=True,
        max_seq_len=None,
        min_seq_len=0,
        planned_num_segments=None,
        segment_coords_to_file=None,
    ):
        model = self.model
        max_seq_len = default(max_seq_len, model.max_seq_len)
        max_seq_len = int(max_seq_len)
        min_seq_len = int(min_seq_len)
        tokens_per_segment = model.num_quantizers * 2
        if max_seq_len > model.max_seq_len:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) exceeds model limit ({model.max_seq_len})"
            )
        if min_seq_len < 0:
            raise ValueError("min_seq_len must be non-negative")
        if min_seq_len > max_seq_len:
            raise ValueError("min_seq_len cannot exceed max_seq_len")
        if not divisible_by(min_seq_len, tokens_per_segment):
            raise ValueError(
                f"min_seq_len ({min_seq_len}) must be divisible by "
                f"2 x num_quantizers ({tokens_per_segment})"
            )
        if exists(prompt):
            prompt_batch = int(prompt.shape[0])
            if exists(batch_size) and int(batch_size) != prompt_batch:
                raise ValueError("batch_size does not match prompt batch")
            batch_size = prompt_batch
        batch_size = int(default(batch_size, 1))

        codes = default(prompt, torch.empty((batch_size, 0), dtype=torch.long, device=model.device))
        codes = codes.to(model.device)

        curr_length = codes.shape[-1]

        # A1 uses the same planner for every species.  No species-specific EOS
        # floor is applied: each sample stops at its predicted conditional length.
        if model.length_progress_conditioning and not exists(planned_num_segments):
            planned_num_segments = model.predict_num_segments(
                batch_size=batch_size,
                species_id=species_id,
                attribute_ids=attribute_ids,
                attribute_values=attribute_values,
                text_embeddings=text_embeddings,
            )

        planned = None
        target_code_lengths = None
        generation_stop = max_seq_len
        if exists(planned_num_segments):
            planned = (
                model.prepare_planned_num_segments(
                    planned_num_segments,
                    batch_size,
                    model.device,
                )
                .round()
                .long()
            )
            effective_max_segments = max_seq_len // tokens_per_segment
            planned = planned.clamp(1, effective_max_segments)
            target_code_lengths = planned * tokens_per_segment
            if bool((target_code_lengths < curr_length).any()):
                raise ValueError("prompt is longer than planned generation length")
            # One additional prediction emits EOS at the exact segment boundary.
            generation_stop = int(target_code_lengths.max().item()) + 1

        cache = None
        is_eos_codes = codes == model.eos_token_id

        for i in tqdm(range(curr_length, generation_stop)):
            if exists(target_code_lengths):
                already_finished = is_eos_codes.any(dim=-1)
                force_eos = already_finished | (i >= target_code_lengths)
                can_eos = force_eos
            else:
                can_eos = i != 0 and i >= min_seq_len and divisible_by(i, tokens_per_segment)

            output = model.forward_on_codes(
                codes,
                return_loss=False,
                return_cache=cache_kv,
                append_eos=False,
                species_id=species_id,
                attribute_ids=attribute_ids,
                attribute_values=attribute_values,
                text_embeddings=text_embeddings,
                cache=cache,
                planned_num_segments=planned,
            )

            if cache_kv:
                logits, cache = output
            else:
                logits = output

            logits = logits[:, -1]

            if torch.is_tensor(can_eos):
                prohibit_eos = ~can_eos
                logits[prohibit_eos, -1] = -torch.finfo(logits.dtype).max
                if bool(force_eos.any()):
                    logits[force_eos] = -torch.finfo(logits.dtype).max
                    logits[force_eos, -1] = 0.0
            elif not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.0:
                sample = filtered_logits.argmax(dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            if sample.ndim == 1:
                sample = rearrange(sample, "b -> b 1")

            codes, _ = pack([codes, sample], "b *")

            is_eos_codes = codes == model.eos_token_id

            if is_eos_codes.any(dim=-1).all():
                break

        mask = is_eos_codes.float().cumsum(dim=-1) >= 1
        codes = codes.masked_fill(mask, model.pad_id)

        code_len = codes.shape[-1]
        round_down_code_len = code_len // model.num_quantizers * model.num_quantizers
        codes = codes[:, :round_down_code_len]

        if return_codes:
            codes = rearrange(codes, "b (n q) -> b n q", q=model.num_quantizers)
            return codes

        model.tokenizer.eval()

        segment_coords, segment_mask = model.tokenizer.codes_to_segments(codes)

        if not exists(segment_coords_to_file):
            return segment_coords, segment_mask

        files = [
            segment_coords_to_file(coords[mask])
            for coords, mask in zip(segment_coords, segment_mask)
        ]

        return files
