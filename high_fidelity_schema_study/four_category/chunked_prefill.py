"""Text-only native-prefill adaptation for multi-axis position IDs.

Transformers 5.16.1 slices position_ids[:, start:end] during chunked prefill.
For Qwen's [axes, batch, sequence] positions that slices batch, not sequence.
This opt-in adapter preserves native model forward/cache handling and slices
the last axis. Fresh text-only generation is the only supported workload.
"""
from types import MethodType

VERSION = 'text-last-axis/v1'


def install(model, profile):
    version = profile['runtime'].get('chunked_prefill_adapter')
    if version is None:
        return
    if version != VERSION:
        raise ValueError('unsupported_chunked_prefill_adapter')
    native = model._prefill

    def prefill(self, input_ids, generation_config, model_kwargs, is_first_iteration=True):
        size = generation_config.prefill_chunk_size
        if size is None:
            return native(input_ids, generation_config, model_kwargs, is_first_iteration=is_first_iteration)
        cache = model_kwargs.get('past_key_values')
        if (not is_first_iteration or cache is None or cache.get_seq_length() != 0
                or model_kwargs.get('inputs_embeds') is not None
                or any(model_kwargs.get(k) is not None for k in ('pixel_values', 'pixel_values_videos', 'input_features'))):
            raise ValueError('last_axis_prefill_requires_fresh_text_cache')
        attention_mask = model_kwargs.pop('attention_mask', None)
        position_ids = model_kwargs.pop('position_ids', None)
        past_length = 0
        try:
            for chunk in input_ids.split(size, dim=-1):
                current_length = past_length + chunk.shape[-1]
                if attention_mask is not None:
                    model_kwargs['attention_mask'] = attention_mask[..., :current_length]
                if position_ids is not None:
                    model_kwargs['position_ids'] = position_ids[..., past_length:current_length]
                prepared = self.prepare_inputs_for_generation(chunk, **model_kwargs)
                outputs = self(**prepared, return_dict=True)
                model_kwargs['past_key_values'] = outputs.past_key_values
                past_length = current_length
            return outputs
        finally:
            model_kwargs['attention_mask'] = attention_mask
            model_kwargs['position_ids'] = position_ids

    model._prefill = MethodType(prefill, model)
