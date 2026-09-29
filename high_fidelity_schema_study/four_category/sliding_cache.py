"""Release oversized backing allocations retained by cropped sliding-cache views.

The installed DynamicSlidingWindowLayer retains a slice of the entire prefill
allocation. Clone only the stored window; return the original full tensors to
the current attention call. No token, cache value or attention range changes.
"""
from types import MethodType

VERSION = 'clone-window/v1'


def compact_layer(layer):
    if getattr(layer, '_schema_compacted_update', False):
        return
    native = layer.update
    def update(self, *args, **kwargs):
        full = native(*args, **kwargs)
        if not self.record_past:
            for name in ('keys', 'values'):
                value = getattr(self, name)
                visible = value.numel() * value.element_size()
                if visible and value.untyped_storage().nbytes() > 2 * visible:
                    setattr(self, name, value.clone())
        return full
    layer.update = MethodType(update, layer)
    layer._schema_compacted_update = True


def install(model, profile):
    version = profile['runtime'].get('sliding_cache_compaction')
    if version is None:
        return
    if version != VERSION:
        raise ValueError('unsupported_sliding_cache_compaction')
    from transformers.cache_utils import DynamicSlidingWindowLayer
    model._schema_compaction_enabled = True
    def before_forward(module, args, kwargs):
        if not module._schema_compaction_enabled:
            return
        cache = kwargs.get('past_key_values')
        if cache is None:
            raise ValueError('sliding_compaction_requires_cache')
        for layer in cache.layers:
            if isinstance(layer, DynamicSlidingWindowLayer):
                compact_layer(layer)
    model.register_forward_pre_hook(before_forward, with_kwargs=True)
