"""Explicit Flex compilation limits and rejection of dense eager fallback."""
import warnings

VERSION = 'flex-compiled-only/v1'
_defaults = None
_message = r'flex_attention called without torch\.compile\(\).*'


def configure(profile, torch):
    global _defaults
    config = torch._dynamo.config
    keys = ('recompile_limit', 'accumulated_recompile_limit')
    if _defaults is None:
        _defaults = {k: getattr(config, k) for k in keys}
    policy = profile['runtime'].get('compile_policy')
    if policy is None:
        values = _defaults
    else:
        if (not isinstance(policy, dict) or set(policy) != {'version', *keys}
                or policy['version'] != VERSION
                or any(isinstance(policy[k], bool) or not isinstance(policy[k], int)
                       or not 1 <= policy[k] <= 4096 for k in keys)
                or policy['accumulated_recompile_limit'] < policy['recompile_limit']):
            raise ValueError('invalid_compile_policy')
        values = policy
    for key in keys:
        setattr(config, key, values[key])
    # Dense eager Flex materializes an O(sequence_length**2 * heads) tensor.
    # A cache limit must produce an explicit failure rather than that fallback.
    warnings.filterwarnings('error' if policy is not None else 'default', message=_message, category=UserWarning)


def observe(torch):
    return {k: getattr(torch._dynamo.config, k) for k in ('recompile_limit', 'accumulated_recompile_limit')}
