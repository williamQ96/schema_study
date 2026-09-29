"""Pure decoders for versioned model response channel protocols."""
from __future__ import annotations

import re


POLICY = 'muse-atem/v1'
SELF_HEADER = ' to=self<|message|>'
USER_HEADER = ' to=user<|message|>'
NEXT_USER_HEADER = '<|start|>assistant to=user<|message|>'
EOM = '<|eom|>'
EOT = '<|eot|>'
SPECIAL_TOKEN = re.compile(r'<\|[^<>]*?\|>')


def _next_token(raw: str, start: int) -> tuple[str, int, int]:
    match = SPECIAL_TOKEN.search(raw, start)
    if match is None:
        raise ValueError('muse_atem_missing_terminator')
    if (partial := raw.find('<|', start)) != -1 and partial < match.start():
        raise ValueError('muse_atem_malformed_special_token')
    return match.group(), match.start(), match.end()


def decode_muse_atem(raw_response: str) -> dict:
    """Extract only a structurally final answer from the frozen Muse ATEM stream.

    Offsets count Python Unicode code points in the original decoded response.
    The caller retains that response as the auditable source; this function never
    parses, repairs, or searches for JSON in the reasoning channel.
    """
    if not isinstance(raw_response, str):
        raise ValueError('muse_atem_response_text_required')

    reasoning_span = None
    if raw_response.startswith(SELF_HEADER):
        reasoning_start = len(SELF_HEADER)
        token, reasoning_end, token_end = _next_token(raw_response, reasoning_start)
        if token != EOM:
            raise ValueError('muse_atem_reasoning_terminator_invalid')
        reasoning_span = {'start': reasoning_start, 'end': reasoning_end}
        if not raw_response.startswith(NEXT_USER_HEADER, token_end):
            raise ValueError('muse_atem_final_header_missing_or_invalid')
        final_start = token_end + len(NEXT_USER_HEADER)
    elif raw_response.startswith(USER_HEADER):
        final_start = len(USER_HEADER)
    else:
        raise ValueError('muse_atem_initial_header_invalid')

    token, final_end, token_end = _next_token(raw_response, final_start)
    if token != EOT:
        raise ValueError('muse_atem_final_terminator_invalid')
    if token_end != len(raw_response):
        raise ValueError('muse_atem_trailing_content')
    if not raw_response[final_start:final_end].strip():
        raise ValueError('muse_atem_empty_final')

    return {'policy': POLICY, 'text': raw_response[final_start:final_end],
            'final_span': {'start': final_start, 'end': final_end},
            'reasoning_span': reasoning_span, 'span_unit': 'unicode_codepoints'}
