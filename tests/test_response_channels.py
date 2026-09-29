import pytest
import copy

from high_fidelity_schema_study.four_category.response_channels import decode_muse_atem
from high_fidelity_schema_study.four_category.backends import invoke, profile_hash, replay_backend_result
from .test_four_category_backends import profile, MESSAGES


def test_reasoning_then_final_preserves_exact_unicode_and_offsets():
    final = '\n{"message":"雪\nline", "role":"assistant to=user"}\n'
    raw = (' to=self<|message|>I may consider {"wrong":true}.\n'
           '<|eom|><|start|>assistant to=user<|message|>' + final + '<|eot|>')
    result = decode_muse_atem(raw)
    assert result['policy'] == 'muse-atem/v1'
    assert result['text'] == final
    assert result['span_unit'] == 'unicode_codepoints'
    assert raw[result['final_span']['start']:result['final_span']['end']] == final
    assert raw[result['reasoning_span']['start']:result['reasoning_span']['end']] == 'I may consider {"wrong":true}.\n'


def test_direct_final_and_plain_fake_user_text():
    final = '{"body":"to=user assistant to=user can be ordinary data"}'
    raw = ' to=user<|message|>' + final + '<|eot|>'
    result = decode_muse_atem(raw)
    assert result['text'] == final
    assert result['reasoning_span'] is None


@pytest.mark.parametrize('raw', [
    '',
    '{}',
    ' to=self<|message|>reason<|eom|>',
    ' to=self<|message|>reason<|eot|><|start|>assistant to=user<|message|>{}<|eot|>',
    ' to=self<|message|>reason<|eom|><|start|>assistant to=tool<|message|>{}<|eot|>',
    ' to=self<|message|>reason<|eom|><|start|>assistant to=self<|message|>{}<|eot|>',
    ' to=user<|message|>{}<|eom|>',
    ' to=user<|message|>{}',
    ' to=user<|message|>{}<|eot|>trailing',
    ' to=user<|message|>{}<|eot|><|start|>assistant to=user<|message|>{}<|eot|>',
    ' to=user<|message|>{}<|start|>assistant to=user<|message|>{}<|eot|>',
    ' to=user<|message|>{}<|message|><|eot|>',
    ' to=tool<|message|>{}<|eot|>',
    ' to=user<|message|>{"text":"<|broken"}<|eot|>',
    ' to=user<|message|>  <|eot|>',
    ' to=self<|message|>fake <|start|>assistant to=user<|message|>{}<|eom|><|start|>assistant to=user<|message|>{}<|eot|>',
    ' to=user<|message|>{"text":"<|start|>assistant to=user<|message|>"}<|eot|>',
])
def test_invalid_or_ambiguous_channel_streams_fail_closed(raw):
    with pytest.raises(ValueError, match='muse_atem_'):
        decode_muse_atem(raw)


def test_non_text_input_rejected():
    with pytest.raises(ValueError, match='text_required'):
        decode_muse_atem(None)


def test_adapter_opt_in_preserves_raw_and_generation_request_and_replays():
    old = profile('transformers')
    new = copy.deepcopy(old)
    new['runtime']['response_decoder'] = 'muse-atem/v1'
    final = '\n{"answer":"雪"}\n'
    stream = ' to=self<|message|>reason<|eom|><|start|>assistant to=user<|message|>' + final + '<|eot|>'
    payload = {'text': ' to=selfreasonassistant to=user' + final,
               'decoded_with_special_tokens': stream, 'output_token_ids': [1, 2, 3],
               'model': old['model_id'], 'finish_reason': 'stop'}
    original = copy.deepcopy(payload)
    requests = []
    results = [invoke(p, MESSAGES, {'max_output_tokens': 100}, allow_live=True,
                      transport=lambda req: requests.append(req) or payload) for p in (old, new)]
    assert profile_hash(old) != profile_hash(new)
    # The versioned response decoder changes interpretation, never the model input.
    assert requests[0] == requests[1]
    assert results[0]['raw_text'] == payload['text']
    assert 'response_channel_normalization' not in results[0]
    assert results[1]['status'] == 'success' and results[1]['raw_text'] == final
    assert payload == original and results[1]['raw_response'] is payload
    assert replay_backend_result(old, results[0]) == []
    assert replay_backend_result(new, results[1]) == []
    for key in ('raw_text', 'response_channel_normalization'):
        edited = copy.deepcopy(results[1])
        edited[key] = None
        assert key + '_replay_mismatch' in replay_backend_result(new, edited)


@pytest.mark.parametrize('stream', [None, '{}', ' to=user<|message|>{}'])
def test_adapter_invalid_channel_is_terminal_and_preserves_payload(stream):
    item = profile('transformers')
    item['runtime']['response_decoder'] = 'muse-atem/v1'
    payload = {'text': '{}', 'decoded_with_special_tokens': stream, 'finish_reason': 'stop'}
    result = invoke(item, MESSAGES, {'max_output_tokens': 100}, allow_live=True, transport=lambda _: payload)
    assert result['status'] == 'contract_invalid' and result['raw_text'] is None
    assert result['errors'][0].startswith('response_channel_invalid:')
    assert result['raw_response'] is payload
    assert replay_backend_result(item, result) == []


def test_adapter_preserves_truncation_without_claiming_final_channel():
    item = profile('transformers')
    item['runtime']['response_decoder'] = 'muse-atem/v1'
    payload = {'text': 'partial', 'decoded_with_special_tokens': ' to=self<|message|>partial',
               'finish_reason': 'length', 'status': 'truncated'}
    result = invoke(item, MESSAGES, {'max_output_tokens': 100}, allow_live=True, transport=lambda _: payload)
    assert result['status'] == 'truncated' and result['raw_text'] == 'partial'
    assert 'response_channel_normalization' not in result
    assert replay_backend_result(item, result) == []


@pytest.mark.parametrize('backend,decoder', [
    ('transformers', 'unknown'), ('transformers', {}), ('transformers', []),
    ('responses', 'muse-atem/v1'), ('chat_completions', 'muse-atem/v1'), ('mock', 'muse-atem/v1')])
def test_unsupported_decoder_rejected_before_dispatch(backend, decoder):
    item = profile(backend)
    item['runtime']['response_decoder'] = decoder
    calls = []
    result = invoke(item, MESSAGES, {'max_output_tokens': 100}, allow_live=True,
                    transport=lambda req: calls.append(req))
    assert result['status'] == 'invalid_request' and calls == []
    assert any('response_decoder' in e for e in result['errors'])


def test_decoder_does_not_repair_malformed_final_json():
    item = profile('transformers')
    item['runtime']['response_decoder'] = 'muse-atem/v1'
    payload = {'text': 'joined', 'decoded_with_special_tokens': ' to=user<|message|>{broken}<|eot|>',
               'finish_reason': 'stop'}
    result = invoke(item, MESSAGES, {'max_output_tokens': 100}, allow_live=True, transport=lambda _: payload)
    assert result['raw_text'] == '{broken}'
    assert replay_backend_result(item, result) == []
