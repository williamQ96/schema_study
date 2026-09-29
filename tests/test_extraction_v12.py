from __future__ import annotations

import copy
import json
import zipfile
import pytest
from high_fidelity_schema_study.four_category import extraction_v10, extraction_v12 as v12
from high_fidelity_schema_study.four_category import name_mapping_v2, fidelity_v2
from high_fidelity_schema_study.four_category.common import file_digest, seal
from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
from high_fidelity_schema_study.four_category.feature_regions_v2 import regions
from .test_fidelity_validation import table_paper, dataset, payload


def grouped():
    p = table_paper()
    t = p['pages'][0]['tables'][0]
    t['caption']['text'] = 'Table 7. Features used in the study'
    t['columns'] = [{'column_index': 0, 'header': ''}]
    t['header_units'] = []
    t['rows'] = [{'row_id': 'r0', 'unit_id': 'r0', 'text': 'Main features Area Perimeter',
                  'cells': [{'unit_id': 'c0', 'text': 'Main features Area Perimeter', 'column_index': 0}]}]
    return p


def response(p):
    g = v12.plan_region(p, [1]); idx = make_disabled_index(p)
    w = next(w['window_index'] for w in extraction_v10.window_catalog(p) if w['unit_id'] == 'c0')
    anchor = next(a['anchor_id'] for a in v12.anchors(p, g) if a['window_id'] == w)
    fact = {'primary_anchor': anchor, 'support_windows': [], 'categories': ['structure'],
            'claim': {'kind': 'attribute', 'predicate': 'reported_name',
                      'assertion': {'status': 'reported', 'value': 'Area', 'basis': None}}}
    value = {'schema_version': v12.RESPONSE_VERSION, 'coverage': [{'window_id': i, 'state': 'reviewed'} for i in g['window_ids']],
             'objects': [{'kind': 'field', 'normalized_label': n, 'context': None, 'source_windows': [w],
                          'facts': [copy.deepcopy(fact)] if n == 'Area' else []} for n in ['Area', 'Perimeter']],
             'table_reviews': [{'table_id': 't1', 'purpose': 'dataset_schema', 'field_axis': 'grouped_cells',
                                'rationale': 'Multiple feature names in one source region.', 'source_windows': [w]}],
             'feature_coverage': [{'unit_id': 'c0', 'decision': 'complete', 'emitted_labels': ['Area', 'Perimeter'],
                                   'rationale': 'Both named features enumerated; Main features is a heading.'}]}
    return value, idx, g


def test_nested_facts_cannot_use_source_windows_as_object_ids():
    p = grouped(); value, idx, g = response(p)
    assert v12.validate_response(value, p, idx, g) == []
    result = v12.admit_response(json.dumps(value), p, idx, g)
    assert result['status'] == 'success'
    assert result['materialized_v10_payload']['facts'][0]['subject_mention'] == 0
    value['objects'][0]['facts'][0]['subject_mention'] = 65
    assert v12.validate_response(value, p, idx, g)


def test_multi_feature_cells_and_false_empty_completeness():
    p = grouped(); value, idx, g = response(p)
    assert regions(p)[0]['grouped_inventory_candidate']
    assert regions(p)[0]['candidate_inventory_exhaustive'] is False
    value['objects'] = []
    value['feature_coverage'][0].update(decision='not_field', emitted_labels=[])
    value['table_reviews'][0].update(purpose='other', field_axis='neither')
    result = v12.admit_response(json.dumps(value), p, idx, g)
    assert result['status'] == 'incomplete'
    assert 'feature_inventory_without_fields:t1' in result['completeness_reasons']


def test_coverage_labels_require_objects_and_cell_evidence():
    p = grouped(); value, idx, g = response(p)
    value['feature_coverage'][0]['emitted_labels'].append('Humidity')
    assert any('without_bound_object' in e for e in v12.validate_response(value, p, idx, g))
    value, idx, g = response(p)
    value['feature_coverage'][0].update(decision='complete', emitted_labels=[])
    assert any('complete_region_without_objects' in e for e in v12.validate_response(value, p, idx, g))


def test_uncertainty_and_limits_are_not_success():
    p = grouped(); value, idx, g = response(p)
    value['feature_coverage'][0]['decision'] = 'partial'
    assert v12.admit_response(json.dumps(value), p, idx, g)['status'] == 'incomplete'
    value, idx, g = response(p)
    value['coverage'][0]['state'] = 'uncertain'
    assert v12.admit_response(json.dumps(value), p, idx, g)['status'] == 'incomplete'


def test_link_target_remains_named_evidence_object():
    p = grouped(); value, idx, g = response(p)
    fact = value['objects'][0]['facts'][0]
    fact['claim'] = {'kind': 'link', 'predicate': 'parent', 'target': {'kind': 'table',
        'normalized_label': 'Feature inventory', 'source_windows': value['objects'][0]['source_windows']},
        'assertion': {'status': 'inferred', 'value': 'parent', 'basis': 'Table membership'}}
    assert v12.validate_response(value, p, idx, g) == []
    fact['claim']['target'] = 65
    assert v12.validate_response(value, p, idx, g)


def mapping_fixture(dataset):
    _, cat, root = dataset
    with zipfile.ZipFile(root/'data.zip', 'w') as z:
        z.write(root/'data.csv', 'data.csv')
        z.writestr('Readme.txt', '- hum: Normalized humidity. Values divided by 100.\n- code: if a value is present\n')
    spec = {'catalog_sha256': cat['catalog_sha256'], 'source_url': 'https://example.org/test-fixture',
            'archive_path': 'data.zip', 'archive_file_bytes_sha256': file_digest(root/'data.zip'),
            'codebook_member': 'Readme.txt', 'scope_members': [{'scope_id': 'main', 'dataset_member': 'data.csv'}]}
    return spec, cat, root


def test_mapping_replays_declaration_dataset_and_rejects_resealed_edit(dataset):
    spec, cat, root = mapping_fixture(dataset)
    doc = name_mapping_v2.derive(spec, cat, root)
    rows = name_mapping_v2.verify(doc, cat, root)
    match = name_mapping_v2.match('humidity', cat, rows)
    assert match['status'] == 'matched' and match['mapping_evidence']
    assert name_mapping_v2.match('code correction', cat, rows)['status'] == 'unresolved'
    doc['entries'][0]['alias'] = 'temperature'
    with pytest.raises(ValueError, match='derivation'):
        name_mapping_v2.verify(seal(doc, 'mapping_sha256'), cat, root)
    with zipfile.ZipFile(root/'bad.zip', 'w') as z:
        z.writestr('data.csv', 'forged')
        z.writestr('Readme.txt', '- hum: humidity')
    spec.update(archive_path='bad.zip', archive_file_bytes_sha256=file_digest(root/'bad.zip'))
    with pytest.raises(ValueError, match='member_identity'):
        name_mapping_v2.derive(spec, cat, root)


def test_grouped_evidence_is_not_permission_to_correct_names(dataset):
    _, cat, _ = dataset
    p = grouped()
    good = fidelity_v2.evaluate(payload(p, 'Area', 'c0'), p, cat, admission_status='success')
    assert good['automatic_supported_object_ids']
    bad = fidelity_v2.evaluate(payload(p, 'hum', 'c0'), p, cat, admission_status='success')
    assert not bad['automatic_supported_object_ids']
    assert bad['formal_precision'] is None


def test_prompt_is_paper_only_and_new_task_does_not_mutate_v11():
    from high_fidelity_schema_study.four_category import extraction_v11
    p = grouped(); _, idx, g = response(p)
    before = extraction_v11.make_task()
    messages = v12.render_group(v12.make_task(), p, idx, g)
    body = json.loads(messages[1]['content'])
    assert set(body) == {'task_sha256', 'target_schema', 'full_source', 'target', 'auxiliary_hints'}
    assert 'objects' in body['target_schema']['properties']
    assert 'facts' not in body['target_schema']['properties']
    assert before == extraction_v11.make_task()
