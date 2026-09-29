"""Compose replaceable model profiles and qualify a Mercury batch before dispatch.

No model names or architecture allowlist live here. Source tasks remain unchanged.
The deployment identity augments (never replaces) the existing run/packet binding.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

from .backends import preflight_parameters, profile_hash, validate_profile
from .common import ROOT, digest, file_digest, now, read_json, write_new
from .workflow import experiment_errors, run_batch, tasks_for_config


def code_identity(root=None) -> dict:
    """Bind active source, not the Git HEAD of an often intentionally dirty checkout."""
    root = ROOT if root is None else Path(root)
    paths = set(root.glob('*.py'))
    for directory in ('four_category', 'extractors'):
        paths.update((root / directory).rglob('*.py'))
    paths.update((root / 'templates').glob('four_category*'))
    for name in ('paper_to_schema_system_v4.txt', 'paper_to_schema_user_v4.txt',
                 'paper_to_schema_user_v5.txt',
                 'paper_to_schema_system_v6.txt', 'paper_to_schema_user_v6.txt',
                 'paper_extraction_observations_v5.schema.json',
                 'paper_to_schema_system_v7.txt', 'paper_to_schema_user_v7.txt',
                 'paper_extraction_observations_v6.schema.json',
                 'paper_to_schema_system_v8.txt', 'paper_to_schema_user_v8.txt',
                 'paper_extraction_observations_v7.schema.json',
                 'paper_to_schema_system_v9.txt', 'paper_to_schema_user_v9.txt',
                 'paper_extraction_observations_v9.schema.json',
                 'paper_to_schema_system_v10.txt', 'paper_to_schema_user_v10.txt',
                 'paper_extraction_observations_v10.schema.json',
                 'paper_category_response_v1.schema.json', 'paper_category_response_v2.schema.json', 'paper_category_response_v3.schema.json', 'paper_derived_schema_v4.schema.json',
                 'paper_evidence_input_v3.schema.json', 'paper_evidence_layout_v3.schema.json',
                 'paper_evidence_input_v4.schema.json', 'paper_evidence_layout_v4.schema.json',
                 'paper_evidence_input_v5.schema.json', 'paper_evidence_layout_v5.schema.json',
                 'paper_evidence_input_v6.schema.json', 'paper_evidence_layout_v6.schema.json'):
        paths.add(root / 'templates' / name)
    return {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths) if p.is_file()}


def parameter_errors(config: dict) -> list[str]:
    profiles = {p['profile_id']: p for p in config['profiles']}
    requests = [(config['classification']['profile_id'], config['classification']['parameters'])]
    requests.extend((pid, {**config['local_parameters'], 'seed': rep['seed']})
                    for pid in config['roles']['locals'] for rep in config['replicates'])
    requests.append((config['roles']['soft_reference'], config['reference_parameters']))
    errors = {f'{pid}:{error}' for pid, params in requests
              for error in preflight_parameters(profiles[pid], params)}
    for pid, params in requests:
        allowance = params.get('max_output_tokens') if isinstance(params, dict) else None
        window = profiles[pid].get('context_window')
        if isinstance(allowance, int) and isinstance(window, int) and window <= allowance:
            errors.add(f'{pid}:context_window_cannot_fit_output_plus_nonempty_input')
    from .token_counting import counter_config_errors
    for pid, profile in profiles.items():
        if profile['backend'] in ('responses', 'chat_completions'):
            runtime = profile['runtime']
            if 'token_counter' in runtime:
                errors.update(f'{pid}:{e}' for e in counter_config_errors(profile))
            elif not runtime.get('token_count_observations'):
                errors.add(f'{pid}:explicit_http_token_counting_required')
    return sorted(errors)


def checkpoint_binding_errors(config: dict, *, verify_bytes: bool = False) -> list[str]:
    """A revision string alone cannot identify a mutable local weight directory."""
    errors = []
    for profile in config['profiles']:
        if profile['backend'] != 'transformers':
            continue
        pid = profile['profile_id']
        binding = profile['runtime'].get('checkpoint_manifest')
        if (not isinstance(binding, dict) or not isinstance(binding.get('path'), str)
                or not isinstance(binding.get('file_bytes_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', binding['file_bytes_sha256'])):
            errors.append(pid + ':checkpoint_manifest_binding_required')
            continue
        if verify_bytes:
            from .checkpoints import verify_manifest
            try:
                path = Path(binding['path'])
                if file_digest(path) != binding['file_bytes_sha256']:
                    errors.append(pid + ':checkpoint_manifest_file_identity_mismatch')
                    continue
                errors.extend(pid + ':' + e for e in verify_manifest(
                    read_json(path), Path(profile['model_id']), expected_revision=profile['revision']))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                errors.append(pid + ':checkpoint_verification:' + str(exc))
    return errors


def compose(base: dict, catalog: dict, selection: dict, host: dict, *, freeze: bool = False,
            image_sha256: str | None = None) -> tuple[dict, dict]:
    """Create a NEW condition. A classifier pin always refers to the catalog profile.

    Resource placement is part of the resulting effective profile, whose new hash
    is independently pinned in the experiment. No implicit classifier rebinding.
    """
    if catalog.get('schema_version') != 'four-category-model-catalog/v1':
        raise ValueError('unsupported_model_catalog')
    if selection.get('schema_version') != 'mercury-selection/v1' or host.get('schema_version') != 'mercury-host/v1':
        raise ValueError('unsupported_selection_or_host')
    rows = catalog.get('profiles')
    if not isinstance(rows, list) or not rows or any(not isinstance(p, dict) for p in rows):
        raise ValueError('catalog_profiles_required')
    ids = [p.get('profile_id') for p in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('duplicate_or_invalid_catalog_profile_id')
    profiles = dict(zip(ids, rows))
    local = selection.get('locals')
    if not isinstance(local, list) or len(local) != 3 or any(not isinstance(i, str) for i in local) or len(set(local)) != 3:
        raise ValueError('three_distinct_local_profiles_required')
    classifier = selection.get('classifier', {})
    if not isinstance(classifier, dict):
        raise ValueError('classifier_binding_must_be_object')
    classifier_id = classifier.get('profile_id')
    reference = selection.get('soft_reference')
    selected = list(dict.fromkeys(local + [reference, classifier_id]))
    if any(i not in profiles for i in selected):
        raise ValueError('unbound_profile_selection')
    if reference in local:
        raise ValueError('soft_reference_must_be_separate')
    pin = classifier.get('profile_sha256')
    expected_pin = profile_hash(profiles[classifier_id])
    if pin is not None and pin != expected_pin:
        raise ValueError('classifier_catalog_pin_mismatch')
    if freeze and pin != expected_pin:
        raise ValueError('explicit_classifier_catalog_pin_required')
    preset_id = selection.get('resource_preset')
    presets = host.get('resource_presets', {})
    if not isinstance(presets, dict):
        raise ValueError('resource_presets_must_be_object')
    preset = presets.get(preset_id)
    if not isinstance(preset, dict):
        raise ValueError('unknown_resource_preset')
    gpu_count = preset.get('gpu_count')
    if isinstance(gpu_count, bool) or gpu_count not in (1, 2, 4):
        raise ValueError('resource_gpu_count_must_be_1_2_or_4')
    placement = preset.get('transformers_settings', {})
    if (not isinstance(placement, dict) or set(placement) != {'device_map', 'max_memory'}
            or not isinstance(placement['max_memory'], dict)
            or set(placement['max_memory']) != {str(i) for i in range(gpu_count)} | {'cpu'}):
        raise ValueError('preset_requires_relative_visible_gpu_memory_budgets')
    config = copy.deepcopy(base)
    config['experiment_id'] = selection.get('experiment_id')
    if not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip() or config['experiment_id'] == base.get('experiment_id'):
        raise ValueError('new_nonempty_experiment_id_required')
    config['profiles'] = []
    sources = code_identity()
    for pid in selected:
        profile = copy.deepcopy(profiles[pid])
        if profile.get('backend') == 'transformers':
            settings = profile.setdefault('runtime', {}).setdefault('settings', {})
            for key, value in placement.items():
                if key in settings and settings[key] != value:
                    raise ValueError(f'{pid}:profile_resource_preset_conflict:{key}')
                settings[key] = copy.deepcopy(value)
        # Job/cache identities include the actual runtime condition, not only
        # the requested model name. A new SIF or source revision cannot reuse a
        # generation produced by another deployment under the same local path.
        profile.setdefault('runtime', {})['deployment_identity'] = {
            'image_file_bytes_sha256': image_sha256,
            'source_tree_canonical_sha256': digest(sources),
            'resource_preset_sha256': digest(preset),
        }
        errors = validate_profile(profile, for_execution=freeze)
        if freeze and any('REPLACE' in str(profile.get(k, '')).upper() for k in ('model_id', 'revision')):
            errors.append('replace_placeholder_model_identity_before_freeze')
        if errors:
            raise ValueError(f'{pid}: {errors}')
        config['profiles'].append(profile)
    config['roles'] = {'locals': local, 'soft_reference': reference}
    effective = next(p for p in config['profiles'] if p['profile_id'] == classifier_id)
    config['classification']['profile_id'] = classifier_id
    # Missing catalog pin remains nonoperational even for an otherwise valid draft.
    config['classification']['profile_sha256'] = profile_hash(effective) if pin else None
    parameters = selection.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('parameters_must_be_object')
    if set(parameters) - {'local', 'reference', 'classification'}:
        raise ValueError('unknown_parameter_override')
    for role, params in parameters.items():
        if role == 'classification':
            config['classification']['parameters'] = copy.deepcopy(params)
        else:
            config[role + '_parameters'] = copy.deepcopy(params)
    if 'replicates' in selection:
        config['replicates'] = copy.deepcopy(selection['replicates'])
    config['status'] = 'frozen' if freeze else 'draft'
    config['inference_enabled'] = freeze
    tasks = tasks_for_config(config)
    hashes = {kind: task['task_sha256'] for kind, task in tasks.items()}
    if config.get('task_hashes') != hashes:
        raise ValueError('base_task_hashes_do_not_match_current_sources')
    config['notes'] = ['Mercury deployment condition; model and hardware qualification are separate from semantic evaluation.',
                       'Composed from versioned inputs; serial execution; no automatic model substitution.']
    if image_sha256 is not None and not re.fullmatch('[0-9a-f]{64}', image_sha256):
        raise ValueError('image_sha256_must_be_lowercase_file_bytes_sha256')
    if freeze and image_sha256 is None:
        raise ValueError('frozen_deployment_requires_actual_sif_sha256')
    config['deployment'] = {
        'schema_version': 'mercury-deployment/v1', 'host_id': host.get('host_id'),
        'host_spec_sha256': digest(host), 'resource_preset': preset_id,
        'resources': copy.deepcopy(preset),
        'required_gpu_count': gpu_count if any(p['backend'] == 'transformers' for p in config['profiles']) else 0,
        'base_experiment_sha256': digest(base), 'catalog_sha256': digest(catalog),
        'selection_sha256': digest(selection), 'classifier_catalog_sha256': pin,
        'image_file_bytes_sha256': image_sha256, 'source_file_bytes_sha256': sources,
    }
    structural = experiment_errors(config)
    if structural:
        raise ValueError(structural)
    readiness = experiment_errors(config, execution=True) + parameter_errors(config) + checkpoint_binding_errors(config)
    if freeze and readiness:
        raise ValueError(readiness)
    report = {'schema_version': 'mercury-compose-report/v1', 'status': 'pass',
              'experiment_sha256': digest(config), 'configuration_ready': not readiness,
              'execution_ready': False, 'runtime_qualification': 'pending_image_hardware_checkpoint_and_request_checks',
              'readiness_errors': readiness, 'classifier_catalog_sha256': expected_pin,
              'effective_profile_sha256': {p['profile_id']: profile_hash(p) for p in config['profiles']},
              'task_hashes': hashes, 'live_requests': 0}
    return config, report


def deployment_errors(config: dict, *, image_sha256: str | None = None) -> list[str]:
    deployment = config.get('deployment', {})
    errors = []
    if deployment.get('schema_version') != 'mercury-deployment/v1':
        return ['mercury_deployment_binding_missing']
    if not image_sha256 or image_sha256 != deployment.get('image_file_bytes_sha256'):
        errors.append('actual_sif_identity_mismatch_or_missing')
    if deployment.get('source_file_bytes_sha256') != code_identity():
        errors.append('active_source_identity_mismatch')
    return errors


def resource_errors(config: dict, observation: dict) -> list[str]:
    """A placement budget may not exceed the device's observed physical memory.

    Free memory and KV cache are workload dependent; this is a necessary check,
    not an OOM guarantee or a GPU reservation mechanism.
    """
    errors = []
    devices = observation.get('torch', {}).get('cuda', {}).get('devices', [])
    capacities = {str(d['index']): d.get('memory_total_bytes') for d in devices}
    capacities['cpu'] = observation.get('memory', {}).get('total_bytes')
    for profile in config['profiles']:
        if profile['backend'] != 'transformers':
            continue
        for device, value in profile['runtime'].get('settings', {}).get('max_memory', {}).items():
            if isinstance(value, int):
                budget = value
            else:
                amount, unit = re.fullmatch(r'([0-9.]+)(GiB|MiB|GB|MB)', value).groups()
                budget = float(amount) * {'GiB': 1024**3, 'MiB': 1024**2, 'GB': 10**9, 'MB': 10**6}[unit]
            capacity = capacities.get(device)
            if capacity is not None and budget > capacity:
                errors.append(f'{profile["profile_id"]}:placement_budget_exceeds_observed_capacity:{device}')
    return sorted(set(errors))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compose_parser = commands.add_parser('compose', help='Compose a new draft or explicitly frozen experiment')
    for name in ('base', 'catalog', 'selection', 'host', 'output'):
        compose_parser.add_argument('--' + name, type=Path, required=True)
    compose_parser.add_argument('--freeze', action='store_true')
    compose_parser.add_argument('--image-sha256')
    compose_parser.add_argument('--report', type=Path)
    catalog_parser = commands.add_parser('catalog', help='Inspect profile identities; no inference')
    catalog_parser.add_argument('--catalog', type=Path, required=True)
    checkpoint = commands.add_parser('checkpoint', help='Hash a complete local model directory, without loading weights')
    checkpoint.add_argument('--model-path', type=Path, required=True)
    checkpoint.add_argument('--revision', required=True)
    checkpoint.add_argument('--output', type=Path, required=True)
    doctor = commands.add_parser('doctor', help='Observe hardware and test CUDA allocation, without loading weights')
    doctor.add_argument('--gpu-count', type=int, choices=(0, 1, 2, 4), required=True)
    doctor.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='Verify deployment/hardware, then invoke the existing resumable runner')
    for name in ('config', 'corpus', 'source-root', 'output'):
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--allow-live', action='store_true')
    run.add_argument('--max-jobs', type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == 'compose':
            config, result = compose(read_json(args.base), read_json(args.catalog), read_json(args.selection),
                                     read_json(args.host), freeze=args.freeze, image_sha256=args.image_sha256)
            write_new(args.output, config)
            if args.report:
                write_new(args.report, result)
        elif args.command == 'catalog':
            rows = read_json(args.catalog)['profiles']
            result = {'profiles': [{'profile_id': p['profile_id'], 'profile_sha256': profile_hash(p),
                                    'errors': validate_profile(p)} for p in rows]}
        elif args.command == 'checkpoint':
            from .checkpoints import build_manifest
            if args.output.resolve().is_relative_to(args.model_path.resolve()):
                raise ValueError('checkpoint_manifest_must_be_outside_weight_directory')
            result = build_manifest(args.model_path, args.revision)
            write_new(args.output, result)
            result = {'status': 'pass', 'manifest': str(args.output),
                      'file_bytes_sha256': file_digest(args.output), 'files': len(result['files'])}
        else:
            from .hardware import collect_hardware, assess_hardware
            if args.command == 'run':
                config = read_json(args.config)
                errors = (experiment_errors(config, execution=True) + parameter_errors(config)
                          + deployment_errors(config, image_sha256=os.environ.get('MERCURY_IMAGE_SHA256')))
                if not args.allow_live:
                    errors.append('live_execution_requires_allow_live')
                if errors:
                    raise ValueError(errors)
                checkpoint_errors = checkpoint_binding_errors(config, verify_bytes=True)
                if checkpoint_errors:
                    raise ValueError(checkpoint_errors)
                count = config['deployment']['required_gpu_count']
            else:
                count = args.gpu_count
            observed = collect_hardware()
            assessment = assess_hardware(observed, count, require_cuda=count > 0)
            if args.command == 'run':
                placement_errors = resource_errors(config, observed)
                assessment['placement_errors'] = placement_errors
                if placement_errors:
                    assessment['status'] = 'blocked'
            result = {'schema_version': 'mercury-hardware-report/v1', 'created_at': now(),
                      'status': assessment['status'], 'observation': observed, 'assessment': assessment,
                      'image_file_bytes_sha256': os.environ.get('MERCURY_IMAGE_SHA256')}
            if args.command == 'doctor':
                write_new(args.output, result)
            else:
                # Every resume gets a separate observation. Hardware failure never dispatches.
                report_id = digest(result)
                write_new(args.output / 'deployment_checks' / (report_id + '.json'), result)
                if result['status'] != 'pass':
                    raise ValueError('hardware_preflight_failed:' + report_id)
                batch = run_batch(config, read_json(args.corpus), source_root=args.source_root,
                                  output=args.output, allow_live=True, max_jobs=args.max_jobs)
                result = {'status': batch.get('status', 'pass'), 'deployment_check': report_id,
                          'batch': batch}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result = {'status': 'blocked', 'errors': [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result.get('status') in ('fail', 'blocked') else 0


if __name__ == '__main__':
    raise SystemExit(main())
