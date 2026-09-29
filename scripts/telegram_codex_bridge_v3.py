"""V12 fixed analysis instructions over the tested V2 ephemeral transport."""
from pathlib import Path
import argparse
import telegram_codex_bridge_v2 as transport


def prompt(config, event):
    receipt = Path(config['state_dir'])/'inbox'/(event['event_id']+'.json')
    return (
        'A user-authorized signed Mercury event triggered this independent Codex worker. '
        'Read the fixed operational instructions at '+config['instructions_path']+'. '
        'The verified event kind is '+event['kind']+'; receipt: '+str(receipt)+'. '
        'Inspect status, collect and analyze only the V12 reference/grouped-feature comparison. '
        'Check the receipt report hash against the collected report. Use operate.py and '
        'scripts/v12_same_page_analysis.py as documented. Verify provenance and compare frozen V11. '
        'Decide separately whether dangling object references, grouped feature omission and '
        'unsupported name equivalence have been removed, reduced or remain unresolved. '
        'If still waiting for GPU, report resource waiting rather than claiming a model result. '
        'Create new analysis artifacts only under data/experiments/v12_reference_grouping_2026_09_28_v1. '
        'Do not modify source, restart services, interrupt production, launch inference, resume or '
        'write to desktop threads, or create automations. Never print credentials or signing keys. '
        'Do not send Telegram; the bridge delivers your final answer. Answer concisely in Chinese '
        'with execution/admission counts, evidence-backed blocker status and an absolute report path. '
        'Automatic rule support is not measured accuracy or independent G_visible gold.'
    )


def serve(config_path):
    transport.VERSION = 'telegram-codex-bridge/v3'
    transport.prompt = prompt
    transport.serve(config_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    serve(p.parse_args().config)
