"""Offline evidence audit of a fixed capture; does not relabel production results."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from high_fidelity_schema_study.four_category import extraction_v10 as v10
from high_fidelity_schema_study.four_category import extraction_v12 as v12
from high_fidelity_schema_study.four_category import extraction_v13 as v13
from high_fidelity_schema_study.four_category.common import seal_errors
from high_fidelity_schema_study.four_category.response_channels import decode_muse_atem, SELF_HEADER, NEXT_USER_HEADER, EOM, EOT
from high_fidelity_schema_study.four_category.workflow import replay_run
from high_fidelity_schema_study.four_category.replay_cache import replay_validation_scope


def analyze(root):
    manifest = json.loads((root/'capture.json').read_text())
    def load(name):
        ref = manifest['members'][name]
        data = (root/ref['local']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['file_bytes_sha256']:
            raise ValueError('captured_member_changed:' + name)
        return json.loads(data)
    condition = load('condition.json')
    papers = {p['paper_id']:load('sources/'+p['source']['artifacts']['input']['path'])
              for p in condition['corpus']['papers']
              if 'sources/'+p['source']['artifacts']['input']['path'] in manifest['members']}
    indexes = {v['index_sha256']:v for name in manifest['members'] if name.startswith('indexes/') for v in [load(name)]}
    windows = {pid:v10.window_catalog(p) for pid,p in papers.items()}
    counts=Counter(); errors=Counter(); classifier=[]; channels=[]; object_errors=[]; table_errors=[]; replay=[]; lengths=[]
    with replay_validation_scope():
        for i,row in enumerate(manifest['rows']):
            record = load(row['record']); kind=row['job']['kind']; worker=row['worker']; pid=row['job']['paper_id']
            counts[(kind,worker,record['status'])]+=1
            b=record['backend_result']; req=b.get('request') or {}; raw=b.get('raw_response') or {}
            text=raw.get('decoded_with_special_tokens',''); payload=record.get('parsed_response')
            errors.update((kind,worker,e) for e in {e.split(':')[0] for e in record.get('validation_errors',[])})
            base={'record':row['record'],'paper':pid,'worker':worker,'status':record['status'],
                  'group_index':(record.get('extraction_group') or record.get('classification_group'))['index']}
            if kind=='classification':
                classifier.append({**base,'structured_requested':'structured_output' in req,
                  'structured_applied':b.get('structured_output_applied'),
                  'target_units':len(record['classification_group']['unit_ids']),
                  'invalid_states':dict(Counter(e.get('state') for e in (payload or {}).get('entries',[])
                                              if e.get('state') not in {'classified','none','uncertain'})),
                  'errors':list(dict.fromkeys(record.get('validation_errors',[])))})
            if worker=='muse-1':
                reasoning=text[len(SELF_HEADER):].split(EOM,1)[0] if text.startswith(SELF_HEADER) else ''
                user=req['messages'][-1]['content']; n=0
                for a,z in zip(reasoning,user):
                    if a!=z:break
                    n+=1
                try: decoded=decode_muse_atem(text); channel_error=None
                except ValueError as exc: decoded=None; channel_error=str(exc)
                channels.append({**base,'output_tokens':b['usage']['output_tokens'],
                    'reasoning_header':text.startswith(SELF_HEADER),'reasoning_ended':EOM in text,
                    'final_header':NEXT_USER_HEADER in text,'final_ended':text.endswith(EOT),
                    'copied_input_prefix_characters':n,'reasoning_characters':len(reasoning),
                    'channel_error':channel_error,'normalization_matches':None if decoded is None else b['raw_text']==decoded['text'],
                    'decoded_final_characters':len(decoded['text']) if decoded else 0})
            if record['status']=='truncated' and worker!='muse-1':
                core=raw.get('text','')
                try:
                    obj,end=json.JSONDecoder().raw_decode(core.lstrip()); tail=core.lstrip()[end:]
                    lengths.append({**base,'json_complete':True,'trailing_characters':len(tail),
                                    'only_whitespace_after_json':not tail.strip(),
                                    'offline_validation_errors':v13.validate_response(obj,papers[pid],indexes[record['index_sha256']],record['extraction_group']),
                                    'production_status_unchanged':True})
                except (ValueError,TypeError,KeyError) as exc:lengths.append({**base,'json_complete':False,'error':str(exc)[:200]})
            if kind=='local_extraction' and payload and record['status']=='contract_invalid':
                group=record['extraction_group']; target=set(group['window_ids']); catalog=windows[pid]
                for obj in payload.get('objects',[]):
                    if not target.intersection(obj['source_windows']):
                        object_errors.append({**base,'target_pages':group['pages'],'label':obj['normalized_label'],
                          'source_windows':[catalog[w] for w in obj['source_windows']], 'fact_count':len(obj['facts'])})
                for review,table in zip(payload['table_reviews'],v12.context(papers[pid],group)['table_candidates']):
                    if any(catalog[w]['unit_id'] in table['unit_ids'] for w in review['source_windows']):continue
                    own=[w for w in catalog if w['unit_id'] in table['unit_ids']]
                    prompt=json.loads(record['messages'][1]['content'])
                    shown=next(t for t in prompt['target']['table_candidates'] if t['table_id']==table['table_id'])
                    table_errors.append({**base,'target_pages':group['pages'],'table_id':table['table_id'],
                      'review':review,'cited_windows':[catalog[w] for w in review['source_windows']],
                      'eligible_windows_count':len(own),'eligible_examples':own[:3],
                      'all_citations_are_table_unit_indexes':set(review['source_windows'])<=set(shown['unit_indexes']),
                      'same_page_citations':all(catalog[w]['page'] in group['pages'] for w in review['source_windows']),
                      'identical_text_other_unit':any(catalog[w]['text']==x['text'] for w in review['source_windows'] for x in own)})
            problems=replay_run(record,record['task'],papers[pid],index=indexes.get(record['index_sha256']))
            replay.append({**base,'errors':problems,'record_seal_errors':seal_errors(record,'record_sha256')})
            if (i+1)%20==0:print(json.dumps({'replayed':i+1,'total':len(manifest['rows'])}),flush=True)
    report={'capture_time_utc':datetime.fromtimestamp(manifest['captured_at'],timezone.utc).isoformat(),
      'condition':manifest['snapshot']['condition'],'records':len(manifest['rows']),
      'counts':[dict(stage=k[0],worker=k[1],status=k[2],count=n) for k,n in counts.items()],
      'replay':{'checked':len(replay),'failed':[r for r in replay if r['errors'] or r['record_seal_errors']]},
      'error_group_counts':[dict(stage=k[0],worker=k[1],error=k[2],groups=n) for k,n in errors.items()],
      'classification':classifier,'muse_channels':channels,'non_muse_truncations':lengths,
      'object_evidence_rejections':object_errors,'table_evidence_rejections':table_errors,
      'indexes':[{'paper':v['source_identity']['document_id'],'sha256':v['index_sha256'],
                  'availability_counts':v['availability_counts'],'generation_coverage':v['generation_coverage']} for v in indexes.values()],
      'interpretation':'Offline diagnosis of frozen returned outcomes; no semantic gold, production relabeling, or new model generation.'}
    (root/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['capture_time_utc','records','counts','replay','error_group_counts','indexes','non_muse_truncations']},ensure_ascii=False))


if __name__=='__main__':analyze(Path(sys.argv[1]))
