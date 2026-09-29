"""CPU-only replay of saved Muse tokens against the frozen XGrammar contract."""
import json
from pathlib import Path
import sys

import torch
import xgrammar as xgr
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from high_fidelity_schema_study.four_category import structured_output as so
from high_fidelity_schema_study.four_category import extraction_v13 as v13

torch.set_num_threads(2)
assert torch.cuda.device_count()==0, 'CPU diagnostic must not access production GPUs'
request=json.loads(Path(sys.argv[1]).read_text())
rows=[]
for item in request['records']:
    record=json.loads(Path('/run/'+item['path']).read_text())
    profile=record['profile']; b=record['backend_result']; req=b['request']; raw=b['raw_response']
    tok=AutoTokenizer.from_pretrained(profile['model_id'],local_files_only=True,trust_remote_code=False)
    cfg=AutoConfig.from_pretrained(profile['model_id'],local_files_only=True,trust_remote_code=False)
    gen=GenerationConfig.from_pretrained(profile['model_id'],local_files_only=True)
    eos=gen.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos])
    info=so._tokenizer_info(xgr,tok,cfg.get_text_config().vocab_size,eos,muse=True)
    paper=json.loads(Path('/sources/mineru/'+record['job']['paper_id']+'/input.json').read_text())
    index=next(v for p in Path('/run/indexes').glob('*.json') for v in [json.loads(p.read_text())]
               if v['index_sha256']==record['index_sha256'])
    schema=v13.response_schema(record['task'],paper,index,record['extraction_group'])
    assert schema==req['structured_output']['schema']
    # any_order=False depends on the reconstructed schema's insertion order.
    # Canonically serialized record dictionaries alone do not preserve it.
    grammar=xgr.GrammarCompiler(info,max_threads=2).compile_structural_tag(
        so.muse_format(schema,req['generation_parameters']['max_new_tokens'],tok))
    matcher=xgr.GrammarMatcher(grammar,terminate_without_stop_token=True)
    bitmask=xgr.allocate_token_bitmask(1,info.vocab_size)
    first_rejected=None
    for i,token in enumerate(raw['output_token_ids']):
        matcher.fill_next_token_bitmask(bitmask)
        if not ((int(bitmask[0,token//32])>>(token%32))&1):first_rejected=i;break
        if not matcher.accept_token(token):first_rejected=i;break
    rows.append({'record':item['path'],'production_status':record['status'],
                 'output_tokens':len(raw['output_token_ids']),'first_grammar_rejected_token':first_rejected,
                 'grammar_terminated':matcher.is_terminated(),
                 'reconstructed_schema_hash_matches':so.schema_digest(schema)==req['structured_output']['schema_sha256'],
                 'stored_property_order_matches':list(schema['properties'])==list(req['structured_output']['schema']['properties']),
                 'decoded_tokens_match_raw':tok.decode(raw['output_token_ids'],skip_special_tokens=False)==raw['decoded_with_special_tokens'],
                 'reasoning_budget':req['generation_parameters']['max_new_tokens']})

# A bounded grammar distinguishes syntactic permission from response completion.
specials=['<|message|>','<|eot|>','<|eom|>','<|start|>','<|tool|>']
vocab=[chr(i) for i in range(32,127)]+specials
class Tokenizer:
    all_special_tokens=specials
    def get_vocab(self):return {v:i for i,v in enumerate(vocab)}
schema={'type':'object','properties':{'state':{'enum':['classified','none','uncertain']}},'required':['state'],'additionalProperties':False}
grammar=xgr.GrammarCompiler(xgr.TokenizerInfo(vocab,stop_token_ids=[]),max_threads=2).compile_structural_tag(so.muse_format(schema,16,Tokenizer()))
ids={v:i for i,v in enumerate(vocab)}
def accept(parts):
    m=xgr.GrammarMatcher(grammar,terminate_without_stop_token=True)
    mask=xgr.allocate_token_bitmask(1,len(vocab))
    for part in parts:
        for token in ([part] if part in specials else list(part)):
            m.fill_next_token_bitmask(mask)
            if not ((int(mask[0,ids[token]//32])>>(ids[token]%32))&1):return {'accepted_prefix':False,'terminated':m.is_terminated()}
            if not m.accept_token(ids[token]):return {'accepted_prefix':False,'terminated':m.is_terminated()}
    return {'accepted_prefix':True,'terminated':m.is_terminated()}
checks={
 'unconstrained_thought_at_cap':accept([' to=self','<|message|>','x'*16]),
 'thought_over_cap':accept([' to=self','<|message|>','x'*17]),
 'invalid_final_enum':accept([' to=user','<|message|>','{"state":"value"}','<|eot|>']),
 'valid_final_enum':accept([' to=user','<|message|>','{"state":"classified"}','<|eot|>'])}
print(json.dumps({'gpu_count':torch.cuda.device_count(),'model_generation_calls':0,'replay':rows,'toy_checks':checks}))
