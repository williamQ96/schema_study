"""Start the immutable V13 repair qualification in the background on Mercury."""
import argparse
import json
from pathlib import Path
from deploy_v13_repair import REMOTE, PYTHON, LOCAL, execute, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=int, required=True)
    args = parser.parse_args()
    job = REMOTE + '/candidate-' + format(args.candidate, '02d')
    code = '''import json,os,subprocess,sys,time
from pathlib import Path
j=Path(@@JOB@@)
sys.path.insert(0,str(j/'runtime-v1'))
from scheduler_v2.io import read,write_once,file_sha
from scheduler_v2.processes import identity
assert not (j/'qualification-process.json').exists(), 'qualification_already_launched'
assert all(file_sha(j/n)==h for n,h in read(j/'qualification-tools.json')['files'].items())
env=dict(os.environ,V13_JOB=str(j))
with (j/'qualification.log').open('xb') as log:
 p=subprocess.Popen([@@PYTHON@@,str(j/'v13_repair_launcher.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
result={'identity':identity(p.pid),'time':time.time(),'candidate':str(j),'mode':'qualification_only','production_started':False}
write_once(j/'qualification-process.json',result)
print(json.dumps(result))'''.replace('@@JOB@@', repr(job)).replace('@@PYTHON@@', repr(PYTHON))
    value = execute(code)
    write(LOCAL / ('candidate-' + format(args.candidate, '02d')) / 'qualification-start.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
