import subprocess
from concurrent.futures import ThreadPoolExecutor
script = '/home/choiceoh/st-engine-f4d7-20260911/run-comm.sh'
def run(item):
    rank, node = item
    command = ['bash', script, str(rank)] if node == 1 else ['ssh', '-o', 'BatchMode=yes', f'choiceoh@10.10.10.{node}', 'bash', script, str(rank)]
    p = subprocess.run(command, text=True, capture_output=True, timeout=115)
    return {'node':node, 'rank':rank, 'returncode':p.returncode, 'stdout':p.stdout[-1000:], 'stderr':p.stderr[-1000:]}
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, enumerate((2, 1, 3, 4))):
        print(result, flush=True)
