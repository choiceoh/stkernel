"""Same-build compact/dense candidate merge: exact CUDA outputs and component timing."""
import hashlib
import json
from pathlib import Path
import unittest

import torch
import triton

from engine.kernels.common.vocab_candidates import select_logits
from engine.modules.vocab import CandidateBuffer
from probes.engine_vocab_selection import capture, paired


@torch.inference_mode()
def run(output=None):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction((512 << 20) / torch.cuda.mem_get_info()[1])
    root = Path(__file__).resolve().parents[1]
    paths = ('engine/kernels/common/vocab_merge.py', 'engine/kernels/common/vocab_candidates.py',
             'engine/modules/vocab.py', 'tests/test_engine_vocab_merge.py',
             'probes/engine_vocab_merge.py', 'probes/engine_vocab_selection.py')
    report = dict(scope='Component-only, same input packets; no network or engine tok/s',
                  torch=str(torch.__version__), torch_git=torch.version.git_version,
                  triton=triton.__version__, device=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(),
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ('tests.test_engine_vocab_merge', 'tests.test_engine_vocab_topk', 'tests.test_engine_vocab_packet'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report['tests'] = dict(run=result.testsRun, skipped=len(result.skipped), passed=result.wasSuccessful())
    if not result.wasSuccessful() or result.skipped:
        raise AssertionError('compact merge requires all CUDA equivalence checks to execute and pass')
    generator = torch.Generator(device='cuda').manual_seed(170917)
    cases = []
    for rows in (1, 7, 14, 28):
        logits = torch.randn(rows, 154880, generator=generator, device='cuda').bfloat16()
        packet = torch.cat([select_logits(logits[:, r*38720:(r+1)*38720], r*38720,
                                         38720 if r < 3 else 38696, 16) for r in range(4)], -1)
        workspaces = {name: CandidateBuffer(rows, 154880, 64, 'cuda', compact=name == 'candidate')
                      for name in ('control', 'candidate')}
        graphs = {name: capture(lambda w=w: w.select(packet, 154880, 16), 16)
                  for name, w in workspaces.items()}
        try:
            for graph, _ in graphs.values(): graph.replay()
            for actual, expected in zip(graphs['candidate'][1], graphs['control'][1]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            case = dict(rows=rows, packet_bytes=packet.numel()*packet.element_size(),
                        dense_workspace_bytes=rows*154880*4, exact=True,
                        merge=paired(graphs, copies=16))
            cases.append(case)
            print(json.dumps(case), flush=True)
        finally:
            for graph, _ in graphs.values(): graph.reset()
        del graphs, workspaces, logits, packet
        torch.cuda.empty_cache()
    report.update(cases=cases, passed=True, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report
