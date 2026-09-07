"""Exact pack/unpack and retained-output cases for the owned sanitizer runner."""
import hashlib
from pathlib import Path
from glm53_prefill_int8_check import geometry,reference_packet

ROWS=(129,6143,6144,6912,8192)


def validate(report):
    expected=[(n,changed,destination) for n in ROWS for changed in (False,True) for destination in range(4)]
    if [(r['rows'],r['changed'],r['destination']) for r in report['cases']]!=expected:
        raise ValueError('complete INT8 sanitizer coverage required')
    if any(not all(r[k] for k in ('packet_equal','output_equal','source_unchanged','finite','retained_unchanged'))
           for r in report['cases']):
        raise ValueError('INT8 sanitizer fidelity or lifetime failed')
    return report


def run(torch,h):
    cases=[]
    for rows in ROWS:
        g=geometry(rows)
        generator=torch.Generator(device='cuda').manual_seed(83551+rows)
        inputs=[torch.randn((rows,4096),generator=generator,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
        retained=[]
        for changed in (False,True):
            if changed:
                for x in inputs:x.mul_(-.75)
                trash=[torch.empty_like(inputs[0]) for _ in range(3)];del trash
            packets=[];decoded=[];packet_equal=True;source_unchanged=True
            for x in inputs:
                before=x.clone()
                packet=torch.full((4*g['payload_bytes'],),0xA5,dtype=torch.uint8,device='cuda')
                h._pack_rs_payload_int8[(g['padded_rows']*2,)](x,packet.view(torch.int8),packet.view(torch.float32),
                    N=x.numel(),LOCAL_N=g['local_n'],PAYLOAD_BYTES=g['payload_bytes'],BLOCK=2048)
                expected,values,_=reference_packet(torch,x.cpu())
                packet_equal &= torch.equal(packet.cpu(),expected)
                source_unchanged &= torch.equal(x.view(torch.int16),before.view(torch.int16))
                packets.append(packet.reshape(4,-1));decoded.append(values)
            for destination in range(4):
                # Assemble the exact destination packet order produced by TP4 all-to-all.
                received=torch.stack([p[destination] for p in packets]).reshape(-1)
                output=torch.full((g['local_n']//4096,4096),float('nan'),dtype=torch.bfloat16,device='cuda')
                h._unpack_sum_payload[(g['local_n']//2048,)](received.view(torch.int8),received.view(torch.float32),output,
                    LOCAL_N=g['local_n'],PAYLOAD_BYTES=g['payload_bytes'],TP=4,BLOCK=2048)
                start=destination*output.shape[0];end=start+output.shape[0]
                reference=torch.zeros(output.shape,dtype=torch.float32)
                for value in decoded:reference+=value[start:end]
                case=dict(rows=rows,changed=changed,destination=destination,packet_equal=packet_equal,
                    output_equal=torch.equal(output.cpu().view(torch.int16),reference.to(torch.bfloat16).view(torch.int16)),
                    finite=bool(torch.isfinite(output).all()),source_unchanged=source_unchanged,
                    retained_unchanged=all(torch.equal(a,b) for a,b in retained))
                cases.append(case)
                retained.append((output,output.clone()))
    return validate(dict(cases=cases,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
