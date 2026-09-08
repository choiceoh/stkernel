#!/usr/bin/env python3
"""Reproduce counts from downloaded version-tagged NVIDIA loader templates."""
import argparse,collections,hashlib,json,pathlib,re
root=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source-dir",type=pathlib.Path,required=True,help="Directory containing the two versioned source files from report.json")
args=ap.parse_args()
provenance=json.loads((root/"report.json").read_text())
for item in provenance["source_basis"]:
    if item["file"].startswith("driver-template-"):
        raw=(args.source_dir/item["file"]).read_bytes()
        assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"]
report={}
for version in ('13.0.3','13.3.1'):
    calls=[]
    for line_no,line in enumerate((args.source_dir/f'driver-template-{version}.pyx').read_text().splitlines(),1):
        match=re.search(r"_F_cuGetProcAddress_v2\('([^']+)', &([^,]+), (\d+), ([^,]+), NULL\)",line)
        if match:
            symbol,destination,abi,flags=match.groups()
            if flags=='CU_GET_PROC_ADDRESS_DEFAULT':
                calls.append(dict(line=line_no,symbol=symbol,destination=destination,abi_version=int(abi),flags=flags))
    high=[row for row in calls if row['abi_version']>13000]
    report[version]=dict(default_lookups=len(calls),max_abi=max(row['abi_version'] for row in calls),above_driver=len(high),above_by_version=dict(collections.Counter(row['abi_version'] for row in high)),entries=high)
assert report['13.0.3']['default_lookups']==470
assert report['13.0.3']['above_driver']==0
assert report['13.3.1']['default_lookups']==504
assert report['13.3.1']['above_driver']==34
assert report['13.3.1']['above_by_version']=={13010:9,13020:12,13030:13}
print(json.dumps(report,indent=2))
