"""Read-only host identity for the fixed AR consumer GPU campaign."""
from concurrent.futures import ThreadPoolExecutor
import json

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
IPS = '10.10.10.2,10.10.10.1,10.10.10.3,10.10.10.4'
INIT = 'tcp://10.10.10.2:29758'

# No container or CUDA process is started. These are image metadata, driver
# inventory and stable sysfs/configuration reads, not a GPU validation workload.
RUNTIME_SCRIPT = r'''
import errno, hashlib, json, os, subprocess
from pathlib import Path

def run(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()

def read(path, optional=False):
    try:
        return path.read_text().strip()
    except OSError as exc:
        # Some RoCE ports expose InfiniBand-only attributes (for example lid)
        # but reject reads. Preserve that stable state in the identity; all
        # permission, I/O and other unexpected errors still fail closed.
        if optional and exc.errno in (errno.EINVAL, errno.ENODATA):
            return {'unavailable': errno.errorcode[exc.errno]}
        raise

image = run('docker', 'image', 'inspect', '--format={{.Id}}', IMAGE)
gpu_values = run('nvidia-smi', '-i', '0',
    '--query-gpu=uuid,name,driver_version,pci.bus_id',
    '--format=csv,noheader,nounits').split(',')
if len(gpu_values) != 4:
    raise ValueError('one GPU identity required')
gpu = dict(zip(('uuid', 'name', 'driver_version', 'pci_bus_id'),
               (value.strip() for value in gpu_values)))
rdma = {}
for device in sorted(Path('/sys/class/infiniband').glob('*')):
    info = {'device_path': str((device / 'device').resolve())}
    for field in ('node_guid', 'sys_image_guid', 'fw_ver', 'hca_type', 'board_id'):
        path = device / field
        if path.is_file():
            info[field] = read(path, optional=True)
    ports = {}
    for port in sorted((device / 'ports').glob('*')):
        fields = {}
        for field in ('state', 'phys_state', 'link_layer', 'rate', 'active_mtu', 'lid', 'sm_lid'):
            path = port / field
            if path.is_file():
                fields[field] = read(path, optional=True)
        for field in ('gids', 'gid_attrs/ndevs', 'gid_attrs/types'):
            # Unpopulated GID slots exist in sysfs but return EINVAL/ENODATA.
            fields[field] = {path.name: read(path, optional=True)
                             for path in sorted((port / field).glob('*')) if path.is_file()}
        ports[port.name] = fields
    info['ports'] = ports
    rdma[device.name] = info
if not rdma:
    raise ValueError('RDMA identity unavailable')
# Normalize addresses; lifetimes/counters change even when topology does not.
rdma_interfaces = {name for device in rdma.values() for port in device['ports'].values()
                   for name in port['gid_attrs/ndevs'].values() if isinstance(name, str) and name}
if not rdma_interfaces:
    raise ValueError('no populated RDMA network-device mapping')
network = []
for interface in json.loads(run('ip', '-json', 'address', 'show')):
    if (interface['ifname'] not in rdma_interfaces
            and not any(addr.get('local') in TOPOLOGY_IPS for addr in interface.get('addr_info', []))):
        continue
    network.append({'ifname': interface['ifname'], 'address': interface.get('address'),
        'addresses': sorted(({'family': addr['family'], 'local': addr['local'],
            'prefixlen': addr['prefixlen'], 'scope': addr.get('scope')}
            for addr in interface.get('addr_info', [])), key=lambda addr: (addr['family'], addr['local']))})
if not any(addr['local'] == NODE_IP for interface in network for addr in interface['addresses']):
    raise ValueError('fixed rank IP is missing from host interfaces')
rdma['network'] = sorted(network, key=lambda interface: interface['ifname'])
sanitizer = Path('/usr/local/cuda/compute-sanitizer')
manifest = {}
for path in sorted(sanitizer.rglob('*')):
    if path.is_file():
        manifest[str(path.relative_to(sanitizer))] = hashlib.sha256(path.read_bytes()).hexdigest()
if not manifest:
    raise ValueError('compute-sanitizer fingerprint unavailable')
sanitizer_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
print(json.dumps({'image': image, 'gpu': gpu,
    'kernel': os.uname().release + '|' + os.uname().version,
    'rdma': rdma, 'sanitizer_sha256': sanitizer_sha256}, sort_keys=True))
'''


def validate_runtime(runtime):
    """Reject absent identity rather than making an unknown host cacheable."""
    if (not isinstance(runtime, dict) or runtime.get('schema') != 1
            or runtime.get('image') != IMAGE
            or runtime.get('topology') != {'nodes': list(NODES), 'ips': IPS, 'init': INIT}
            or not isinstance(runtime.get('nodes'), dict)
            or set(runtime['nodes']) != set(NODES)):
        raise ValueError('incomplete runtime/topology identity')
    uuids = []
    for node in NODES:
        host = runtime['nodes'][node]
        if not isinstance(host, dict) or host.get('image') != IMAGE:
            raise ValueError('runtime image mismatch: ' + node)
        gpu = host.get('gpu')
        if (not isinstance(gpu, dict) or gpu.get('name') != 'NVIDIA GB10'
                or any(not isinstance(gpu.get(key), str) or not gpu[key].strip()
                       for key in ('uuid', 'driver_version', 'pci_bus_id'))):
            raise ValueError('incomplete GPU identity: ' + node)
        if (not isinstance(host.get('kernel'), str) or not host['kernel'].strip()
                or not isinstance(host.get('rdma'), dict) or not host['rdma']):
            raise ValueError('incomplete kernel/RDMA identity: ' + node)
        digest = host.get('sanitizer_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('missing sanitizer fingerprint: ' + node)
        uuids.append(gpu['uuid'])
    if len(set(uuids)) != len(NODES):
        raise ValueError('duplicate physical GPUs in runtime topology')
    return runtime


def collect_runtime(remote):
    """Snapshot actual image, GPU, driver, RDMA and sanitizer on every rank."""
    def collect(node):
        constants = ('IMAGE = ' + repr(IMAGE) + '\nTOPOLOGY_IPS = ' + repr(IPS.split(','))
                     + '\nNODE_IP = ' + repr(IPS.split(',')[NODES.index(node)]) + '\n')
        result = remote(node, ['python3', '-c', constants + RUNTIME_SCRIPT],
                        capture_output=True, text=True, timeout=180)
        return json.loads(result.stdout)

    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        hosts = dict(zip(NODES, pool.map(collect, NODES)))
    return validate_runtime({'schema': 1, 'image': IMAGE,
        'topology': {'nodes': list(NODES), 'ips': IPS, 'init': INIT}, 'nodes': hosts})


def runtime_key(runtime, group):
    """Local stages depend only on their GPU; RDMA stages bind the full cohort."""
    validate_runtime(runtime)
    nodes = ('local',) if group.startswith('local-') else NODES
    return {'schema': runtime['schema'], 'image': runtime['image'],
            'topology': runtime['topology'],
            'nodes': {node: runtime['nodes'][node] for node in nodes}}
