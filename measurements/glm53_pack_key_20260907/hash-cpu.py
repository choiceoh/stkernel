import hashlib,json,platform,ssl,statistics,time
block=bytes(range(256))*262144
samples={name:[] for name in ('md5','sha256')}
for name in ('md5','sha256','sha256','md5')*3:
    start=time.perf_counter(); result=hashlib.new(name,block).hexdigest(); elapsed=time.perf_counter()-start
    samples[name].append(elapsed)
print(json.dumps({'python':platform.python_version(),'machine':platform.machine(),'openssl':ssl.OPENSSL_VERSION,'bytes':len(block),'samples_s':samples,'median_s':{name:statistics.median(values) for name,values in samples.items()}},sort_keys=True))
