from pathlib import Path
import ctypes
repo=Path(__file__).resolve().parents[2]
original=repo/'scripts/bench_pi05.py'
source=original.read_text()
source=source.replace('    for i in range(opt.warmup + opt.runs):','''    marker=ctypes.CDLL('/usr/local/musa/lib/libmusart.so').musaGetDeviceCount
    marker.argtypes=[ctypes.POINTER(ctypes.c_int)];marker.restype=ctypes.c_int
    marker_count=ctypes.c_int()
    for i in range(opt.warmup + opt.runs):''')
source=source.replace('        start = time.perf_counter()', '        if i >= opt.warmup:assert marker(ctypes.byref(marker_count))==0\n        start = time.perf_counter()')
source=source.replace('        wall = (time.perf_counter()-start)*1000','        wall = (time.perf_counter()-start)*1000\n        if i >= opt.warmup:assert marker(ctypes.byref(marker_count))==0')
exec(compile(source,str(original),'exec'),{'__file__':str(original),'__name__':'__main__'})
