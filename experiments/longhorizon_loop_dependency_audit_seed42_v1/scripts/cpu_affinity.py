"""Task-process affinity from observed Windows CPU-set classes; no OS settings."""
import ctypes,json,os,pathlib,struct
from ctypes import wintypes
ROOT=pathlib.Path(__file__).resolve().parents[1]
def apply_task_affinity():
    if os.name!='nt':return {'status':'not_windows'}
    lib=ctypes.WinDLL('kernel32',use_last_error=True)
    lib.GetCurrentProcess.restype=wintypes.HANDLE;handle=lib.GetCurrentProcess()
    get=lib.GetSystemCpuSetInformation
    get.argtypes=[ctypes.c_void_p,wintypes.ULONG,ctypes.POINTER(wintypes.ULONG),wintypes.HANDLE,wintypes.ULONG];get.restype=wintypes.BOOL
    size=wintypes.ULONG();get(None,0,ctypes.byref(size),handle,0);assert size.value
    buffer=ctypes.create_string_buffer(size.value);assert get(buffer,size,ctypes.byref(size),handle,0)
    data=buffer.raw[:size.value];rows=[];offset=0
    while offset<len(data):
        length,kind=struct.unpack_from('<II',data,offset);assert length>=8 and offset+length<=len(data)
        if kind==0:
            identifier,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from('<IHBBBBBB',data,offset+8)
            rows.append({'id':identifier,'group':group,'logical_index':logical,'core_index':core,'efficiency_class':efficiency})
        offset+=length
    assert rows and all(r['group']==0 for r in rows),'Do not guess multi-group affinity'
    proc=ctypes.c_size_t();system=ctypes.c_size_t()
    lib.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    assert lib.GetProcessAffinityMask(handle,ctypes.byref(proc),ctypes.byref(system))
    classes={r['efficiency_class'] for r in rows};lowest=min(classes)
    selected=[r['logical_index'] for r in rows if r['efficiency_class']==lowest and proc.value&(1<<r['logical_index'])]
    mask=sum(1<<i for i in selected);assert mask and mask&proc.value==mask
    lib.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t];lib.SetProcessAffinityMask.restype=wintypes.BOOL
    assert lib.SetProcessAffinityMask(handle,mask),ctypes.get_last_error()
    report={'status':'process_only_affinity_applied','CPU_sets_observed':rows,'selected_logical_indices':selected,'selected_efficiency_class':lowest,
            'previous_mask_hex':hex(proc.value),'selected_mask_hex':hex(mask),'scope':'this foreground runner and its children only; no global scheduling, voltage, BIOS or model changes',
            'interpretation':'lower numerical efficiency class is slower/more energy efficient according to Windows API; this does not diagnose CPU damage',
            'documentation':'https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information'}
    (ROOT/'outputs/cpu_affinity_control.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('CPU_AFFINITY',json.dumps(report),flush=True)
    return report
