"""Process-local affinity workaround; verify the Windows API result explicitly."""
import ctypes
import sys


def configure_process_affinity():
    if sys.platform != 'win32':
        return None
    requested = 0xffff0000
    kernel = ctypes.windll.kernel32
    setter = kernel.SetProcessAffinityMask
    setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    setter.restype = ctypes.c_int
    getter = kernel.GetProcessAffinityMask
    getter.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    getter.restype = ctypes.c_int
    handle = ctypes.c_void_p(-1)
    if not setter(handle, requested):
        raise OSError('Process-local CPU affinity could not be applied.')
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not getter(handle, ctypes.byref(process), ctypes.byref(system)):
        raise OSError('Process-local CPU affinity could not be verified.')
    if process.value != requested:
        raise AssertionError(f'CPU affinity mismatch: {process.value:#x}')
    return {'requested': hex(requested), 'actual': hex(process.value), 'system': hex(system.value)}
