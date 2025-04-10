from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

_T = TypeVar("_T", bound=Callable[..., Any])

@overload
def njit(
    func: _T
) -> _T:
    ...

@overload
def njit(
    parallel: bool = ...,
    fastmath: bool = ...,
    boundscheck: bool = ...,
    cache: bool = ...,
    # etc. add other parameters as needed
) -> Callable[[_T], _T]:
    ...

@overload
def prange(stop: int) -> range: ...
@overload
def prange(start: int, stop: int, step: int = 1) -> range: ...

__all__ = [
    'cfunc', 'from_dtype', 'guvectorize', 'jit', 'experimental', 'njit',
    'stencil', 'jit_module', 'typeof', 'prange', 'gdb', 'gdb_breakpoint',
    'gdb_init', 'vectorize', 'objmode', 'literal_unroll', 'get_num_threads',
    'set_num_threads', 'set_parallel_chunksize', 'get_parallel_chunksize',
    'parallel_chunksize', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    'uint32', 'uint64', 'intp', 'uintp', 'intc', 'uintc', 'ssize_t', 'size_t',
    'boolean', 'float32', 'float64', 'complex64', 'complex128', 'bool_',
    'byte', 'char', 'uchar', 'short', 'ushort', 'int_', 'uint', 'long_',
    'ulong', 'longlong', 'ulonglong', 'float_', 'double', 'void', 'none', 'b1',
    'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16',
    'optional', 'ffi_forced_object', 'ffi', 'deferred_type', 'NumbaWarning',
    'NumbaPerformanceWarning', 'NumbaDeprecationWarning',
    'NumbaPendingDeprecationWarning', 'NumbaParallelSafetyWarning',
    'NumbaTypeSafetyWarning', 'NumbaExperimentalFeatureWarning',
    'NumbaInvalidConfigWarning', 'NumbaPedanticWarning',
    'NumbaIRAssumptionWarning', 'NumbaDebugInfoWarning', 'NumbaSystemWarning',
    'NumbaError', 'UnsupportedError', 'UnsupportedBytecodeError',
    'UnsupportedRewriteError', 'IRError', 'RedefinedError', 'NotDefinedError',
    'VerificationError', 'DeprecationError', 'LoweringError',
    'UnsupportedParforsError', 'ForbiddenConstruct', 'TypingError',
    'UntypedAttributeError', 'ByteCodeSupportError', 'CompilerError',
    'ConstantInferenceError', 'InternalError', 'InternalTargetMismatchError',
    'NonexistentTargetError', 'RequireLiteralValue', 'ForceLiteralArg',
    'LiteralTypingError', 'NumbaValueError', 'NumbaTypeError',
    'NumbaAttributeError', 'NumbaAssertionError', 'NumbaNotImplementedError',
    'NumbaKeyError', 'NumbaIndexError', 'NumbaRuntimeError'
]
