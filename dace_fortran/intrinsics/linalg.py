"""Linear-algebra Fortran intrinsics -> dedicated DaCe library nodes.

Each lowers directly from a first-class HLFIR op: ``hlfir.matmul`` ->
``blas.MatMul``, ``hlfir.transpose`` -> ``linalg.Transpose``,
``hlfir.dot_product`` -> ``blas.Dot``.  ``MatMul`` is a meta-node whose
``SpecializeMatMul`` expansion dispatches to Gemm/BatchedMatMul/Gemv/Dot by
operand rank.  ``Transpose`` is rank-2 only; ``Dot`` takes two rank-1 inputs.
"""

from dace_fortran.intrinsics.base import LibNodeIntrinsic

LINALG: dict[str, LibNodeIntrinsic] = {
    'matmul': LibNodeIntrinsic('matmul', module='blas', node_cls='MatMul'),
    'transpose': LibNodeIntrinsic('transpose', module='linalg', node_cls='Transpose'),
    'dot_product': LibNodeIntrinsic('dot_product', module='blas', node_cls='Dot'),
    # ``hlfir.matmul_transpose`` (fused ``MATMUL(TRANSPOSE(A), B)``):
    # ``emit_libcall`` composes a Transpose + MatMul pair; entry exists so
    # ``libnode_spec("matmul_transpose")`` resolves at dispatch time.
    'matmul_transpose': LibNodeIntrinsic('matmul_transpose', module='blas', node_cls='MatMul'),
}

# Generic/non-linalg library nodes emitted via the same ``kind="libcall"``
# path; separate dict for readability -- ``libnode_spec`` looks up across both.
STANDARD: dict[str, LibNodeIntrinsic] = {
    'count': LibNodeIntrinsic('count', module='standard', node_cls='CountLibraryNode'),
    'merge': LibNodeIntrinsic('merge', module='standard', node_cls='MergeLibraryNode'),
    'argmin': LibNodeIntrinsic('argmin', module='standard', node_cls='ArgMin'),
    'argmax': LibNodeIntrinsic('argmax', module='standard', node_cls='ArgMax'),
    'cshift': LibNodeIntrinsic('cshift', module='standard', node_cls='CShift'),
    'eoshift': LibNodeIntrinsic('eoshift', module='standard', node_cls='EOShift'),
    'norm2': LibNodeIntrinsic('norm2', module='standard', node_cls='Norm2'),
    'broadcast': LibNodeIntrinsic('broadcast', module='standard', node_cls='Broadcast'),
}
