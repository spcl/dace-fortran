"""Linear-algebra Fortran intrinsics -> dedicated DaCe library nodes.

Each lowers directly from a first-class HLFIR op, no post-SDFG
pattern-match pass needed:

    hlfir.matmul       -> dace.libraries.blas.nodes.matmul.MatMul
    hlfir.transpose    -> dace.libraries.linalg.nodes.transpose.Transpose
    hlfir.dot_product  -> dace.libraries.blas.nodes.dot.Dot

``MatMul`` is a meta-node: its ``SpecializeMatMul`` expansion dispatches
to ``Gemm``, ``BatchedMatMul``, ``Gemv``, or ``Dot`` depending on the
operand ranks, so the single registry entry handles matrix-matrix,
matrix-vector, and vector-matrix Fortran ``matmul`` calls alike.

``Transpose`` is defined on rank-2 arrays only and specialises
MKL / OpenBLAS / cuBLAS backends internally.

``Dot`` produces a scalar result from two rank-1 inputs.
"""

from dace_fortran.intrinsics.base import LibNodeIntrinsic

LINALG: dict[str, LibNodeIntrinsic] = {
    'matmul': LibNodeIntrinsic('matmul', module='blas', node_cls='MatMul'),
    'transpose': LibNodeIntrinsic('transpose', module='linalg', node_cls='Transpose'),
    'dot_product': LibNodeIntrinsic('dot_product', module='blas', node_cls='Dot'),
    # ``hlfir.matmul_transpose`` -- ``MATMUL(TRANSPOSE(A), B)`` fused.
    # ``emit_libcall`` materialises a transposed-A transient and
    # composes a Transpose + MatMul pair; the registry entry exists so
    # ``libnode_spec("matmul_transpose")`` resolves at dispatch time.
    'matmul_transpose': LibNodeIntrinsic('matmul_transpose', module='blas', node_cls='MatMul'),
}

# Generic / non-linalg standard library nodes that the bridge emits via the
# same ``kind="libcall"`` path.  Kept as a separate dict so the family
# boundary stays readable; ``libnode_spec`` looks up across both.
STANDARD: dict[str, LibNodeIntrinsic] = {
    'count': LibNodeIntrinsic('count', module='standard', node_cls='CountLibraryNode'),
    'merge': LibNodeIntrinsic('merge', module='standard', node_cls='MergeLibraryNode'),
    'argmin': LibNodeIntrinsic('argmin', module='standard', node_cls='ArgMin'),
    'argmax': LibNodeIntrinsic('argmax', module='standard', node_cls='ArgMax'),
    'cshift': LibNodeIntrinsic('cshift', module='standard', node_cls='CShift'),
}
