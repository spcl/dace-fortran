! MATMUL(TRANSPOSE(A), B) -- the optimised ``hlfir.matmul_transpose`` op
! is only emitted by the ``hlfir-optimized-bufferization`` pass, which
! we do not run.  In our default ``flang-new -fc1 -emit-hlfir`` flow
! the expression lowers as a separate ``hlfir.transpose`` plus
! ``hlfir.matmul``, both of which the bridge already recognises.
SUBROUTINE matmul_transpose_run(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(m, n)
  REAL(8), INTENT(IN) :: B(m, k)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(TRANSPOSE(A), B)
END SUBROUTINE matmul_transpose_run
