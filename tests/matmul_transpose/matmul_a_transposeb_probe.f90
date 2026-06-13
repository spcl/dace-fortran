! MATMUL(A, TRANSPOSE(B)) -- lowers as a separate hlfir.transpose(B)
! followed by hlfir.matmul.
SUBROUTINE matmul_a_transposeb(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(n, m)
  REAL(8), INTENT(IN) :: B(k, m)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(A, TRANSPOSE(B))
END SUBROUTINE matmul_a_transposeb
