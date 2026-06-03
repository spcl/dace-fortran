! MATMUL(TRANSPOSE(A), TRANSPOSE(B)) -- two separate transposes plus
! a matmul under the default pipeline; the optimised pipeline does NOT
! emit a fused op for this case (only LHS-transpose has one).
SUBROUTINE matmul_transpose_both(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(m, n)
  REAL(8), INTENT(IN) :: B(k, m)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(TRANSPOSE(A), TRANSPOSE(B))
END SUBROUTINE matmul_transpose_both
