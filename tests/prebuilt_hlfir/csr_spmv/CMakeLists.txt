# CSR sparse matrix-vector multiplication -- second mock project
# for the prebuilt-HLFIR pipeline test.  Deliberately tiny + no
# external deps (no MPI, no netCDF) so the helper's general path
# is exercised on a structurally-different project than the
# jacobi sibling.

cmake_minimum_required(VERSION 3.18)
project(prebuilt_hlfir_csr_spmv Fortran)

add_executable(csr_demo
    mod_csr.f90
    main.f90)
