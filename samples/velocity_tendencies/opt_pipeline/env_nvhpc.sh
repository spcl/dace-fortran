# Single-vendor toolchain for the velocity GPU builds: nvcc, nvc++ and nvfortran all come
# from nvhpc, so the DaCe binary and the OpenACC Fortran reference are built by the same
# compiler family and the 1.5 comparison is not confounded by the host compiler.
#
# Source it, never execute it:  . ./env_nvhpc.sh
eval "$(spack load --sh nvhpc@26.3)"

# nvhpc's shipped localrc was generated against gcc@16.1.0, whose libstdc++ headers nvc++ 26.3
# cannot parse (binders.h / functional fail with "expected a type specifier"). This rc rebinds
# it to the system gcc 14.2, the newest GNU toolchain this nvhpc supports. Regenerate with:
#   makelocalrc -x -d $NVLOCALRC_DIR -gcc /usr/bin/gcc-14 -gpp /usr/bin/g++-14 <nvhpc compilers dir>
export NVLOCALRC=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran/samples/_work/nvhpc_localrc/localrc

export _USE_NVHPC=1
export CXX=nvc++
export GENCODE_NUMBER="${GENCODE_NUMBER:-90a}"
