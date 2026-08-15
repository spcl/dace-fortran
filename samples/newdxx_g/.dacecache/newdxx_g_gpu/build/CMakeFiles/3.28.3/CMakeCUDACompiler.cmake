set(CMAKE_CUDA_COMPILER "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/bin/g++")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "13.3.33")
set(CMAKE_CUDA_DEVICE_LINKER "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "20")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "16.1")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "13.3.33")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "90-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/lib/stubs;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/fftw-3.3.11-vetzoguk5rdtmnbog7s75bgtzlkf2r5k/include;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/include/cccl;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/include/c++/16.1.0;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/include/c++/16.1.0/aarch64-unknown-linux-gnu;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/include/c++/16.1.0/backward;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/lib/gcc/aarch64-unknown-linux-gnu/16.1.0/include;/usr/local/include;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/include;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/lib/gcc/aarch64-unknown-linux-gnu/16.1.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;atomic_asneeded;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/lib/stubs;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/cuda-13.3.0-ickio72otm2ifsockd42dw425cntauby/targets/sbsa-linux/lib;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/binutils-2.46.0-tqnyevg2jwxtbgtqnsbggcc2i3clzcxv/bin;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/lib/gcc/aarch64-unknown-linux-gnu/16.1.0;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/lib64;/lib64;/usr/lib64;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/fftw-3.3.11-vetzoguk5rdtmnbog7s75bgtzlkf2r5k/lib;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/nvhpc-26.3-jdwehzfw7nrgblbshjqgmy5dezpi5aur/Linux_aarch64/26.3/compilers/lib;/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/gcc-16.1.0-v2vbdi5nbakakjrfjfrmgdeo6bmsah75/lib;/lib;/usr/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/binutils-2.46.0-tqnyevg2jwxtbgtqnsbggcc2i3clzcxv/bin/ld")
set(CMAKE_AR "/capstor/scratch/cscs/ybudanaz/aarch64/spack/linux-neoverse_v2/binutils-2.46.0-tqnyevg2jwxtbgtqnsbggcc2i3clzcxv/bin/ar")
set(CMAKE_MT "")
