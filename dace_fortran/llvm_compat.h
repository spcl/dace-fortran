// ============================================================================
// llvm_compat.h  --  LLVM-major-version shims for the bridge (LLVM 21 and 22).
// ============================================================================
// Every construct whose spelling differs between the supported majors lives
// here behind LLVM_VERSION_MAJOR, so the passes stay version-agnostic and
// adding a major is a one-file edit.  Keep in sync with CMakeLists.txt's
// LLVM_SUPPORTED_VERSIONS and llvm_toolchain.py's SUPPORTED_LLVM_VERSIONS.
// ============================================================================
#pragma once

#include <cstdint>

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Config/llvm-config.h"
#include "mlir/IR/Builders.h"

namespace hlfir_bridge {

/// ``hlfir.declare`` through the explicit-result-types builder, which pins both
/// result types instead of letting the short builder infer them.
///
/// LLVM 22 gave ``hlfir.declare`` the ``storage`` / ``storage_offset`` operands
/// (FortranVariableStorageOpInterface) plus ``skip_rebox`` / ``dummy_arg_no``, so
/// its generated builder takes four more arguments than LLVM 21's.  All four are
/// absent/default here: these declares describe freshly allocated local temps.
inline hlfir::DeclareOp createDeclare(mlir::OpBuilder& b, mlir::Location loc, mlir::Type resultType0,
                                      mlir::Type resultType1, mlir::Value memref, mlir::Value shape,
                                      mlir::ValueRange typeparams, mlir::StringAttr uniqName) {
#if LLVM_VERSION_MAJOR >= 22
  return b.create<hlfir::DeclareOp>(loc, resultType0, resultType1, memref, shape, typeparams,
                                    /*dummy_scope=*/mlir::Value{},
                                    /*storage=*/mlir::Value{},
                                    /*storage_offset=*/static_cast<std::uint64_t>(0), uniqName,
                                    /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
                                    /*data_attr=*/cuf::DataAttributeAttr{},
                                    /*skip_rebox=*/mlir::UnitAttr{},
                                    /*dummy_arg_no=*/mlir::IntegerAttr{});
#else
  return b.create<hlfir::DeclareOp>(loc, resultType0, resultType1, memref, shape, typeparams,
                                    /*dummy_scope=*/mlir::Value{}, uniqName,
                                    /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
                                    /*data_attr=*/cuf::DataAttributeAttr{});
#endif
}

/// One entry per ``hlfir.declare`` operand segment; a short array silently reads past its end.
inline mlir::NamedAttribute declareSegments(mlir::OpBuilder& b, bool hasShape) {
  llvm::SmallVector<std::int32_t, 5> sizes{1, hasShape ? 1 : 0, 0, 0};
#if LLVM_VERSION_MAJOR >= 22
  sizes.push_back(0);
#endif
  return b.getNamedAttr("operandSegmentSizes", b.getDenseI32ArrayAttr(sizes));
}

}  // namespace hlfir_bridge
