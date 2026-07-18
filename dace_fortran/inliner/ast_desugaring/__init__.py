"""AST desugaring and analysis tools for the Fortran frontend.

Desugaring rewrites high-level Fortran constructs the internal AST can't
represent directly into simpler, SDFG-buildable equivalents. Analysis gathers
scope/type/shape info used by both desugaring and SDFG generation.
"""
