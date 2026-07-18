# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vendored fparser AST desugaring / inlining package.

Faithful copy of upstream DaCe Fortran frontend's ``ast_desugaring`` package
plus a minimal ``ast_utils`` shim, trimmed to what the source-text single-TU
inliner (:mod:`dace_fortran.fparser_inliner`) needs.  Kept import-compatible
with upstream (``from .. import ast_utils``).  Public API is in
:mod:`dace_fortran.fparser_inliner` (``inline_to_single_tu``/``inline_to_ast``).
"""
