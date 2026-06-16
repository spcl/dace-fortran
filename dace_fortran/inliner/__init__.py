# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vendored fparser AST desugaring / inlining package.

This subpackage is a faithful copy of the upstream DaCe Fortran
frontend's ``ast_desugaring`` package plus a minimal ``ast_utils`` shim,
trimmed to exactly what the source-text single-TU inliner
(:mod:`dace_fortran.fparser_inliner`) needs.  The modules are kept
import-compatible with upstream (``from .. import ast_utils``) so they
require no source edits beyond the one relocated import in ``utils.py``.

The public, user-facing API lives in :mod:`dace_fortran.fparser_inliner`
(``inline_to_single_tu`` / ``inline_to_ast``); the modules here are the
implementation passes.
"""
