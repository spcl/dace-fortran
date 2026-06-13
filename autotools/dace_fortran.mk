# ============================================================================
# dace_fortran.mk -- make rules + helper macros for the dace-fortran
# preprocess CLI.
# ============================================================================
#
# Companion to ``autotools/dace_fortran.m4``: the m4 macro probes Python
# and exports ``$(DACE_FORTRAN_PYTHON)`` / ``$(DACE_FORTRAN_PREPROCESS_CLI)``;
# this file provides the make-side machinery (pattern rule + helper
# function) the user includes from their ``Makefile.am``:
#
#   # configure.ac
#   m4_include([m4/dace_fortran.m4])
#   DACE_FORTRAN_PREPROCESS
#
#   # Makefile.am
#   include $(top_srcdir)/dace_fortran.mk        # ship this file in-tree
#
#   DACE_FORTRAN_PASSES      = all_defaults rewrite_external
#   DACE_FORTRAN_SEARCH_DIRS = utils
#   mylib_a_SOURCES = $(call dace_fortran_preprocess, kernel.f90 helper.f90)
#
# The standalone ``.mk`` is the right shape: a ``@FOO@``-substituted
# variable value in Makefile.am ends up as a make-variable definition
# (the whole multi-line block becomes one variable's value), not as
# inlined rules.  An ``include`` directive embeds the rules verbatim,
# which is what we need.
#
# Required variables the m4 macro provides:
#   $(DACE_FORTRAN_PYTHON)
#   $(DACE_FORTRAN_PREPROCESS_CLI)
#   $(DACE_FORTRAN_BUILD_DIR)
# ============================================================================

DACE_FORTRAN_PASSES        ?= all_defaults
DACE_FORTRAN_SEARCH_DIRS   ?=
DACE_FORTRAN_KIND_MAP      ?=

# Translate one pass name (``rewrite_external``, ``all_defaults``, ...)
# to its CLI flag form (``--rewrite-external``, ``--all-defaults``).
# Pure make functions  --  no shell escapes, no automake quoting
# ambiguity.
_dace_fortran_to_pass_flag = --$(subst _,-,$(1))
_dace_fortran_pass_flags = $(foreach _p,$(DACE_FORTRAN_PASSES),$(call _dace_fortran_to_pass_flag,$(_p)))
_dace_fortran_search_flags = $(foreach _d,$(DACE_FORTRAN_SEARCH_DIRS),--search-dir $(_d))
_dace_fortran_kind_flags = $(foreach _km,$(DACE_FORTRAN_KIND_MAP),--kind-map $(_km))

# Pattern rule: ``foo.f90`` ->
# ``$(DACE_FORTRAN_BUILD_DIR)/foo.preprocessed.f90``.
$(DACE_FORTRAN_BUILD_DIR)/%.preprocessed.f90: %.f90
	@mkdir -p $(@D)
	$(DACE_FORTRAN_PREPROCESS_CLI) \
	    $(_dace_fortran_pass_flags) \
	    $(_dace_fortran_search_flags) \
	    $(_dace_fortran_kind_flags) \
	    --in $< --out $@

# Helper: $(call dace_fortran_preprocess, foo.f90 bar.f90) rewrites
# each input to its preprocessed sibling under
# $(DACE_FORTRAN_BUILD_DIR).
dace_fortran_preprocess = $(patsubst %.f90,$(DACE_FORTRAN_BUILD_DIR)/%.preprocessed.f90,$(1))
