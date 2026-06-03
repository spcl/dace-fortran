dnl ===========================================================================
dnl dace_fortran.m4  --  autoconf macros for the dace-fortran preprocess CLI.
dnl ===========================================================================
dnl
dnl Provides DACE_FORTRAN_PREPROCESS and a make-rule snippet so a project
dnl using GNU Automake can wire the preprocess passes into its existing
dnl ``*.f90``-to-object compile path with a single configure-time call.
dnl
dnl Goal: minimal manual steps.  A project adds, in ``configure.ac``:
dnl
dnl     m4_include([m4/dace_fortran.m4])
dnl     DACE_FORTRAN_PREPROCESS
dnl
dnl and in its ``Makefile.am``:
dnl
dnl     @DACE_FORTRAN_RULES@
dnl     dace_fortran_passes      = all_defaults rewrite_external rewrite_string_enum
dnl     dace_fortran_search_dirs = ../utils
dnl     mylib_a_SOURCES          = $(call dace_fortran_preprocess, \
dnl                                  kernel.f90 utils/helper.f90)
dnl
dnl The macro:
dnl   * Locates a Python 3 with the ``dace_fortran`` package importable.
dnl   * Exports ``$(DACE_FORTRAN_PYTHON)``, ``$(DACE_FORTRAN_PREPROCESS)``,
dnl     and a generic ``%.preprocessed.f90: %.f90`` pattern rule
dnl     substituted into the Makefile via ``@DACE_FORTRAN_RULES@``.
dnl   * Provides the ``$(call dace_fortran_preprocess, ...)`` macro that
dnl     remaps each input ``foo.f90`` to its preprocessed sibling
dnl     ``$(DACE_FORTRAN_BUILD_DIR)/foo.preprocessed.f90``.
dnl
dnl Requirements:
dnl   * autoconf 2.69 or newer.
dnl   * Python 3 with the ``dace_fortran`` package; AC_SUBST'd as
dnl     ``$(DACE_FORTRAN_PYTHON)``.
dnl
dnl ===========================================================================

AC_DEFUN([DACE_FORTRAN_PREPROCESS], [
    AC_REQUIRE([AC_PROG_MAKE_SET])

    dnl ----- 1. Locate Python ------------------------------------------------
    AC_ARG_VAR([DACE_FORTRAN_PYTHON],
               [Python 3 interpreter for the dace-fortran preprocess CLI])
    if test -z "$DACE_FORTRAN_PYTHON"; then
        AC_PATH_PROGS([DACE_FORTRAN_PYTHON], [python3 python])
    fi
    if test -z "$DACE_FORTRAN_PYTHON"; then
        AC_MSG_ERROR([no Python 3 found.  Set DACE_FORTRAN_PYTHON to a path.])
    fi

    dnl ----- 2. Probe importability of the dace_fortran package --------------
    AC_MSG_CHECKING([whether $DACE_FORTRAN_PYTHON can import dace_fortran])
    if $DACE_FORTRAN_PYTHON -c "import dace_fortran.preprocess_cli" \
            > /dev/null 2>&1; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
        AC_MSG_ERROR([the dace_fortran Python package is not importable
                      from $DACE_FORTRAN_PYTHON.
                      Install it (``pip install dace-fortran`` from a
                      checkout) or set DACE_FORTRAN_PYTHON to an
                      interpreter that has it.])
    fi

    dnl ----- 3. Export the CLI invocation + a make-rule snippet --------------
    DACE_FORTRAN_PREPROCESS_CLI="$DACE_FORTRAN_PYTHON -m dace_fortran.preprocess_cli"
    AC_SUBST([DACE_FORTRAN_PREPROCESS_CLI])

    dnl Default output directory inside the build tree.  Users can
    dnl override per-Makefile.am via ``DACE_FORTRAN_BUILD_DIR := ...``.
    if test -z "$DACE_FORTRAN_BUILD_DIR"; then
        DACE_FORTRAN_BUILD_DIR='$(top_builddir)/dace_fortran_preprocessed'
    fi
    AC_SUBST([DACE_FORTRAN_BUILD_DIR])

    dnl The make snippet that downstream Makefile.am uses via
    dnl ``@DACE_FORTRAN_RULES@``.  Declares the pattern rule that turns
    dnl ``foo.f90`` into ``$(DACE_FORTRAN_BUILD_DIR)/foo.preprocessed.f90``
    dnl plus the ``dace_fortran_preprocess`` helper macro.
    DACE_FORTRAN_RULES='# --- dace-fortran preprocess auto-rules (configure-generated) ---
DACE_FORTRAN_PASSES        ?= all_defaults
DACE_FORTRAN_SEARCH_DIRS   ?=
DACE_FORTRAN_KIND_MAP      ?=

dace_fortran_pass_flags = $$(echo $(DACE_FORTRAN_PASSES) | \
                              sed -e "s/all_defaults/--all-defaults/g" \
                                  -e "s/\\([[a-z_]]*\\)/--\\1/g" | \
                              tr "_" "-")
dace_fortran_search_flags = $$(for d in $(DACE_FORTRAN_SEARCH_DIRS); do \
                                  echo --search-dir $$d; done)
dace_fortran_kind_flags = $$(for km in $(DACE_FORTRAN_KIND_MAP); do \
                                echo --kind-map $$km; done)

$(DACE_FORTRAN_BUILD_DIR)/%.preprocessed.f90: %.f90
	@mkdir -p $$(dirname $$@)
	$(DACE_FORTRAN_PREPROCESS_CLI) \
	    $(dace_fortran_pass_flags) \
	    $(dace_fortran_search_flags) \
	    $(dace_fortran_kind_flags) \
	    --in $$< --out $$@

# Helper: $$(call dace_fortran_preprocess, list of .f90 files)
# rewrites each ``foo.f90`` to its preprocessed sibling.
dace_fortran_preprocess = $$(patsubst %.f90, \
        $(DACE_FORTRAN_BUILD_DIR)/%.preprocessed.f90,$$(1))
# --- end dace-fortran preprocess auto-rules ---
'
    AC_SUBST([DACE_FORTRAN_RULES])
])
