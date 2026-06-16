"""Coverage for the Fortran source pre-processor: the ``IF (intvar)``
rewrite, the ``x**2`` / ``x**3`` -> explicit-multiply expansion, the
single/default REAL literal -> double-precision promotion, and the
OpenMP / OpenACC sentinel + ``#ifdef`` block strip.
"""

import re
from pathlib import Path

from dace_fortran.preprocess import (
    merge_used_modules,
    normalize_kind_parameters,
    preprocess_fortran,
    preprocess_fortran_source,
    promote_real_literals_to_double,
    rewrite_integer_powers,
    strip_openmp_directives,
)


def test_rewrites_bare_integer_if():
    src = """
SUBROUTINE legacy(flag)
  INTEGER :: flag
  IF (flag) THEN
    CALL do_thing()
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    assert "IF (flag /= 0)" in out
    assert "IF (flag)" not in out


def test_leaves_logical_if_alone():
    src = """
SUBROUTINE clean(p, q)
  LOGICAL :: p
  INTEGER :: q
  IF (p) THEN
    q = 1
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # ``p`` is LOGICAL -- bridge must NOT rewrite it.
    assert "IF (p)" in out
    assert "IF (p /= 0)" not in out


def test_leaves_compound_condition_alone():
    src = """
SUBROUTINE compound(a, b)
  INTEGER :: a, b
  IF (a /= 0 .AND. b > 0) THEN
    CALL do_thing()
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # The ``IF (a /= 0 .AND. ...)`` shape was already legal Fortran;
    # the rewriter only handles single-identifier conditions.
    assert "IF (a /= 0 .AND. b > 0)" in out


def test_rewrites_multi_decl_line():
    src = """
SUBROUTINE multi(flag1, flag2)
  INTEGER :: flag1, flag2
  IF (flag1) flag2 = 1
  IF (flag2) RETURN
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    assert "IF (flag1 /= 0)" in out
    assert "IF (flag2 /= 0)" in out


def test_skips_integer_arrays():
    src = """
SUBROUTINE arr(a, n)
  INTEGER :: n
  INTEGER :: a(n)
  IF (n) RETURN
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # ``n`` is a true scalar INTEGER and should be rewritten; the
    # presence of the integer array ``a(n)`` declaration must not
    # confuse the scalar-name collector.
    assert "IF (n /= 0)" in out


def test_idempotent():
    src = """
SUBROUTINE leg(f)
  INTEGER :: f
  IF (f) RETURN
END SUBROUTINE
"""
    once = preprocess_fortran(src)
    twice = preprocess_fortran(once)
    assert once == twice


def test_no_integer_decls_passthrough():
    src = """
SUBROUTINE plain(x)
  REAL(8) :: x
  x = x + 1.0D0
END SUBROUTINE
"""
    assert preprocess_fortran(src) == src


# --------------------------------------------------------------------------
# rewrite_integer_powers -- only integer-valued REAL exponents become
# repeated multiplies; bare integers / fractional powers are untouched.
# --------------------------------------------------------------------------


def test_pow_real_two_and_three():
    # Minimal change: one outer pair only -- the base is a primary so
    # each factor needs no wrapping of its own.
    assert rewrite_integer_powers("y = x**2.0") == "y = (x*x)"
    assert rewrite_integer_powers("y = x**3.0_JPRB") == "y = (x*x*x)"


def test_pow_base_parenthesised_for_precedence():
    # Already-parenthesised base keeps its own parens; the single outer
    # pair preserves precedence (2.0*(t*t), a/(b*b), -(x*x)).
    assert rewrite_integer_powers("z = (a-b)**2.0") == "z = ((a-b)*(a-b))"
    assert rewrite_integer_powers("f = 2.0*t**2.0 + a/b**2.0") == "f = 2.0*(t*t) + a/(b*b)"
    assert rewrite_integer_powers("g = -x**2.0") == "g = -(x*x)"


def test_pow_call_and_array_bases_left_untouched():
    # Duplicating a function/array reference would call twice -- unsafe
    # for impure functions and shared inlined accumulators.  Such
    # powers are left for flang's own lowering.
    assert rewrite_integer_powers("h = arr(i,j)**2.0D0") == "h = arr(i,j)**2.0D0"
    assert rewrite_integer_powers("k = s%m(2)%v**3.0") == "k = s%m(2)%v**3.0"
    assert rewrite_integer_powers("q = custom_sum(d)**2.0") == "q = custom_sum(d)**2.0"
    # A pure designator chain (no call/subscript) is still safe.
    assert rewrite_integer_powers("w = a%b%c**2.0") == "w = (a%b%c*a%b%c)"


def test_pow_leaves_bare_integer_and_fractional_alone():
    # Bare integer exponent: flang lowers x**2 correctly itself.
    assert rewrite_integer_powers("c = z**2") == "c = z**2"
    # Genuine fractional powers must stay as pow().
    assert rewrite_integer_powers("d = r**0.5_JPRB") == "d = r**0.5_JPRB"
    assert rewrite_integer_powers("e = w**2.5") == "e = w**2.5"
    assert rewrite_integer_powers("p = rho**0.78") == "p = rho**0.78"


def test_pow_comment_untouched_and_idempotent():
    assert rewrite_integer_powers("z = a**2.0  ! b**2.0 keep") == "z = (a*a)  ! b**2.0 keep"
    once = rewrite_integer_powers("v = (p-q)**2.0 + zt**3.0_JPRB")
    assert rewrite_integer_powers(once) == once


# --------------------------------------------------------------------------
# promote_real_literals_to_double -- single/default REAL literals become
# explicit double; already-double and integers are left as-is.
# --------------------------------------------------------------------------


def test_double_bare_and_single_kind():
    assert promote_real_literals_to_double("x = 2.0") == "x = 2.0D0"
    assert promote_real_literals_to_double("y = 4.2_JPRM + 1.0_4") == "y = 4.2D0 + 1.0D0"
    assert promote_real_literals_to_double("b = 1.0e-3 + .5 + 1.") == "b = 1.0D-3 + .5D0 + 1.D0"


def test_double_leaves_already_double_and_integers():
    # _JPRB / _8 / D-exponent are already double.
    assert promote_real_literals_to_double("z = 0.85E5_JPRB + 1.5D0 + 1.0_8") == "z = 0.85E5_JPRB + 1.5D0 + 1.0_8"
    # Integers and kind selectors must not be touched.
    assert promote_real_literals_to_double("n = 137 + i*2") == "n = 137 + i*2"
    assert promote_real_literals_to_double("REAL(KIND=8) :: q") == "REAL(KIND=8) :: q"
    assert promote_real_literals_to_double("k = SELECTED_REAL_KIND(13,300)") == "k = SELECTED_REAL_KIND(13,300)"


def test_double_skips_identifiers_strings_comments():
    assert promote_real_literals_to_double("r = R2ES + X1 + a2b") == "r = R2ES + X1 + a2b"
    assert promote_real_literals_to_double("msg = 'keep 2.0 here'  ! 3.0 too") == "msg = 'keep 2.0 here'  ! 3.0 too"
    assert promote_real_literals_to_double("u = 6.0 ! 7.0 stays") == "u = 6.0D0 ! 7.0 stays"


def test_double_idempotent():
    once = promote_real_literals_to_double("v = 2.0 + 0.85E5 + 1.0_JPRM")
    assert promote_real_literals_to_double(once) == once


# --------------------------------------------------------------------------
# strip_openmp_directives -- OpenMP / OpenACC sentinel lines, the
# ICON ``omp_definitions.inc`` cpp include, and ``#ifdef _OPENMP`` /
# ``#ifdef _OPENACC`` blocks are removed; unrelated cpp passes through.
# --------------------------------------------------------------------------


def test_strip_openmp_acc_sentinels():
    src = ("subroutine k(a, n)\n"
           "  real(8) :: a(n)\n"
           "  integer :: i, n\n"
           "!$OMP PARALLEL DO\n"
           "  do i = 1, n\n"
           "!$ACC LOOP VECTOR\n"
           "    a(i) = a(i) + 1.0D0\n"
           "  end do\n"
           "!$OMP END PARALLEL DO\n"
           "end subroutine\n")
    out = strip_openmp_directives(src)
    assert "!$OMP" not in out
    assert "!$ACC" not in out
    # Real code is preserved.
    assert "do i = 1, n" in out and "a(i) = a(i) + 1.0D0" in out


def test_strip_openmp_continuation_and_conditional():
    src = ("subroutine k\n"
           "  integer :: i\n"
           "!$OMP PARALLEL DEFAULT(SHARED) &\n"
           "!$OMP&   PRIVATE(i)\n"
           "!$ i = 0\n"
           "  i = 1\n"
           "end subroutine\n")
    out = strip_openmp_directives(src)
    assert "!$OMP" not in out and "!$ " not in out
    assert "i = 1" in out
    # The `!$ i = 0` conditional line is OMP-only -> dropped.
    assert "i = 0" not in out


def test_strip_omp_acc_ifdef_blocks_and_else():
    src = ("subroutine k(a, n)\n"
           "  integer :: n\n"
           "  real(8) :: a(n)\n"
           "#ifdef _OPENACC\n"
           "  call acc_only_path(a, n)\n"
           "#else\n"
           "  call host_path(a, n)\n"
           "#endif\n"
           "#ifdef _OPENMP\n"
           "  call omp_path(a, n)\n"
           "#endif\n"
           "#ifndef _OPENMP\n"
           "  call serial_fallback(a, n)\n"
           "#endif\n"
           "end subroutine\n")
    out = strip_openmp_directives(src)
    # OPENACC body dropped, #else body kept.
    assert "acc_only_path" not in out and "host_path" in out
    # OPENMP body dropped.
    assert "omp_path" not in out
    # !OPENMP body kept.
    assert "serial_fallback" in out
    # No `#ifdef _OPENMP` / `#ifdef _OPENACC` directive lines survive.
    assert "_OPENACC" not in out and "_OPENMP" not in out


def test_strip_omp_acc_passes_through_unrelated_cpp():
    src = ("subroutine k(a)\n"
           "  real(8) :: a(:)\n"
           "#ifdef __SWAPDIM\n"
           "  a = a + 1.0D0\n"
           "#else\n"
           "  a = a - 1.0D0\n"
           "#endif\n"
           "end subroutine\n")
    out = strip_openmp_directives(src)
    # Unrelated `#ifdef __SWAPDIM` block is untouched (both directives
    # AND both branches survive -- evaluating it is not this pass's job).
    assert "#ifdef __SWAPDIM" in out and "#else" in out and "#endif" in out
    assert "a + 1.0D0" in out and "a - 1.0D0" in out


def test_strip_omp_drops_omp_definitions_include():
    src = ("module m\n"
           "#include \"omp_definitions.inc\"\n"
           "#include \"hamocc_omp_definitions.inc\"\n"
           "#include \"icon_definitions.inc\"\n"
           "  implicit none\n"
           "end module\n")
    out = strip_openmp_directives(src)
    assert "omp_definitions.inc" not in out
    assert "hamocc_omp_definitions.inc" not in out
    # Unrelated icon_definitions.inc include must survive.
    assert "icon_definitions.inc" in out


def test_strip_omp_handles_defined_paren_form():
    src = ("subroutine k\n"
           "#if defined(_OPENMP)\n"
           "  call omp_only(); call omp_only_2()\n"
           "#endif\n"
           "#if !defined(_OPENACC)\n"
           "  call host_fallback()\n"
           "#endif\n"
           "end subroutine\n")
    out = strip_openmp_directives(src)
    assert "omp_only" not in out
    assert "host_fallback" in out
    assert "defined(_OPENMP)" not in out
    assert "defined(_OPENACC)" not in out


def test_strip_openmp_idempotent_and_clean_passthrough():
    clean = "subroutine k\n  integer :: i\n  i = 1\nend subroutine\n"
    assert strip_openmp_directives(clean) == clean
    noisy = ("subroutine k\n"
             "!$OMP PARALLEL DO\n"
             "#ifdef _OPENMP\n"
             "  call omp()\n"
             "#endif\n"
             "  i = 1\n"
             "end subroutine\n")
    once = strip_openmp_directives(noisy)
    twice = strip_openmp_directives(once)
    assert once == twice


# --------------------------------------------------------------------------
# merge_used_modules -- inlining USE'd modules.  Cover the two
# correctness traps the bridge hits on real ICON sources:
#   (1) blocks must be separated by a newline (a module file whose
#       final END MODULE lacks a trailing \n would otherwise glue
#       into the next module's MODULE opener);
#   (2) cpp/comment preamble above a MODULE opener must travel with
#       the module so a leading #include "<defs>.inc" survives the
#       extraction (without that, macros it defines never expand and
#       flang errors on bare macro invocations downstream).
# --------------------------------------------------------------------------


def test_merge_inserts_newline_between_module_blocks(tmp_path):
    """Two module files whose ``END MODULE`` lines lack trailing
    newlines must not glue together in the merged output."""
    (tmp_path / "mod_a.f90").write_text("MODULE mod_a\nEND MODULE mod_a")  # no trailing \n
    (tmp_path / "mod_b.f90").write_text("MODULE mod_b\nEND MODULE mod_b")  # no trailing \n
    src = "subroutine k\n  use mod_a\n  use mod_b\nend subroutine\n"
    out = merge_used_modules(src, search_dirs=[tmp_path])
    # Both modules survive, neither glues into the next opener.
    assert "MODULE mod_a" in out and "MODULE mod_b" in out
    assert "END MODULE mod_aMODULE mod_b" not in out
    assert "END MODULE mod_a\n" in out  # the inserted separator landed


def test_merge_carries_leading_cpp_include_with_its_module(tmp_path):
    """A ``#include "defs.inc"`` above a ``MODULE`` opener must be
    captured into the module's block; otherwise the macros that
    header defines vanish from the merged source and downstream
    references to them break (the ICON failure mode)."""
    (tmp_path / "mod_a.f90").write_text("! header comment\n"
                                        "#include \"defs.inc\"\n"
                                        "#define LOCAL_MACRO 1\n"
                                        "MODULE mod_a\n"
                                        "  integer :: x = LOCAL_MACRO\n"
                                        "END MODULE mod_a\n")
    src = "subroutine k\n  use mod_a\nend subroutine\n"
    out = merge_used_modules(src, search_dirs=[tmp_path])
    # Module is inlined, AND its preceding cpp preamble + comment are
    # carried with it -- so cpp will resolve the include/macros.
    assert "MODULE mod_a" in out
    assert "#include \"defs.inc\"" in out
    assert "#define LOCAL_MACRO 1" in out
    assert "! header comment" in out


def test_merge_preamble_does_not_bleed_previous_module_body(tmp_path):
    """When one source file holds two modules back-to-back, the
    second module's preamble walk must stop at the first module's
    ``END MODULE`` -- it cannot retroactively pull part of mod_a
    into mod_b's block."""
    (tmp_path / "two_mods.f90").write_text("MODULE mod_a\n"
                                           "  integer :: a_value = 1\n"
                                           "END MODULE mod_a\n"
                                           "! comment between\n"
                                           "#define SHARED_MACRO 7\n"
                                           "MODULE mod_b\n"
                                           "  integer :: b_value = SHARED_MACRO\n"
                                           "END MODULE mod_b\n")
    src = "subroutine k\n  use mod_a\n  use mod_b\nend subroutine\n"
    out = merge_used_modules(src, search_dirs=[tmp_path])
    # Both modules present.
    assert "MODULE mod_a" in out and "MODULE mod_b" in out
    # The between-modules comment / #define attach to mod_b (its
    # preamble), not mod_a's body.  ``a_value = 1`` must NOT be
    # repeated and the ``#define`` lands between the two modules.
    assert out.count("a_value = 1") == 1
    assert "#define SHARED_MACRO 7" in out


def test_merge_passthrough_for_self_contained_source(tmp_path):
    """A source that USEs only intrinsic modules (or no external
    modules at all) is returned unchanged -- merge is a no-op."""
    src = ("subroutine k(a, n)\n"
           "  use iso_c_binding\n"
           "  integer :: n\n"
           "  real(8) :: a(n)\n"
           "  a(1) = 0.0d0\n"
           "end subroutine\n")
    assert merge_used_modules(src, search_dirs=[tmp_path]) == src


# ---------------------------------------------------------------------------
# merge_engine selector -- regex (default) vs the fparser AST inliner
# ---------------------------------------------------------------------------


def _gfortran_compiles(text: str) -> bool:
    import shutil
    import subprocess
    from tempfile import TemporaryDirectory
    if not shutil.which("gfortran"):
        import pytest
        pytest.skip("gfortran not on PATH")
    with TemporaryDirectory() as td:
        f = Path(td) / "m.f90"
        f.write_text(text)
        r = subprocess.run(["gfortran", "-shared", "-fPIC", "-ffree-line-length-none", "-c",
                            str(f)],
                           cwd=td,
                           capture_output=True)
        if r.returncode:
            print(r.stderr.decode())
        return r.returncode == 0


def _two_module_project(tmp_path):
    """A driver module that ``USE``s a helper module across files -- the shape
    that needs a real merge (regex splice or fparser inline)."""
    (tmp_path / "helper.f90").write_text("module helper\n"
                                         "  implicit none\n"
                                         "contains\n"
                                         "  real function dbl(x)\n"
                                         "    real, intent(in) :: x\n"
                                         "    dbl = 2.0 * x\n"
                                         "  end function dbl\n"
                                         "end module helper\n")
    driver = ("module drv\n"
              "  use helper, only: dbl\n"
              "  implicit none\n"
              "contains\n"
              "  subroutine run(a)\n"
              "    real, intent(inout) :: a\n"
              "    a = dbl(a)\n"
              "  end subroutine run\n"
              "end module drv\n")
    return driver


def test_merge_engine_fparser_produces_compilable_single_tu(tmp_path):
    """``merge_engine='fparser'`` inlines the helper module and the result is a
    single, self-contained, compilable translation unit."""
    driver = _two_module_project(tmp_path)
    out = preprocess_fortran_source(driver, search_dirs=[tmp_path], merge_engine="fparser")
    assert "FUNCTION dbl" in out
    assert "SUBROUTINE run" in out
    assert _gfortran_compiles(out)


def test_merge_engine_regex_and_fparser_both_compile(tmp_path):
    """Both engines turn the multi-file project into one compilable TU."""
    driver = _two_module_project(tmp_path)
    rgx = preprocess_fortran_source(driver, search_dirs=[tmp_path], merge_engine="regex")
    fps = preprocess_fortran_source(driver, search_dirs=[tmp_path], merge_engine="fparser")
    assert _gfortran_compiles(rgx)
    assert _gfortran_compiles(fps)


def test_merge_engine_fparser_resolves_intrinsic_and_strips_stub(tmp_path):
    """The fparser engine resolves ``USE iso_c_binding`` from a built-in stub
    while parsing, but the stub module is not emitted (the compiler ships its
    own), so ``REAL(c_double)`` lowers to ``REAL(KIND=8)``."""
    src = ("subroutine k(a)\n"
           "  use iso_c_binding, only: c_double\n"
           "  implicit none\n"
           "  real(c_double), intent(inout) :: a\n"
           "  a = a * 2.0_c_double\n"
           "end subroutine\n")
    out = preprocess_fortran_source(src, search_dirs=[tmp_path], merge_engine="fparser")
    # The stub module is not emitted; the ``USE iso_c_binding`` is kept so the
    # compiler's own intrinsic module resolves ``c_double``.  The single TU
    # compiles without a colliding stub definition.
    assert "MODULE ISO_C_BINDING" not in out.upper()
    assert _gfortran_compiles(out)


def test_merge_engine_invalid_raises(tmp_path):
    """An unknown ``merge_engine`` is a clear error."""
    import pytest
    with pytest.raises(ValueError, match="merge_engine"):
        preprocess_fortran_source("subroutine k\nend subroutine\n", search_dirs=[tmp_path], merge_engine="bogus")


# ---------------------------------------------------------------------------
# normalize_kind_parameters
# ---------------------------------------------------------------------------


def test_kind_eq_form_rewrites_to_default_fp64():
    """``REAL(KIND=wp)`` -> ``REAL(KIND=8)`` at the default fp64."""
    src = "subroutine k(x)\n  real(KIND=wp), intent(in) :: x\nend subroutine\n"
    out = normalize_kind_parameters(src)
    assert "KIND=8" in out and "KIND=wp" not in out


def test_type_paren_short_form_rewrites():
    """``REAL(wp)`` shorthand also rewrites; ``INTEGER(wp)`` likewise."""
    src = ("subroutine k(x, n)\n"
           "  real(wp) :: x\n"
           "  integer(wp) :: n\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src)
    assert "real(8)" in out
    assert "integer(8)" in out
    assert "(wp)" not in out


def test_literal_kind_suffix_rewrites():
    """``1.0_wp``, ``1.0E0_wp``, ``2_wp`` all rewrite to the literal kind."""
    src = ("subroutine k(x)\n"
           "  real(8) :: x\n"
           "  x = 1.0_wp + 2.5E0_wp + 3_wp\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src)
    assert "1.0_8" in out
    assert "2.5E0_8" in out
    assert "3_8" in out
    assert "_wp" not in out


def test_local_integer_param_binding_is_left_alone():
    """``INTEGER, PARAMETER :: wp = 8`` already resolves; don't touch
    any of the alias sites either -- the rewrite would no-op anyway,
    so leaving the source byte-identical surfaces upstream issues."""
    src = ("module m\n"
           "  integer, parameter :: wp = 8\n"
           "contains\n"
           "  subroutine k(x)\n"
           "    real(kind=wp), intent(in) :: x\n"
           "    real(kind=wp) :: y\n"
           "    y = 1.0_wp\n"
           "  end subroutine\n"
           "end module\n")
    out = normalize_kind_parameters(src)
    # The locally-bound alias is dropped from the substitution set, so
    # the original ``wp`` references survive verbatim.
    assert "kind=wp" in out
    assert "1.0_wp" in out


def test_kind_map_override_per_alias():
    """``kind_map={"wp": 4}`` lowers an fp32 build correctly."""
    src = "subroutine k(x)\n  real(KIND=wp) :: x\n  x = 1.0_wp\nend subroutine\n"
    out = normalize_kind_parameters(src, kind_map={"wp": 4})
    assert "KIND=4" in out
    assert "1.0_4" in out


def test_kind_map_none_disables_one_alias():
    """``kind_map={"wp": None}`` leaves ``wp`` alone but still
    rewrites the other defaults (``sp`` etc.)."""
    src = ("subroutine k(x, y)\n"
           "  real(KIND=wp) :: x\n"
           "  real(KIND=sp) :: y\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src, kind_map={"wp": None})
    assert "KIND=wp" in out  # wp left alone
    assert "KIND=4" in out and "KIND=sp" not in out  # sp still rewritten


def test_passthrough_flag_disables_pass():
    """``passthrough=True`` returns the input verbatim."""
    src = "subroutine k(x)\n  real(KIND=wp) :: x\nend subroutine\n"
    assert normalize_kind_parameters(src, passthrough=True) == src


def test_custom_alias_via_kind_map():
    """ECMWF-style aliases (``JPRB``/``JPIM``) aren't in the default
    table -- the caller supplies them via ``kind_map``."""
    src = ("subroutine k(x, n)\n"
           "  real(KIND=JPRB) :: x\n"
           "  integer(KIND=JPIM) :: n\n"
           "  x = 1.0_JPRB\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src, kind_map={"JPRB": 8, "JPIM": 4})
    assert "KIND=8" in out
    assert "KIND=4" in out
    assert "1.0_8" in out
    assert "JPRB" not in out and "JPIM" not in out


def test_idempotent_on_already_resolved_source():
    """A second pass over already-substituted source must be a no-op."""
    src = ("subroutine k(x)\n  real(KIND=wp) :: x\n  x = 1.0_wp\nend subroutine\n")
    once = normalize_kind_parameters(src)
    twice = normalize_kind_parameters(once)
    assert once == twice


def test_does_not_touch_strings_or_comments():
    """Kind aliases that appear inside character literals or after a
    ``!`` comment are not rewritten -- they're not real code."""
    src = ("subroutine k(x)\n"
           "  real(kind=wp) :: x  ! 1.0_wp here is a comment\n"
           '  character(len=20) :: s = "REAL(KIND=wp)"\n'
           "  x = 1.0_wp\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src)
    # Code sites rewritten.
    assert "kind=8" in out
    assert "1.0_8" in out
    # Comment + string preserved.
    assert "! 1.0_wp here is a comment" in out
    assert '"REAL(KIND=wp)"' in out


def test_does_not_touch_user_variable_with_alias_name_prefix():
    """A user identifier that *contains* ``wp`` as a substring (e.g.
    ``twp``, ``wpos``) must not be rewritten -- the pattern is bounded
    by word boundaries."""
    src = ("subroutine k(twp, wpos)\n"
           "  real(8) :: twp\n"
           "  integer :: wpos\n"
           "  twp = real(wpos, kind=8)\n"
           "end subroutine\n")
    out = normalize_kind_parameters(src)
    assert out == src


def test_handles_double_precision_alias_dp():
    """``dp`` defaults to 8 (double precision)."""
    src = "subroutine k(x)\n  real(kind=dp) :: x\nend subroutine\n"
    out = normalize_kind_parameters(src)
    assert "kind=8" in out


def test_handles_selected_real_kind_rhs():
    """``wp = SELECTED_REAL_KIND(...)`` doesn't match the integer-literal
    PARAMETER regex, so the alias falls through to the substitution
    path -- exactly the externally-defined case in disguise."""
    src = ("module m\n"
           "  integer, parameter :: wp = SELECTED_REAL_KIND(15,300)\n"
           "contains\n"
           "  subroutine k(x)\n"
           "    real(kind=wp), intent(out) :: x\n"
           "    x = 1.0_wp\n"
           "  end subroutine\n"
           "end module\n")
    out = normalize_kind_parameters(src)
    assert "kind=8" in out
    assert "1.0_8" in out


def test_preprocess_fortran_source_threads_kind_options(tmp_path):
    """The end-to-end composition forwards ``kind_map`` and
    ``kind_passthrough`` to the kind-normalisation stage."""
    src = "subroutine k(x)\n  real(KIND=wp) :: x\n  x = 1.0_wp\nend subroutine\n"

    # The fparser merge re-emits the source, normalising ``KIND=wp`` to
    # ``KIND = wp`` (spaces around ``=``); match whitespace-insensitively so
    # the assertion checks the kind value, not the emitter's spacing.
    def _kind(out):
        return re.sub(r"\s+", "", out)

    # Default: substitutes.
    assert "KIND=8" in _kind(preprocess_fortran_source(src, search_dirs=[tmp_path]))
    # Override: fp32.
    out_fp32 = preprocess_fortran_source(src, search_dirs=[tmp_path], kind_map={"wp": 4})
    assert "KIND=4" in _kind(out_fp32)
    # Passthrough: source untouched (except OpenMP strip, which is a
    # no-op here, and the integer-power rewrite, also a no-op here).
    assert "KIND=wp" in _kind(preprocess_fortran_source(src, search_dirs=[tmp_path], kind_passthrough=True))
