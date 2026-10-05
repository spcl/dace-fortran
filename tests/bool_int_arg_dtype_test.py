"""Boolean/integer argument dtype correctness. Regression guard for the cloudsc_full PLUDE
diagnosis (a LOGICAL array read with the wrong element width corrupted LDCUM across element
boundaries): a default LOGICAL is its 4-byte storage on the SDFG signature. E2e against an
f2py-compiled reference."""

import numpy as np

from tests._util import build_sdfg
from tests._helpers import f2py


def test_bool_logical_array_pass_through(tmp_path):
    """LOGICAL 1-D input read element-by-element, alternating True/False -- surfaces any
    byte-stride bug."""
    src = """
SUBROUTINE bool_pass(flags, out, n)
integer, intent(in) :: n
logical, intent(in) :: flags(n)
integer, intent(out) :: out(n)
integer i
DO i = 1, n
    IF (flags(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE bool_pass
"""
    ref = f2py(src, tmp_path / "ref", "bool_pass_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="bool_pass", entry="bool_pass").build()

    flags = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    n = flags.size
    # f2py returns intent(out) arrays.
    out_ref = ref.bool_pass(flags)

    out = np.zeros(n, dtype=np.int32)
    sdfg(flags=flags.astype(np.uint32), out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_bool_logical_array_2d_pass_through(tmp_path):
    """2-D ``LOGICAL`` input (the LDCUM shape)."""
    src = """
SUBROUTINE bool_2d_pass(flags, out, klon, nblocks)
integer, intent(in) :: klon, nblocks
logical, intent(in) :: flags(klon, nblocks)
integer, intent(out) :: out(klon, nblocks)
integer i, j
DO j = 1, nblocks
    DO i = 1, klon
        IF (flags(i, j)) THEN
            out(i, j) = 1
        ELSE
            out(i, j) = 0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE bool_2d_pass
"""
    ref = f2py(src, tmp_path / "ref", "bool_2d_pass_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="bool_2d_pass", entry="bool_2d_pass").build()

    klon, nblocks = 1, 4
    flags = np.array([[False, True, True, False]], dtype=np.bool_, order="F")
    out_ref = ref.bool_2d_pass(flags)

    out = np.zeros((klon, nblocks), dtype=np.int32, order="F")
    sdfg(flags=np.asfortranarray(flags.astype(np.uint32)), out=out, klon=klon, nblocks=nblocks)
    np.testing.assert_array_equal(out, out_ref)


def test_int32_array_pass_through(tmp_path):
    """Plain INTEGER input pass-through."""
    src = """
SUBROUTINE int_double(inp, out, n)
integer, intent(in) :: n
integer, intent(in) :: inp(n)
integer, intent(out) :: out(n)
integer i
DO i = 1, n
    out(i) = inp(i) * 2
ENDDO
END SUBROUTINE int_double
"""
    ref = f2py(src, tmp_path / "ref", "int_double_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="int_double", entry="int_double").build()

    inp = np.array([1, 2, 3, 100, -5, 1000000], dtype=np.int32)
    n = inp.size
    out_ref = ref.int_double(inp)

    out = np.zeros(n, dtype=np.int32)
    sdfg(inp=inp, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_bool_scalar_logical_pass_through(tmp_path):
    """Scalar LOGICAL passed in the storage dtype the SDFG declares for it."""
    src = """
SUBROUTINE bool_scalar(flag, out, n)
integer, intent(in) :: n
logical, intent(in) :: flag
integer, intent(out) :: out(n)
integer i
DO i = 1, n
    IF (flag) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE bool_scalar
"""
    ref = f2py(src, tmp_path / "ref", "bool_scalar_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="bool_scalar", entry="bool_scalar").build()

    from dace.data import Scalar

    desc = sdfg.arglist().get("flag")

    def _route_bool(v):
        """Route scalar LOGICAL to whatever the bridge declared (``Scalar`` or ``Array(1,)``)
        in its LOGICAL storage dtype."""
        storage = desc.dtype.as_numpy_dtype()
        if isinstance(desc, Scalar):
            return storage.type(v)
        return np.array([v], dtype=storage)

    n = 5
    out_ref = ref.bool_scalar(True, n=n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(flag=_route_bool(True), out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)

    out_ref = ref.bool_scalar(False, n=n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(flag=_route_bool(False), out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
