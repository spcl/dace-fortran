"""The disk memo behind the f2py reference builds and the generated-code warning verdicts."""

import numpy as np
import pytest

from tests import _cache
from tests._helpers import f2py

SOURCE = """\
subroutine twice(a, n, b)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: b(n)
  b = a * 2.0d0
end subroutine
"""


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DACE_FORTRAN_TEST_CACHE", str(tmp_path / "cache"))


def test_a_stored_file_is_found_by_its_own_key_only(cache_root, tmp_path):
    payload = tmp_path / "payload"
    payload.write_text("verdict")
    key = _cache.digest("tu", "flags")

    _cache.store("kind", key, payload)

    assert _cache.entry("kind", key).read_text() == "verdict"
    assert _cache.entry("kind", _cache.digest("tu", "other flags")) is None
    assert _cache.entry("other kind", key) is None


def test_digest_keeps_part_boundaries():
    assert _cache.digest("ab", "c") != _cache.digest("a", "bc")


def test_a_cache_switched_off_stores_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("DACE_FORTRAN_TEST_CACHE", "off")
    payload = tmp_path / "payload"
    payload.write_text("verdict")

    _cache.store("kind", "key", payload)

    assert _cache.entry("kind", "key") is None


def test_an_identical_f2py_reference_is_built_once(cache_root, tmp_path, monkeypatch):
    built = f2py(SOURCE, tmp_path / "first", "cached_twice")
    # a second build of the same source must come from the cache: make a real f2py run impossible
    monkeypatch.setattr("tests._helpers.f2py_build_with_retry", lambda *a, **k: pytest.fail("rebuilt"))

    reused = f2py(SOURCE, tmp_path / "second", "cached_twice")

    assert np.array_equal(reused.twice(np.arange(3.0)), built.twice(np.arange(3.0)))
    assert np.array_equal(reused.twice(np.arange(3.0)), [0.0, 2.0, 4.0])
