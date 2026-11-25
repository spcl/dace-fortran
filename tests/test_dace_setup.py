# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""dace_setup leaves a git, path, or editable DaCe alone and honours the opt-out."""

import os
from unittest import mock

from dace_fortran import dace_setup


def test_a_dace_not_from_pypi_is_left_alone():
    with mock.patch.object(dace_setup, "installed_from_pypi", return_value=False):
        assert dace_setup.ensure_dace() is False


def test_the_opt_out_skips_the_switch():
    with (
        mock.patch.dict(os.environ, {dace_setup.OPT_OUT: "1"}),
        mock.patch.object(dace_setup, "installed_from_pypi", return_value=True),
    ):
        assert dace_setup.ensure_dace() is False


def test_the_install_command_names_the_branch():
    assert f"dace @ git+{dace_setup.DACE_REPOSITORY}@FaCe" in dace_setup.install_command("FaCe")


if __name__ == "__main__":
    test_a_dace_not_from_pypi_is_left_alone()
    test_the_opt_out_skips_the_switch()
    test_the_install_command_names_the_branch()
