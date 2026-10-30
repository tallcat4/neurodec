# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest

from neurodec.blackbox.paths import encoder_bin

needs_codec = pytest.mark.skipif(not encoder_bin().exists(), reason="build ambe2-c first (make -C ../ambe2-c)")
needs_mrl = pytest.mark.skipif(not encoder_bin("mrl").exists(), reason="build mrl-celp-c first (make -C ../mrl-celp-c)")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
