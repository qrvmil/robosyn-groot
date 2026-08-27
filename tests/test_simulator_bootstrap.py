from tools.bootstrap_simulator import DEXSIM_SHA256, DEXSIM_WHEEL
from tools.evaluation import EMBODICHAIN_COMMIT, ROBOSYN_COMMIT


def test_simulator_bootstrap_uses_pinned_sources_and_wheel():
    assert len(DEXSIM_SHA256) == 64
    assert DEXSIM_WHEEL == "dexsim_engine-0.4.3-cp311-cp311-manylinux_2_31_x86_64.whl"
    assert EMBODICHAIN_COMMIT == "9ebee30011f378f94a7cbe78b01d8c2eacba231a"
    assert ROBOSYN_COMMIT == "93f95f898b76548cc259d20e2b90860a6f79120d"
