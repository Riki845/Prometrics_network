import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prometrics_network_analysis as pna


def test_estimators_recover_known_parameters():
    assert pna.selftest()
