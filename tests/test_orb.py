"""Tests for the flow-following strategy."""
from __future__ import annotations

import pandas as pd

from iron_condor.config import StrategyParams
from iron_condor.orb import _classify_aggressor, _strikes_near_spot


def test_classify_aggressor_buy_when_price_above_open() -> None:
    assert _classify_aggressor(trade_price=1.10, bar_open=1.05) == "buy"
    assert _classify_aggressor(trade_price=1.05, bar_open=1.05) == "buy"


def test_classify_aggressor_sell_when_price_below_open() -> None:
    assert _classify_aggressor(trade_price=1.00, bar_open=1.05) == "sell"


def test_classify_aggressor_unknown_when_bar_missing() -> None:
    assert _classify_aggressor(trade_price=1.10, bar_open=None) == "unknown"
    assert _classify_aggressor(trade_price=1.10, bar_open=float("nan")) == "unknown"


def test_strikes_near_spot_filters_by_window() -> None:
    spot = 580.0
    contracts = [
        {"strike_price": 570, "contract_type": "call"},
        {"strike_price": 578, "contract_type": "call"},
        {"strike_price": 580, "contract_type": "put"},
        {"strike_price": 583, "contract_type": "put"},
        {"strike_price": 590, "contract_type": "call"},
    ]
    out = _strikes_near_spot(spot, contracts, window=5.0)
    strikes = sorted(c["strike_price"] for c in out)
    assert strikes == [578, 580, 583]  # 570 and 590 are >5 away


def test_strikes_near_spot_skips_missing_strike() -> None:
    contracts = [
        {"strike_price": None, "contract_type": "call"},
        {"contract_type": "put"},
        {"strike_price": "not_a_number", "contract_type": "call"},
        {"strike_price": 580, "contract_type": "call"},
    ]
    out = _strikes_near_spot(580.0, contracts, window=5.0)
    assert len(out) == 1
    assert out[0]["strike_price"] == 580


def test_default_params_size_threshold() -> None:
    p = StrategyParams()
    assert p.size_threshold == 1500
    assert p.strike_window == 5.0
    assert p.skip_fridays is True
