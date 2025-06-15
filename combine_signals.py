"""Aggregate trading signals from the various pattern modules."""

import numpy as np
import pandas as pd

from flags_pennants import find_flags_pennants_trendline
from head_shoulders import find_hs_patterns
from mp_support_resist import (
    support_resistance_levels,
    sr_penetration_signal,
)
from harmonic_patterns import get_extremes, find_xabcd, ALL_PATTERNS


def load_data(path="BTCUSDT3600.csv"):
    data = pd.read_csv(path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
    return data


def flag_pennant_signals(close_arr, order=10):
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(close_arr, order)
    signal = np.zeros(len(close_arr))
    for p in bull_flags + bull_pennants:
        signal[p.conf_x] = 1.0
    for p in bear_flags + bear_pennants:
        signal[p.conf_x] = -1.0
    return signal


def head_shoulders_signals(close_arr, order=6):
    hs, ihs = find_hs_patterns(close_arr, order)
    signal = np.zeros(len(close_arr))
    for p in ihs:
        signal[p.break_i] = 1.0
    for p in hs:
        signal[p.break_i] = -1.0
    return signal


def harmonic_signals(ohlc, sigma=0.02, err_thresh=0.2):
    extremes = get_extremes(ohlc, sigma)
    output = find_xabcd(ohlc, extremes, err_thresh)
    combined = np.zeros(len(ohlc))
    for pat in ALL_PATTERNS:
        pat_data = output[pat.name]
        combined += pat_data["bull_signal"] + pat_data["bear_signal"]
    return combined


def sr_signals(data, lookback=365):
    levels = support_resistance_levels(data, lookback)
    return sr_penetration_signal(data, levels)



def aggregate_signals(path: str = "BTCUSDT3600.csv", include_pip_miner: bool = False) -> pd.DataFrame:
    """Return a dataframe with a signal column for each strategy.

    Parameters
    ----------
    path:
        CSV file containing OHLC data.
    include_pip_miner:
        Whether to calculate the PIP pattern miner signal. The miner relies on
        the ``pyclustering`` package and can be slow, so it is optional.
    """

    data = load_data(path)
    log_close = np.log(data["close"]).to_numpy()

    sr_sig = sr_signals(data)
    hs_sig = head_shoulders_signals(log_close)
    flag_sig = flag_pennant_signals(log_close)
    harm_sig = harmonic_signals(data)

    df = pd.DataFrame(index=data.index)
    df["sr_signal"] = sr_sig
    df["hs_signal"] = hs_sig
    df["flag_signal"] = flag_sig
    df["harmonic_signal"] = harm_sig

    if include_pip_miner:
        from wf_pip_miner import WFPIPMiner

        miner = WFPIPMiner(
            n_pips=5,
            lookback=24,
            hold_period=6,
            train_size=24 * 365 * 2,
            step_size=24 * 365,
        )
        pip_sig = np.zeros(len(log_close))
        for i in range(len(log_close)):
            pip_sig[i] = miner.update_signal(log_close, i)
        df["pip_miner_signal"] = pip_sig

    df["combined_signal"] = df.sum(axis=1)
    return df


if __name__ == "__main__":
    combined = aggregate_signals()
    print(combined.head())
