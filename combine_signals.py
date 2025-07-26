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

# ``StandardScaler`` is used to normalise the various signal columns so that the
# resulting dataframe can be directly fed into machine learning models. The
# import is optional to keep the core functionality lightweight when scaling is
# not required.
try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - only executed if sklearn missing
    StandardScaler = None

# Optional dependency used for additional technical indicators. The library
# provides dozens of common indicators which we expose as extra columns in the
# combined dataframe. The import is performed lazily inside ``ta_indicators`` to
# avoid the dependency when not required.


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


def ta_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe of technical indicator columns.

    This function relies on the ``ta`` package which includes a wide range of
    indicators. If the package is not installed, an informative error will be
    raised.
    """

    try:
        from ta import add_all_ta_features
    except Exception as exc:  # pragma: no cover - only executed if ta missing
        raise ImportError("The 'ta' package is required for additional indicators" ) from exc

    df = data.copy()
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )
    extra_cols = [c for c in df.columns if c not in data.columns]
    return df[extra_cols]



def aggregate_signals(path: str = "BTCUSDT3600.csv", include_pip_miner: bool = False) -> pd.DataFrame:
    """Return a dataframe with OHLC values and scaled trading signals.

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
    ta_df = ta_indicators(data)

    signal_df = pd.DataFrame(index=data.index)
    signal_df["sr_signal"] = sr_sig
    signal_df["hs_signal"] = hs_sig
    signal_df["flag_signal"] = flag_sig
    signal_df["harmonic_signal"] = harm_sig
    for col in ta_df.columns:
        signal_df[col] = ta_df[col]

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
        signal_df["pip_miner_signal"] = pip_sig

    # Scale all signal columns so the features have comparable magnitudes
    if StandardScaler is not None:
        scaler = StandardScaler()
        signal_df[signal_df.columns] = scaler.fit_transform(signal_df)

    signal_df["combined_signal"] = signal_df.sum(axis=1)

    # Add OHLC columns to the returned dataframe
    df = pd.DataFrame(index=data.index)
    for col in ["open", "high", "low", "close"]:
        if col in data.columns:
            df[col] = data[col]
    df = pd.concat([df, signal_df], axis=1)
    return df


if __name__ == "__main__":
    combined = aggregate_signals()
    print(combined.head())
