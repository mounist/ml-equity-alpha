"""
Earnings surprise (SUE) feature from IBES.

Look-ahead bias control: SUE is computed from the actual EPS announcement
and the most recent consensus estimate prior to announcement (statpers < anndats).
The signal is then propagated to month-end only if the announcement occurred
at least 1 trading day before month-end. Signals older than 90 days are
set to NaN to prevent stale information from persisting.
"""

import numpy as np
import pandas as pd

from config import START_DATE, END_DATE


def compute_sue(
    ibes_actuals: pd.DataFrame,
    ibes_statsum: pd.DataFrame,
    ibes_crsp_link: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Standardised Unexpected Earnings (SUE) and map to a monthly panel.

    Parameters
    ----------
    ibes_actuals :  [ibes_ticker, pends, anndats, eps_actual]
    ibes_statsum :  [ibes_ticker, statpers, fpedats, meanest, stdev, numest]
    ibes_crsp_link: [ibes_ticker, permno]

    Returns
    -------
    DataFrame with [permno, date, sue].
    """
    # ── 1. Merge actuals with consensus (statpers < anndats) ────────────
    actuals = ibes_actuals.copy()
    statsum = ibes_statsum.copy()

    actuals["anndats"] = pd.to_datetime(actuals["anndats"])
    actuals["pends"] = pd.to_datetime(actuals["pends"])
    statsum["statpers"] = pd.to_datetime(statsum["statpers"])
    statsum["fpedats"] = pd.to_datetime(statsum["fpedats"])

    merged = actuals.merge(
        statsum,
        left_on=["ibes_ticker", "pends"],
        right_on=["ibes_ticker", "fpedats"],
        how="inner",
    )
    merged = merged.loc[merged["statpers"] < merged["anndats"]].copy()

    # Keep the latest consensus snapshot per announcement
    merged = merged.sort_values("statpers").drop_duplicates(
        subset=["ibes_ticker", "pends", "anndats"], keep="last",
    )

    # ── 2. Compute SUE ─────────────────────────────────────────────────
    stdev = merged["stdev"].replace(0, np.nan)
    merged["sue"] = (merged["eps_actual"] - merged["meanest"]) / stdev
    merged = merged.dropna(subset=["sue"])

    # ── 3. Map ibes_ticker → permno ─────────────────────────────────────
    link = ibes_crsp_link[["ibes_ticker", "permno"]].drop_duplicates()
    merged = merged.merge(link, on="ibes_ticker", how="inner")

    sue_events = merged[["permno", "anndats", "sue"]].copy()
    sue_events["permno"] = sue_events["permno"].astype("int64")
    sue_events = sue_events.sort_values(["permno", "anndats"]).drop_duplicates(
        subset=["permno", "anndats"], keep="last",
    )

    # ── 4. Build monthly panel via merge_asof (single call) ─────────────
    # Create scaffold of (permno, month_end) for permnos with SUE data
    month_ends = pd.date_range(START_DATE, END_DATE, freq="ME")
    permnos = sue_events["permno"].unique()

    scaffold = pd.MultiIndex.from_product(
        [permnos, month_ends], names=["permno", "date"],
    ).to_frame(index=False)

    # lookup_date = month_end - 1 business day (announcement must be before)
    scaffold["lookup_date"] = scaffold["date"] - pd.tseries.offsets.BDay(1)

    # Rename for merge_asof: merge on lookup_date <-> anndats
    # merge_asof requires the 'on' key to be globally sorted
    scaffold = scaffold.sort_values("lookup_date")
    sue_events = sue_events.rename(columns={"anndats": "lookup_date"})
    sue_events = sue_events.sort_values("lookup_date")

    result = pd.merge_asof(
        scaffold, sue_events,
        on="lookup_date", by="permno",
        direction="backward",
        tolerance=pd.Timedelta(days=90),
    )

    result = result[["permno", "date", "sue"]]
    print(f"  [sue] done: {len(result):,} rows, "
          f"{result['sue'].notna().mean():.1%} non-NaN")
    return result
