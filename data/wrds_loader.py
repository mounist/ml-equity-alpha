"""
WRDS data loader with parquet caching.
All SQL queries are centralised here. Each function checks for a cached
parquet file before hitting WRDS.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import wrds

from config import DATA_DIR, END_DATE, START_DATE, WRDS_USERNAME

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _get_connection() -> wrds.Connection:
    """Open a new WRDS connection using the configured username."""
    return wrds.Connection(wrds_username=WRDS_USERNAME)


# ---------------------------------------------------------------------------
# Generic cache-or-query
# ---------------------------------------------------------------------------


def _cache_or_query(name: str, query: str, db: wrds.Connection) -> pd.DataFrame:
    """Return cached parquet if it exists, otherwise execute *query* and cache.

    Parameters
    ----------
    name : str
        Stem of the parquet file (without extension).
    query : str
        SQL query to run against WRDS.
    db : wrds.Connection
        Active WRDS connection.

    Returns
    -------
    pd.DataFrame
    """
    path = DATA_DIR / f"{name}.parquet"
    if path.exists():
        t0 = time.time()
        df = pd.read_parquet(path)
        print(f"[cache]  {name}: {len(df):,} rows loaded in {time.time()-t0:.1f}s")
        return df

    t0 = time.time()
    df = db.raw_sql(query)
    df.to_parquet(path, index=False)
    print(f"[wrds]   {name}: {len(df):,} rows fetched in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def load_crsp_msf(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load CRSP monthly stock file with share-code and exchange filters.

    Computes *dollar_volume* and *market_cap* after loading.
    """
    query = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.vol,
               b.siccd, b.exchcd, b.shrcd
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
          ON a.permno = b.permno
          AND a.date >= b.namedt AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("crsp_msf", query, db)
    finally:
        if own_conn:
            db.close()

    df["date"] = pd.to_datetime(df["date"])
    df["dollar_volume"] = df["prc"].abs() * df["vol"] * 100
    df["market_cap"] = df["prc"].abs() * df["shrout"] * 1000
    return df


def load_crsp_msi(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load CRSP market index returns (value- and equal-weighted)."""
    query = f"""
        SELECT date, vwretd, ewretd
        FROM crsp.msi
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("crsp_msi", query, db)
    finally:
        if own_conn:
            db.close()

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_compustat(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load Compustat quarterly fundamentals (industrial, domestic, standard)."""
    query = f"""
        SELECT gvkey, datadate, fyear,
               at, ceq, revt, cogs, ni, oancf, dltt, sale, xsga, capx, dp, act, lct
        FROM comp.funda
        WHERE datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND indfmt = 'INDL' AND datafmt = 'STD'
          AND popsrc = 'D' AND consol = 'C'
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("compustat", query, db)
    finally:
        if own_conn:
            db.close()

    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def load_ccm_link(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load CRSP-Compustat merged link table (LC/LU, primary P/C)."""
    query = """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
        FROM crsp.ccmxpf_linktable
        WHERE usedflag = 1
          AND linktype IN ('LC', 'LU')
          AND linkprim IN ('P', 'C')
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("ccm_link", query, db)
    finally:
        if own_conn:
            db.close()

    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    return df


def load_ibes_statsum(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load IBES consensus summary statistics (EPS, 1-quarter-ahead)."""
    query = f"""
        SELECT ticker AS ibes_ticker, statpers, fpedats, meanest, stdev, numest
        FROM ibes.statsum_epsus
        WHERE statpers BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND fpi = '1'
          AND measure = 'EPS'
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("ibes_statsum", query, db)
    finally:
        if own_conn:
            db.close()

    for col in ("statpers", "fpedats"):
        df[col] = pd.to_datetime(df[col])
    return df


def load_ibes_actuals(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load IBES actual EPS (quarterly announcements)."""
    query = f"""
        SELECT ticker AS ibes_ticker, pends, anndats, value AS eps_actual
        FROM ibes.actu_epsus
        WHERE anndats BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND pdicity = 'QTR'
          AND measure = 'EPS'
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("ibes_actuals", query, db)
    finally:
        if own_conn:
            db.close()

    for col in ("pends", "anndats"):
        df[col] = pd.to_datetime(df[col])
    return df


def load_ibes_crsp_link(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load IBES-to-CRSP ticker-permno mapping.

    Tries ``ibes.iclink`` first; if unavailable, falls back to joining
    ``ibes.id`` with ``crsp.msenames`` on CUSIP.
    """
    query = """
        SELECT ticker AS ibes_ticker, permno
        FROM ibes.iclink
        WHERE score IN (0, 1, 2)
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        try:
            df = _cache_or_query("ibes_crsp_link", query, db)
        except Exception:
            # Fallback: join ibes.id with crsp.msenames on CUSIP
            fallback_query = """
                SELECT a.ticker AS ibes_ticker, b.permno
                FROM ibes.id AS a
                INNER JOIN crsp.msenames AS b
                  ON a.cusip = b.ncusip
                GROUP BY a.ticker, b.permno
            """
            df = _cache_or_query("ibes_crsp_link", fallback_query, db)
    finally:
        if own_conn:
            db.close()

    return df


def load_ff_monthly(db: Optional[wrds.Connection] = None) -> pd.DataFrame:
    """Load Fama-French monthly factors (Mkt-RF, SMB, HML, UMD, RF).

    Factor values are divided by 100 (WRDS stores as percentages).
    """
    query = f"""
        SELECT date, mktrf, smb, hml, umd, rf
        FROM ff.factors_monthly
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    own_conn = db is None
    if own_conn:
        db = _get_connection()
    try:
        df = _cache_or_query("ff_monthly", query, db)
    finally:
        if own_conn:
            db.close()

    df["date"] = pd.to_datetime(df["date"])
    # WRDS stores factors as percentages (e.g. 1.5 means 1.5%).
    # If the cached parquet was saved after a prior run that already
    # divided by 100, the values will be in decimal form.  Guard
    # against double-division by checking the scale: monthly |mktrf|
    # in percentage space rarely exceeds 40, so a median > 0.5
    # indicates percentages while < 0.5 indicates decimals.
    factor_cols = ("mktrf", "smb", "hml", "umd", "rf")
    if df["mktrf"].abs().median() > 0.5:
        # Values are in percentage form → convert to decimal
        for col in factor_cols:
            df[col] = df[col] / 100.0
    return df


# ---------------------------------------------------------------------------
# Convenience: load everything
# ---------------------------------------------------------------------------


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all datasets using a single WRDS connection and return as a dict.

    Keys match the function names without the ``load_`` prefix.
    """
    db = _get_connection()
    try:
        data = {
            "crsp_msf": load_crsp_msf(db),
            "crsp_msi": load_crsp_msi(db),
            "compustat": load_compustat(db),
            "ccm_link": load_ccm_link(db),
            "ibes_statsum": load_ibes_statsum(db),
            "ibes_actuals": load_ibes_actuals(db),
            "ibes_crsp_link": load_ibes_crsp_link(db),
            "ff_monthly": load_ff_monthly(db),
        }
    finally:
        db.close()
    return data
