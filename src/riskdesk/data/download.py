"""Download daily levels for the risk drivers ONCE and save them to data/market_data.csv.

Run:  uv run python -m riskdesk.data.download

The app never calls a market service at run time; it only reads the saved CSV.

Alignment: the calendar is US stock-market trading days (S&P 500). Other series
are matched to those days; if a series has a holiday gap, the last known value is
carried forward for at most 3 days. Bitcoin is empty before Sept 2014.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import yfinance as yf

from riskdesk.data.market import MARKET_DATA_PATH

START = "2005-01-01"
YF_TICKERS = {"equity": "^GSPC", "gold": "GC=F", "usd": "DX-Y.NYB", "btc": "BTC-USD"}
FRED_SERIES = {"oil": "DCOILBRENTEU", "us10y": "DGS10"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
MAX_FILL_DAYS = 3


def fetch_yahoo(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.astype(float)


def fetch_fred(series: str, start: str) -> pd.Series:
    df = pd.read_csv(FRED_URL.format(series=series), na_values=".")
    date_col = df.columns[0]
    s = pd.Series(pd.to_numeric(df[series], errors="coerce").values, index=pd.to_datetime(df[date_col]))
    return s[s.index >= start].astype(float)


def build(start: str = START) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for driver, ticker in YF_TICKERS.items():
        print(f"  {driver:<6} <- yfinance {ticker}")
        series[driver] = fetch_yahoo(ticker, start)
    for driver, code in FRED_SERIES.items():
        print(f"  {driver:<6} <- FRED {code}")
        series[driver] = fetch_fred(code, start)

    calendar = series["equity"].dropna().index
    out = pd.DataFrame(index=calendar)
    for driver, s in series.items():
        s = s.dropna()
        s = s[~s.index.duplicated(keep="last")]
        # reindex onto stock-market days, carrying values over short holiday gaps only
        full = s.reindex(s.index.union(calendar)).ffill(limit=MAX_FILL_DAYS)
        out[driver] = full.reindex(calendar)
    if (out.drop(columns="btc") <= 0).any().any():
        raise RuntimeError("Found a zero or negative price/yield level; check the source data.")
    out.index.name = "date"
    return out[["equity", "oil", "gold", "usd", "btc", "us10y"]]


def main() -> None:
    print("Downloading market data...")
    df = build()
    MARKET_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(MARKET_DATA_PATH)
    meta = {
        "downloaded_on": date.today().isoformat(),
        "start": START,
        "rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "sources": {**{k: f"yfinance {v}" for k, v in YF_TICKERS.items()},
                    **{k: f"FRED {v}" for k, v in FRED_SERIES.items()}},
        "units": "price levels; us10y in percent (4.25 = 4.25%)",
        "missing_values_per_column": {c: int(n) for c, n in df.isna().sum().items()},
    }
    MARKET_DATA_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(df)} rows ({meta['first_date']} to {meta['last_date']}) -> {MARKET_DATA_PATH}")
    print("Missing values per column:", meta["missing_values_per_column"])


if __name__ == "__main__":
    main()
