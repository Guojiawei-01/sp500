# Macro Data

These files are downloaded from FRED by `scripts/prepare_data.py`.

| File | FRED Series | Description |
| --- | --- | --- |
| `VIXCLS.csv` | `VIXCLS` | CBOE Volatility Index |
| `DGS10.csv` | `DGS10` | 10-year Treasury constant maturity rate |
| `DGS2.csv` | `DGS2` | 2-year Treasury constant maturity rate |
| `FEDFUNDS.csv` | `FEDFUNDS` | Federal funds effective rate |
| `CPIAUCSL.csv` | `CPIAUCSL` | Consumer Price Index for All Urban Consumers |
| `UNRATE.csv` | `UNRATE` | Unemployment rate |
| `USREC.csv` | `USREC` | NBER recession indicator |
| `fred_macro_combined.csv` | multiple | Combined macro table used before as-of joining to headline dates |

The main modeling file is `data/processed/daily_with_macro.csv`.
