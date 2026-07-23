# Data Inventory

This inventory records the development samples retrieved on 2026-07-17. It
does not claim that later files have the same schema. Raw timestamps have not
yet been localized or cleaned; that is Step 2 work.

| Dataset | Official report/API | Observed raw fields | Observed coverage | Grain and spatial notes |
|---|---|---|---|---|
| Actual load | NYISO P-58B | `Time Stamp`, `Time Zone`, `Name`, `PTID`, `Load` | 2024-01-01 through 2024-01-31 | 5-minute records; 11 observed names |
| Load forecast | NYISO P-7 | `Time Stamp` plus 11 zone columns and `NYISO` | 2024-01-01 through 2024-02-05 | Hourly, wide zonal format; archive includes forecast horizon beyond January |
| Day-ahead LBMP | NYISO P-2A | `Time Stamp`, `Name`, `PTID`, LBMP, loss, congestion | 2024-01-01 through 2024-01-31 | Hourly; 15 observed location names |
| Real-time LBMP | NYISO P-24A | `Time Stamp`, `Name`, `PTID`, LBMP, loss, congestion | 2024-01-01 through 2024-02-01 00:00 | 5-minute; 15 observed location names |
| Generation outage forecast | NYISO P-15 | Source: `Date`, `Forecasted Generation Outage (MW)`; saved: `Date`, `generation_outage_mw`, `retrieval_time` | Current snapshot forecasts 2026-07-17 through 2026-08-16 | Daily aggregate MW; no zone, facility, or event identity |
| RT scheduled transmission outages | NYISO P-54A | `Timestamp`, `PTID`, `Equipment Name`, `Scheduled Out Date/Time`, `Scheduled In Date/Time` | Snapshot timestamps in January 2024 | 5-minute repeated system snapshots; no event ID |
| RT actual transmission outages | NYISO P-54B | `Timestamp`, `PTID`, `Equipment Name`, `Outage Date/Time` | Snapshot timestamps in January 2024 | Repeated system snapshots; no event ID or end time |
| DA scheduled transmission outages | NYISO P-54C | `Timestamp`, `PTID`, `Equipment Name`, `Scheduled Out Date/Time`, `Scheduled In Date/Time` | Daily January 2024 snapshots | Scheduled equipment records; no event ID |
| Outage schedules | NYISO P-14B | `PTID`, `Outage ID`, equipment fields, out/in date and time, call/status/message fields | Latest-only snapshot contains old, current, and future schedules | Has event-like ID and start/end components, but no historical archive or explicit forced/scheduled field |
| Weather development sample | Open-Meteo Historical Weather API | Time, temperature, precipitation, 10 m wind speed, relative humidity | 2024-01-01 through 2024-01-07 | Hourly at one Albany coordinate only |

## Standardized interim tables

- `actual_load`: timezone-aware 5-minute interval, zone, actual MW.
- `load_forecast`: timezone-aware hourly valid interval, zone, forecast MW, and
  forecast vintage date extracted from the preserved NYISO archive member name.
- `da_lbmp` and `rt_lbmp`: interval, market, zone, LBMP, derived energy component,
  observed congestion/loss components, and source.
- `weather_albany`: hourly interval and the four observed weather measures.
- `p15_generation_outage_forecast`: daily interval, retrieval time, and forecast
  MW. Interval and retrieval timestamps are intentionally separate.
- `outage_schedules_latest`: P-14B event-like records explicitly marked as an
  incomplete latest-only source; affected MW and zone remain missing.
- `p54a/p54b/p54c_*_snapshots`: timestamped transmission-equipment snapshots;
  these are not declared unique outage events.

All naive clock times are localized to `America/New_York` with ambiguous and
nonexistent DST times configured to raise. Exact validation results are saved in
`outputs/tables/clean_data_summary.csv`.

## Outage-count feasibility

The primary outage-count target is currently **BLOCKED**:

- P-15 is aggregate forecasted generation outage MW, not event data or count.
- P-54A and P-54C contain scheduled start/end times but no event ID and repeat
  records across snapshots.
- P-54B contains actual outage time but no event ID or end time.
- P-14B contains `Outage ID` and out/in components, but its public index exposes
  only the current snapshot. It does not supply historical vintages needed to
  reconstruct what was newly reported or changed through time.
- None of the inspected fields provides a verified common forced/scheduled
  classification suitable for the intended generation-outage count target.

The exact retrieval URLs, rows, columns, missingness, duplicate counts, and
saved paths are recorded in `outputs/tables/data_download_summary.csv`.
