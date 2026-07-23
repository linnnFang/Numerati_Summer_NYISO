# Methodology

The intended count outcome is the number of genuinely new outage events in a
defined zone and time period. It can be constructed only if verified event-level
data provide sufficient identity and start-time fields. A non-negative integer
event count may later support Poisson and Negative Binomial comparisons.

P-15 generation outage MW may be used as a feature but never as the count
outcome. Model validation will use chronological rather than random splits. All
timestamps will be converted to `America/New_York`, with repeated and nonexistent
DST clock times handled explicitly rather than guessed.

## Development-panel construction

The daily panel uses zone-day load summaries and a single Albany weather sample.
Load forecasts retain all observed vintages. For feature construction, the most
recent forecast vintage strictly earlier than the valid local date is selected;
this avoids assuming an unobserved intraday P-7 publication time. The first day
of the sample therefore has no prior-vintage forecast.

The hourly price panel averages 5-minute RT LBMP within each local hour and joins
it to hourly DA LBMP at the same observed zone. `DART = RT_LBMP - DA_LBMP` is a
descriptive price-stress outcome, not an outage cause.

Because no valid `new_outage_count` exists, overdispersion statistics and the
Poisson/NB comparison are not estimated. The pipeline writes explicit blocker
tables instead of replacing the target with P-15 MW.
