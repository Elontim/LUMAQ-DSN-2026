# Data Dictionary

The DSN 2026 notebook generates a reproducible simulated dataset for a 48 V, 100 Ah solar battery system. The dataset supports software validation while hardware logs remain unavailable.

| Field | Unit | Description |
|---|---:|---|
| timestamp | ISO datetime | Five-minute observation time |
| battery_voltage_v | V | Estimated DC battery voltage |
| battery_current_a | A | Estimated battery discharge current |
| state_of_charge_pct | % | Available battery charge |
| load_power_w | W | Total connected electrical load |
| solar_input_w | W | Available solar charging input |
| ambient_temperature_c | °C | Ambient temperature around the system |
| hour | hour | Decimal hour extracted from the timestamp |
| daylight | binary | One during the simulated daylight interval |
| load_category | category | Essential, productive, comfort or heavy load |
| runtime_minutes | minutes | Calculated remaining runtime target |
| known_anomaly | binary | Injected abnormal load label for validation |
| data_source | text | Declares the records as simulated |

