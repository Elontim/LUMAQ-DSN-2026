# LUMAQ Hardware Profile

The LUMAQ pitch presented a working smart energy gateway developed at the Federal University of Technology, Minna. The prototype demonstrated real-time energy metering, bidirectional measurement, offline processing and a live dashboard.

The DSN 2026 repository focuses on the Python machine learning layer. The supplied pitch deck did not contain the firmware source, sensor model numbers or exported hardware readings. For that reason, the repository does not claim that the included dataset came from the physical prototype.

## Demonstration assumptions

| Parameter | Value |
|---|---:|
| Nominal battery voltage | 48 V |
| Demonstration capacity | 100 Ah |
| Usable depth of discharge | 80% |
| Inverter efficiency | 90% |
| Sampling interval | 5 minutes |

These assumptions create a reproducible software demonstration. Hardware logs can replace the simulated records through the same schema documented in `data/data_dictionary.md`.

