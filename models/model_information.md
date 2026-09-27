# Model Information

LUMAQ compares Gradient Boosting and Random Forest regression for remaining-runtime prediction. The pipeline scales numeric variables, encodes the load category and selects the candidate with the lowest test mean absolute error.

An Isolation Forest identifies unusual combinations of voltage, current, power demand and solar input. The recommendation function converts model outputs into short energy guidance messages.

All reported results in the DSN notebook originate from simulated proof-of-concept data. Field validation requires timestamped readings from the physical LUMAQ gateway.

