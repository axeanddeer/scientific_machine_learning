# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** Private research dataset (not publicly available). Contains proprietary microstructure profiler measurements from GEOMAR field experiments. Contact Florian Schütte (fschuette@geomar.de) for access.
- **Dataset Owner/Contact:** Prof. Dr. Florian Schütte, GEOMAR Helmholtz Centre for Ocean Research Kiel, Physical Oceanography group (fschuette@geomar.de, Tel: +49 431 600 4495).

### Dataset Characteristics
- **Number of Observations:** 1,165,844 processed datapoints from 4021 full-depth ocean profiles (29 field experiments, 0.7m vertical resolution).
- **Number of Features:** 9 input features derived from hydrographic profiles and bathymetry.

### Target Variable/Label
- **Label Name:** EPS
- **Label Type:** Regression
- **Label Description:** Turbulent kinetic energy dissipation rate per unit mass (ε in W/kg). Prediction task: estimate ocean turbulence intensity from CTD/hydrographic parameters for mixing parameterization.
- **Label Values:** Positive real numbers, typically 10⁻¹⁰ to 10⁻⁴ W/kg (log-transformed for modeling).
- **Label Distribution:** Log-skew-normal distribution (universal in global ocean turbulence), skewed toward low values in stratified regions.

### Feature Description
- **Feature 1 (N2):** Squared buoyancy frequency (N² = -g/ρ·dρ/dz), float, measures ocean stratification strength (s⁻²).
- **Feature 2 (LN2):** Natural logarithm of N2, float, added for numerical stability in log-space neural network training.
- **Feature 3 (S):** Salinity from CTD measurements, float (PSU), 30-40 PSU range.
- **Feature 4 (T):** Potential temperature from CTD, float (°C), -2 to 30°C range.
- **Feature 5 (Z):** Normalized depth (z ∈ [0,1]), float (m), from surface to seafloor.
- **Feature 6 (dSdz):** Vertical salinity gradient, float (PSU/m), computed from profiles.
- **Feature 7 (dTdz):** Vertical temperature gradient, float (°C/m), computed from profiles.
- **Feature 8 (hab):** Habitat/bottom depth from GEBCO bathymetry grids, float (m), seafloor depth proxy.
- **Feature 9 (lat):** Latitude, float (degrees, -90 to 90), captures regional variability in turbulence.


## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
