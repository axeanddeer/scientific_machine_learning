# Deep Ocean Turbulence Prediction

## Repository Link

[https://github.com/axeanddeer/scientific_machine_learning/tree/main](https://github.com/axeanddeer/scientific_machine_learning/tree/main)

## Description

Predicts turbulent kinetic energy dissipation rate (ε) and diapycnal diffusivity (K) from standard CTD hydrographic profiles using deep learning. Bridges gap between plentiful CTD data and costly microstructure measurements to enable global ocean mixing parameterization for climate models.

### Task Type

[Regression]

### Results Summary

#### Best Model Performance
- **Best Model:** ResMLP Ensemble (3-layer residual MLP with skip connections, SGDR)
- **Evaluation Metric:** RMSE/MAE/R² on log-transformed targets (LK=log(K), Leps=log(ε))
- **Final Performance:** 
  - **LK**: RMSE=0.572, MAE=0.396, R²=0.799
  - **Leps**: RMSE=0.572, MAE=0.397, R²=0.653
  - **Per-profile avg LK RMSE**: 0.590 ± 0.216

- **Improvement Over Baseline:** Basically applied baseline model
- **Best Alternative Model:** CART bootstrap ensemble (competitive baseline)

#### Key Insights
- **Most Important Features:** LN2 (-0.6 corr), dSdz (+0.5), dTdz (+0.4), Z (-0.3), hab (+0.2)
- **Model Strengths:** High R²=0.80 on log(K); physics-consistent (shear²/N² physics); robust 10-fold CV across 29 experiments
- **Model Limitations:** Geographic bias (40-60°N dominant); rare high-ε events underrepresented; coastal/deep extrapolation uncertain
- **Business Impact:** Operational turbulence prediction from existing CTD/glider data → direct upgrade for ocean circulation models → improved climate heat/carbon transport forecasts


## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
