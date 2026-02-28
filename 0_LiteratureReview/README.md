**Source 1: Physics-informed deep-learning parameterization of ocean vertical mixing improves climate simulations (Zhu et al., 2022):**

[[**https://academic.oup.com/nsr/article/9/8/nwac044/6544687?utm_source=chatgpt.com**](https://academic.oup.com/nsr/article/9/8/nwac044/6544687?utm_source=chatgpt.com)](https://academic.oup.com/nsr/article/9/8/nwac044/6544687?utm_source=chatgpt.com)

**Objective**: physics-informed deep neural network parameterization of vertical turbulent mixing for use in climate and ocean general circulation models, improving upon traditional Richardson-number- or KPP-type schemes. The study aims to reduce systematic biases in upper-ocean temperature structure, particularly in the tropical Pacific.

**Methods**:

> **Model/data:** feedforward artificial neural network (ANN) used as a subgrid parameterization module using long-term observational records from the tropical Pacific, including hydrographic profiles (temperature, salinity), vertical shear, stratification (N²), and turbulence-related mixing diagnostics derived from field observations.
>
> **Prediction:** vertical turbulent mixing coefficients (eddy diffusivity \(K_{T}\) and viscosity \(K_{v}\)), rather than ε directly
>
> **Physical constraints:** stability-consistent dependencies on stratification and shear, ensuring physically admissible outputs
>
> **Evaluation:** The method was tested both offline - by comparing it with observationally derived mixing rates - and online, through implementation in a coupled ocean--climate model. Its performance was evaluated based on how well it improved the simulated temperature structure and reduced biases relative to conventional parameterizations such as KPP-type (K-Profile Parameterization) schemes.

**Outcomes:** The approach led to a more accurate representation of the upper-ocean thermal structure and reduced biases in the tropical Pacific.

**Relation to the Project:** Provides a template for embedding ML-derived turbulence relationships into large-scale models, especially if work extends from ε estimation toward diffusivity parameterization.

**Source 2: Deep ocean learning of small-scale turbulence (Mashayek et al., 2022)**

<https://www.researchgate.net/publication/362257520_Deep_ocean_learning_of_small_scale_turbulence>

**Objective**: use supervised machine learning to predict turbulent kinetic energy dissipation rate (ε) and diffusivity in the ocean interior from widely available hydrographic and environmental variables, reducing reliance on sparse microstructure measurements.

**Methods**:

> **Model/data:** multiple ML algorithms - including neural networks and tree-based models - are trained to map predictors (e.g., temperature, salinity, stratification, bathymetric proximity, large-scale flow descriptors) to observed ε.
>
> **Prediction:** turbulent kinetic energy dissipation rate (ε) and inferred mixing diffusivity.
>
> **Physical constraints:** Feature selection is guided by physical reasoning---for example, by considering variables such as the buoyancy frequency (N²) and the influence of topography---but the models themselves remain purely data-driven.
>
> **Evaluation:** Model performance is assessed using train/test splits with cross-validation, comparison of skill against traditional parameterizations (such as internal-wave scaling laws), and statistical measures including R^2^ and error distributions across different regimes.

**Outcomes**: ML models outperform classical empirical scaling laws in predicting ε across diverse regimes. The approach demonstrates that data-driven models can capture nonlinear dependencies not represented in traditional parameterizations, especially in internal-wave-driven and topographically influenced mixing.

**Relation to the Project**: Direct methodological analogue if aim is supervised learning of ε from microstructure datasets.

**Source 3: Probabilistic neural networks for predicting energy dissipation rates in stratified turbulent flows (Lewin et al., 2021)**

<https://arxiv.org/abs/2112.01113>

**Objective**: To develop a probabilistic neural network capable of predicting turbulent energy dissipation rates (ε) in stratified turbulence, with quantified uncertainty, using high-resolution simulation data representative of geophysical flows.

**Methods**:

> **Model/data:** Convolutional neural network (CNN), extended to a probabilistic neural network (PNN) that outputs distributions rather than deterministic values. High-resolution direct numerical simulation (DNS) data of stratified turbulence. Inputs consist of vertical columns of velocity and density gradients representative of microstructure-like signals.
>
> **Prediction:** Turbulent kinetic energy dissipation rate (ε), represented as a probability distribution.
>
> **Physical constraints:** No explicit PINN framework. However, inputs are physically meaningful gradients, and statistical structure of stratified turbulence is implicitly learned. Physical intermittency characteristics are preserved via probabilistic modeling.
>
> **Evaluation:** Comparison between predicted and true ε distributions, assessment of intermittency and heavy-tailed behavior reproduction, and comparison to classical scaling approaches.

**Outcomes**: The probabilistic CNN captures nonlinear gradient--dissipation relationships and reproduces intermittency more accurately than empirical scaling laws.

**Relation to the Project**: Particularly relevant if your work addresses uncertainty quantification and the strongly intermittent nature of ε in stratified microstructure data.
