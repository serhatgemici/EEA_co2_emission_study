from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from ziko_st_toc import table_of_contents

with st.sidebar:
    table_of_contents()

st.set_page_config(page_title="Model Selection and Fine Tuning", layout="wide")


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
data_path1 = (
    BASE_DIR / "data" / "streamlit_samples_final_remodeling" / "y_test_vs_pred_2025.csv"
)

st.image(
    BASE_DIR / "src" / "streamlit" / "fine-tuning.jpg",
    caption="Model Selection and Fine Tuning",
    width="stretch",
)

data_path_mono = (
    BASE_DIR
    / "data"
    / "streamlit_samples_final_remodeling/segmented"
    / "y_test_vs_pred_2025_mono.csv"
)

data_path_phev = (
    BASE_DIR
    / "data"
    / "streamlit_samples_final_remodeling/segmented"
    / "y_test_vs_pred_2025_phev.csv"
)

data_path_hybrid = (
    BASE_DIR
    / "data"
    / "streamlit_samples_final_remodeling/segmented"
    / "y_test_vs_pred_2025_hybrid.csv"
)

df = pd.read_csv(data_path1)
df_mono = pd.read_csv(data_path_mono)
df_phev = pd.read_csv(data_path_phev)
df_hybrid = pd.read_csv(data_path_hybrid)

st.title("Model Selection and Fine Tuning")

content_1 = """## Executive Summary

This report presents the final leakage-aware modeling workflow for predicting official WLTP CO₂ emissions (g/km) for **French** vehicles. 

The **2024 data** were used for **development, group-aware validation, feature selection and hyperparameter tuning**; the **2025 data** were held out as a temporal test set. 

The modeling report consists of **2 parts**, where **part 1** represents execution performed on the **complete** dataset, **part 2** represents the execution that was performed on segmented datasets based on fuel modes; namely **Mono**, **PHEV (Plug-in hybrid electric vehicles)** and **Hybrid**.

### Executive conclusion part 1

An initial modeling test was performed with the initial random split dataset prepared during the data exploration and feature engineering phase. 

Given the results were far away from optimal, the approach was dropped and study continued using a group based splitting and tested feature sets. 

The first trial modeling will not be included in this report, however the Jupyter notebook that contains this study (named `03_modeling.ipynb`) can be found in our repository.

The project progressed from an overly optimistic first benchmark to a leakage-aware final workflow. 

The strongest conventional model was a tuned ExtraTreesRegressor. 

On the final workflow, it achieved validation RMSE 6.63 g/km and 2025 test RMSE 10.58 g/km, MAE 5.03 g/km and R² 0.9625. 

A neural network was also successful: its best reported test result was RMSE 8.67 g/km and MAE 6.67 g/km. 

The Lightning MLP had a lower reported RMSE but a higher MAE than the final ExtraTrees model; because the evaluation workflows may not be identical, the results should not be treated as a definitive head-to-head comparison. 

ExtraTrees remains the primary recommendation because it is the fully validated and interpreted model in the final temporal workflow.

### Executive conclusion part 2

The workflow progressed from duplicate consolidation and leakage investigation to a tuned ExtraTreesRegressor. 

ExtraTrees was selected as the primary conventional model. 

The extension notebook then trained separate models for **Mono-fuel (M), PHEV (P) and Hybrid (H)** vehicles, allowing errors and feature importance to be analysed within powertrain regimes.

The segmented ExtraTrees evaluation on 2025 produced **MAE 5.0343 g/km, RMSE 10.5761 g/km and R² 0.9625.** 

**Mono** and **Hybrid** models performed substantially better than the **PHEV** model. 

The principal weakness was systematic under-prediction for **unseen petrol/electric PHEV** configurations.
"""
st.markdown(content_1, unsafe_allow_html=False)


content_2 = """# MODELING PART I: 
## Analysis On Complete Dataset

## 1. Data Preparation and Experimental Design

The target is `Ewltp (g/km)`, the official WLTP CO₂-emission value. The 2024 French dataset was used for model development, and the 2025 French dataset was held out as a temporal final test. The final notebook reports 15,957 aggregated 2024 records and 14,735 aggregated 2025 records; the final group split contained 12,801 training rows and 3,156 validation rows.

A central improvement was replacing a simple random-style benchmark with a group-aware split. Groups were defined from core technical characteristics: mass, engine capacity, engine power, fuel type and fuel mode. The final split had zero exact core-configuration overlap between training and validation. The 2025 set was not used for model selection or tuning.

Preprocessing was fitted on the training partition only: top-make grouping, group-aware target encoding, frequency maps, IT-code extraction, one-hot encoding and scaling. The final feature design distinguished between a strict design-stage feature set and an extended specification set containing electric range and eco-innovation indicators.

```text
2024 FR raw
│
├── Common cleaning / feature engineering
│
└── 2024 train-validation split
    │
    ├── Train-only fit
    │   ├── Top brands
    │   ├── Frequency maps
    │   ├── Target encoder
    │   ├── Scaler
    │   └── Model
    │
    └── Validation transform only


2025 FR raw
│
└── Common cleaning / feature engineering
    │
    └── Transform only with the artefacts learnt in Train
        │
        └── Final test
```

---

## 2. Trial-and-error journey

| Stage                         | What was tried                                                                 | Result and lesson                                                                                                                                                                                                                                                                                                                                                                                                          |
|------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Initial automated benchmark  | LazyPredict comparison of many regressors                                       | Tree ensembles dominated linear models. ExtraTrees, Random Forest, HistGradientBoosting and Bagging were the leading families.                                                                                                                                                                                                                                                                                             |
| Resource control             | Excluded Gaussian-process/kernel-style candidates after estimating a roughly 61.44 GB kernel-matrix requirement versus a 32 GB machine | A practical modeling decision prevented an unproductive or crashing experiment.                                                                                                                                                                                                                                                                                                                                              |
| First very high scores       | ExtraTrees/RF appeared near R² 0.999 with RMSE around 1.6–1.9 g/km              | The result was suspiciously strong, so leakage, target proxies and duplicate configurations were investigated. In the final notebook's broad comparison, Random Forest and ExtraTrees achieved validation RMSE of 1.16 and 1.19 g/km before the stricter feature-set and group-aware tuning stages.                                                                                                                         |
| Feature audit                | Examined electric range, fuel mode, fuel type, frequency features and possible target-related columns | Electric range was powerful; identifier-derived frequency features were treated cautiously and removed from final feature logic where appropriate.                                                                                                                                                                                                                                                                         |
| Feature-set comparison       | Strict nine-feature design-stage set versus 17-feature extended specification set | Extended set improved validation RMSE from 9.10 to 7.45 g/km, about 18.1% lower.                                                                                                                                                                                                                                                                                                                                             |
| Tuning                       | RandomizedSearchCV with five-fold GroupKFold, only inside the 2024 training data | ExtraTrees achieved the best inner-CV RMSE and the best held-out validation RMSE among tuned candidates.                                                                                                                                                                                                                                                                                                                     |
| Deep learning                | PyTorch MLP, then an organized PyTorch Lightning version                        | The network learned the signal but remained less accurate than the tree ensemble on reported test metrics.                                                                                                                                                                                                                                                                                                                   |

---

## 3. Ensemble modeling results

The broad comparison showed why nonlinear ensembles were appropriate. The target is influenced by interacting vehicle specifications and categorical powertrain regimes; a linear model cannot represent these interactions as naturally without extensive manual feature construction.

| Model/family                        | Validation RMSE (g/km) | Validation R² | Interpretation                                                                                                                                          |
|-------------------------------------|-------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Random Forest / initial comparison  | 1.16                    | 0.9995        | Best result in the broad automated comparison; later tested under stricter features and group-aware tuning.                                            |
| ExtraTrees / initial comparison     | 1.19                    | 0.9995        | Very strong early candidate in the broad automated comparison.                                                                                           |
| LightGBM / initial comparison       | 1.30                    | 0.9994        | Efficient boosting candidate.                                                                                                                            |
| Bagging / initial comparison        | 1.30                    | 0.9994        | Strong ensemble baseline in the broad comparison.                                                                                                        |
| Tuned ExtraTrees                    | 6.63                    | 0.9838        | Final selected model; best inner-CV RMSE and stronger held-out validation result.                                                                        |
| Tuned Random Forest                 | 7.37                    | 0.9799        | Close benchmark, but weaker than tuned ExtraTrees.                                                                                                       |
| Extended ExtraTrees feature-set baseline | 7.45               | 0.9795        | Outperformed the strict design-stage feature set before tuning.                                                                                          |

The selected ExtraTrees configuration was: 678 trees, no maximum-depth limit, `max_features` approximately 0.8268, minimum samples per leaf 3, minimum samples split 2, `random_state` 42 and all available CPU cores. The search optimized negative RMSE using five-fold GroupKFold on the training partition. The held-out validation set and the 2025 test set were not used to choose the parameters.

---

## 4. Feature engineering and interpretation

The extended specification model used **18 features: vehicle mass, engine capacity, engine power, fuel-type indicators, fuel-mode indicators, electric range and IT-code indicators**. The **strict model** used 10 design-stage variables and **excluded electric range and IT features.**

Electric range was the most important interpretive finding. In the exploratory audit, the 0-km group had mean emissions around **138.6 g/km**, while vehicles with positive electric range averaged around **28.3 g/km**. Within the positive-range group, electric range had a strong inverse relationship with emissions: Pearson correlation about **-0.786** and Spearman correlation about **-0.876**. This explains why electric range can produce a major performance improvement, but it is also a potential proxy: it identifies powertrain classes and their emission regimes.

Feature-set comparison showed the trade-off. The strict 10-feature design-stage model achieved validation **RMSE 9.10 g/km**, while adding electric range and IT indicators reduced validation RMSE to **7.45 g/km**, an improvement of approximately **18.1%**. Predictive importance is not automatically causal importance.

Interpretability tools used in notebooks included three features of importance, permutation-importance experiments and controlled feature ablation. The main interpretation is global and model-based, not causal claim. For final deployment, SHAP or grouped permutation importance should be added with confidence intervals and evaluated separately for combustion, hybrid and electric subgroups.

### Feature-set definitions

- **Strict design-stage:** Features expected to be available before the additional specification information is known.
- **Extended specification:** The strict feature set plus electric-range and eco-innovation technology indicators.

The extended specification feature set provides additional predictive information compared with the strict design-stage feature set.

The extended model is therefore used as the primary feature set when electric range and eco-innovation technology indicators are available at prediction time.

The strict design-stage model remains relevant for earlier predictions, where those additional specification variables are not yet available.

| Model use case | Feature set | Prediction context |
|---|---|---|
| Early-stage prediction | Strict design-stage | Only information available at the earlier design stage is used. |
| Specification-based prediction | Extended specification | Electric range and eco-innovation indicators are available. |

The comparison is performed on the same validation split so that the effect of the additional features can be assessed fairly.

| Colon            | Decision               | Reasoning                                                                                             |
| ---------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Fuel consumption | Drop                  | WLTP CO2 is nearly deterministic and comes from the same test context                                 |
| Erwltp (g/km)    | Drop                  | Post-certification eco-innovation CO2 credit.                                                         |
| Mt               | Drop                  | WLTP test mass; determined in the context of testing and type approval                                |
| z (Wh/km)        | Drop                  | WLTP electricity consumption; very close to the test results                                          |
| Make_Encoded     | Drop from the first base model | Target encoding; Since it carries the brand's CO2 average, it is an unnecessary target-derived signal in the physical model |
| Model_Frequency  | Drop                  | A proxy for the model's popularity; not a physical CO2 characteristic                                 |
| VFN_Frequency    | Drop                  | ID proxy derived from the vehicle family identifier                                                   |
| Tan_Frequency    | Drop                  | Type-approval identifier proxy                                                                        |
| T_Frequency      | Drop                  | Vehicle type identifier proxy                                                                         |
| Va_Frequency     | Drop                  | Variant identifier proxy                                                                              |
| Ve_Frequency     | Drop                  | Version identifier proxy                                                                              |
| Ct_Frequency     | Drop                  | Type-approval category frequency; low practical contribution to physical prediction                   |
| RLFI_Frequency   | Drop                  | Road-load-family ID proxy                                                                             |


### Model A: Strict design-stage model
It uses the key characteristics of a vehicle that are known before its official WLTP test results are available:

```python
STRICT_DESIGN_FEATURES = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
    "Ft_petrol/electric",
    "Ft_diesel/electric",
    "Fm_H",
    "Fm_M",
    "Fm_P",
]
```
This model:
- has the lowest risk of leakage.
- measures the ability to generalize realistically to new technical configurations.

### Model B — Extended specification model
It includes additional information that we assume is included in the design/product specifications:

```python
EXTENDED_SPEC_FEATURES = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
    "Ft_petrol/electric",
    "Ft_diesel/electric",
    "Fm_H",
    "Fm_M",
    "Fm_P",
    "Electric range (km)",
    "IT_28",
    "IT_29",
    "IT_32",
    "IT_33",
    "IT_37",
    "IT_38",
    "IT_none",
]
```
***Electric range and eco-innovation technology indicators are assumed to be available as design specifications before the final WLTP CO2 value is predicted.

### Evaluating Candidate Regressors

Evaluate the available regression models on the same encoded and scaled training/validation data.

The purpose of this stage is to identify promising model families before hyperparameter tuning. The comparison uses the same validation split and the same evaluation metrics for every candidate.

### Evaluation metrics

- **MAE:** Mean Absolute Error; lower values are better.
- **RMSE:** Root Mean Squared Error; lower values are better and larger errors receive more weight.
- **R²:** Coefficient of determination; higher values are better.
- **Adjusted R²:** R² adjusted for the number of predictors.
- **Training time:** Useful for understanding the cost of each candidate model.
---

## 5. Deep-learning experiments

The deep-learning notebooks implemented a multilayer perceptron using 17 selected features. The architecture used 128 → 64 → 32 hidden units, BatchNorm, GELU activations and 5% dropout. Training used AdamW with learning rate 0.001, weight decay 1e-5, MSE loss, ReduceLROnPlateau and early stopping. The direct PyTorch run stopped around epoch 32 after validation loss ceased improving.

| Deep-learning version   | Reported result                          | Assessment                                                                                                                                                                      |
|-------------------------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PyTorch Lightning MLP   | MAE 8.0732; RMSE 11.9008                 | Lower reported RMSE but higher MAE than final ExtraTrees; workflow comparability should be confirmed before ranking.                                                             |

The network had approximately 13.2 thousand trainable parameters and used Apple MPS acceleration. The best checkpoint was restored before testing. Reported test metrics were MAE 8.0732 g/km and RMSE 11.9008 g/km. The MLP learned the broad relationship but underperformed the ExtraTrees model.

The result is scientifically useful rather than a failure. Neural networks are not guaranteed to dominate medium-dimensional tabular data, especially when randomized trees capture categorical regimes and threshold-like relationships effectively. The Lightning MLP reported a lower RMSE but a higher MAE than final ExtraTrees; a controlled same-split comparison is required before declaring one model family more accurate overall.

---

## 6. Final evaluation and model choice

## Tuning workflow

```text
2024 training data
    ↓
Group-aware inner cross-validation
    ↓
Randomized hyperparameter search
    ↓
Select the best parameters for each model
    ↓
Evaluate each tuned model on the same validation set
```

The 2025 dataset is not involved in model selection or hyperparameter tuning.
```text
2024 group split:
    X_train
        └── Inner GroupKFold + RandomizedSearchCV
              ├── Tune RandomForest 
              └── Tune ExtraTrees
            
    X_val
        └── Compare the tuned models on the same zero-overlap validation set

2025:
    └── For the final test only; does not involve the tuning/model selection process
```

The final model was set to be the ExtraTreesRegressor model after a 20-iteration randomized search cross validation:

| Model        | Best CV RMSE | Validation MAE | Validation RMSE | Validation R² | Best params                                                                                                                                                      |
|--------------|--------------|----------------|-----------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ExtraTrees   | 7.2341       | 4.5150         | 6.6267          | 0.9838        | `{'max_depth': None, 'max_features': 0.82680517164919, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 678}`                                     |
| RandomForest | 7.4048       | 5.0432         | 7.3651          | 0.9799        | `{'max_depth': 25, 'max_features': 0.41235069657748147, 'min_samples_leaf': 2, 'min_samples_split': 9, 'n_estimators': 791}`                                    |

The 2025 test set contains both familiar and new technical configurations. After retraining on the combined 2024 development data, 83.73% of 2025 rows shared an exact core configuration with the 2024 data, while 16.27% were unseen. Consequently, the overall test score should not be interpreted as pure generalization to entirely new vehicle designs; subgroup reporting for seen versus unseen configurations was investigated.

| Subset                  | Rows  | MAE (g/km) | RMSE (g/km) | R²       |
|-------------------------|-------|------------|-------------|----------|
| All 2025 records        | 14,735| 5.0343     | 10.5761     | 0.9625   |
| Seen core configurations| 12,337| 3.8755     | 8.6901      | 0.9733   |
| Unseen core configurations| 2,398| 10.9959   | 17.2856     | 0.9182   |

Permutation importance was calculated on the held-out 2025 temporal test set. For each feature, its values were randomly permuted ten times while keeping all other features unchanged. Importance was measured as the average increase in RMSE after permutation.

| Feature           | RMSE increase (g/km) |
|-------------------|----------------------|
| Electric range (km) | 31.87              |
| Fm_P              | 29.91                |
| m (kg)            | 18.79                |
| ep (KW)           | 12.97                |
| ec (cm3)          | 6.05                 |
| Fm_H              | 2.56                 |
| Fm_M              | 1.82                 |
| Ft_petrol         | 0.90                 |
| Ft_diesel         | 0.71                 |
| IT_29             | 0.71                 |
| IT_37             | 0.70                 |

The model relied most strongly on `Electric range (km)`. Randomly permuting this feature increased test RMSE by 31.87 g/km, indicating that electric driving range contains substantial predictive information for WLTP CO₂ emissions.

The next most important variable was `Fm_P`, which indicates the plug-in hybrid powertrain group and increased RMSE by 29.91 g/km when permuted. It is closely related to electric range, so the variables should be interpreted as a correlated PHEV feature group rather than as completely independent effects.

Vehicle mass, engine power, and engine capacity also made substantial contributions. This indicates that the model uses physically meaningful vehicle specifications in addition to powertrain type and electric driving capability.

`IT_38` had no measurable importance, while `Ft_diesel/electric` and `Ft_petrol/electric` had slightly negative importance, suggesting that they provided little incremental predictive value after the other features were included.
"""
st.markdown(content_2, unsafe_allow_html=False)

content_3 = """
**Scatter Plot: Actual Vs. Predicted CO2 Emissions (g/km) (2025 Temporal Test)**
"""
st.markdown(content_3, unsafe_allow_html=False)

actual = df["actual_Ewltp"].to_numpy()
predicted = df["predicted_Ewltp"].to_numpy()

# Sample for the plot (max 20k)
plot_sample_size = min(20_000, len(actual))

rng = np.random.default_rng(42)
sample_positions = rng.choice(
    len(actual),
    size=plot_sample_size,
    replace=False,
)

sample_actual = actual[sample_positions]
sample_predicted = predicted[sample_positions]

lower_limit = min(sample_actual.min(), sample_predicted.min())
upper_limit = max(sample_actual.max(), sample_predicted.max())

fig = plt.figure(figsize=(8, 6))

plt.scatter(
    sample_actual,
    sample_predicted,
    alpha=0.15,
    s=12,
)

plt.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect prediction",
)

plt.xlim(lower_limit, upper_limit)
plt.ylim(lower_limit, upper_limit)

plt.xlabel("Actual WLTP CO₂ emissions (g/km)")
plt.ylabel("Predicted WLTP CO₂ emissions (g/km)")
plt.title("2025 Temporal Test: Actual vs Predicted CO₂")
plt.legend()

plt.tight_layout()

st.pyplot(fig, width=1100)

content_4 = """
Scatter plot compares the actual and predicted WLTP CO₂ emissions for the 2025 temporal test set. The red dashed line represents perfect predictions, where the predicted value is equal to the actual value.

Most observations are located close to the perfect-prediction line, indicating a strong overall agreement between the predicted and observed emissions. The model performs particularly well across the central portion of the target distribution and follows the general upward trend across increasing CO₂ values.

However, the plot also reveals several structured error patterns. In the low-emission range, approximately 20–100 g/km, a secondary group of predictions is located substantially below the perfect-prediction line. This indicates systematic under-prediction for some low-emission vehicles, potentially due to differences between plug-in hybrid and conventional powertrain configurations.

The model also shows some under-prediction for a small number of high-emission vehicles, particularly in the upper tail of the distribution. These errors may be associated with rare high-performance or heavy vehicle configurations that are less represented in the training data.

Overall, the plot confirms that the model captures the main relationship between vehicle specifications and WLTP CO₂ emissions. Nevertheless, the remaining structured errors show that overall performance metrics should be complemented with seen-versus-unseen configuration analysis and subgroup-level error analysis.

---

## Residual Analysis

Residuals were defined as:

```math
{residual} = {actual CO2} - {predicted CO2}
```

Therefore, positive residuals indicate underprediction, while negative residuals indicate overprediction.

- **Residual definition:** actual - predicted  
- **Residual min:** -46.76 g/km  
- **Residual max:** +141.12 g/km  
- **Residual mean:** +1.13 g/km  
- **Residual median:** -0.13 g/km  

**Largest positive residual (largest underprediction):**  
- Actual: 350.0 g/km  
- Predicted: 208.88 g/km  
- Residual: +141.12 g/km  

**Largest negative residual (largest overprediction):**  
- Actual: 263.0 g/km  
- Predicted: 309.76 g/km  
- Residual: -46.76 g/km  

The residual distribution showed limited overall bias:

- Mean residual: +1.13 g/km  
- Median residual: -0.13 g/km  
- Minimum residual: -46.76 g/km  
- Maximum residual: +141.12 g/km  

The median residual was close to zero, indicating that the model was broadly well calibrated for a typical 2025 record. The slightly positive mean residual was caused by a small number of large underpredictions.

The largest under-prediction occurred for a vehicle with an actual WLTP CO₂ value of 350.0 g/km. The model predicted 208.88 g/km, resulting in a residual of +141.12 g/km. The largest overprediction occurred for a vehicle with an actual CO₂ value of 263.0 g/km and a prediction of 309.76 g/km, corresponding to a residual of -46.76 g/km.

The residual-versus-actual plot showed that errors were generally concentrated around zero in the central emission range. However, residual spread increased for low-emission and high-emission vehicles. This pattern suggests that the model captures the dominant relationships between vehicle specifications and CO₂ emissions, while rare vehicle configurations and unobserved design factors remain difficult to predict.
"""
st.markdown(content_4, unsafe_allow_html=False)

# Plot residuals vs actual CO₂ emissions
plot_sample_size = min(20_000, len(actual))

rng = np.random.default_rng(42)
sample_positions = rng.choice(
    len(actual),
    size=plot_sample_size,
    replace=False,
)


residuals = actual - predicted

sample_actual = actual[sample_positions]
sample_residuals = residuals[sample_positions]

fig = plt.figure(figsize=(8, 6))

plt.scatter(
    sample_actual,
    sample_residuals,
    alpha=0.15,
    s=12,
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Zero residual",
)

plt.xlabel("Actual WLTP CO₂ emissions (g/km)")
plt.ylabel("Residual: Actual - Predicted (g/km)")
plt.title("2025 Temporal Test: Residuals vs Actual CO₂")
plt.legend()

plt.tight_layout()
st.pyplot(fig, width=1100)


content_5 = """
## Error Analysis by Fuel Type and Fuel Mode

| Fuel type     | Fuel mode | Records | Actual mean (g/km) | Predicted mean (g/km) | Mean residual (g/km) | MAE (g/km) | RMSE (g/km) |
|---------------|-----------|---------|---------------------|------------------------|-----------------------|------------|-------------|
| petrol/electric | P        | 2,466   | 36.58               | 27.29                  | +9.29                 | 9.83       | 20.93       |
| diesel/electric | P        | 229     | 22.67               | 13.63                  | +9.04                 | 9.35       | 18.05       |
| diesel          | H        | 1,276   | 147.29              | 150.12                 | -2.83                 | 6.38       | 9.74        |
| petrol          | H        | 3,907   | 130.63              | 131.63                 | -1.00                 | 4.16       | 6.35        |
| diesel          | M        | 3,121   | 157.19              | 157.27                 | -0.08                 | 3.85       | 6.24        |
| petrol          | M        | 3,736   | 147.82              | 147.99                 | -0.17                 | 3.06       | 5.00        |

Although the model captured that this vehicle had a large engine and very high power, it substantially under-predicted its emissions. This suggests that the model can overgeneralize the low-emission signal associated with PHEV vehicles, especially for rare, high-performance PHEV configurations with short electric range.

A similar pattern appeared for a high-performance PHEV with a 66 km electric range: the actual CO₂ value was 140 g/km, while the model predicted 48.47 g/km. This confirms that electric range alone is insufficient to represent the full WLTP CO₂ behavior of high-performance PHEV vehicles.

Several heavy petrol/electric PHEV records also had large underpredictions of approximately 77–85 g/km. These records were labelled as seen core configurations because their mass, engine capacity, engine power, fuel type, and fuel mode were present in the 2024 data. However, a seen core configuration does not guarantee an identical full vehicle specification, because additional characteristics such as battery configuration, drivetrain architecture, vehicle version, aerodynamic properties, and WLTP homologation details were not included in the final model.

The analysis also identified a large underprediction for an unseen high-emission petrol configuration. Its actual CO₂ value was 456 g/km, compared with a prediction of 375.81 g/km. This result indicates that rare, extreme high-emission vehicles are more difficult to predict because comparable observations are less common in the training data.

The repeated records with similar technical specifications should be interpreted as multiple records with related specifications rather than as 20 completely independent vehicle cases. Overall, the largest errors were concentrated in rare high-performance PHEV vehicles and extreme high-emission conventional vehicles, which likely require additional information not available in the leakage-free feature set.

---

## Distribution of Absolute Prediction Errors

Overall regression metrics such as MAE and RMSE summarize model performance with a single value, but they do not show how errors are distributed across individual records.

To better understand the practical accuracy of the final model, the distribution of absolute prediction errors was analyzed on the held-out 2025 temporal test set.
"""
st.markdown(content_5, unsafe_allow_html=False)
# Calculate absolute prediction errors for the 2025 temporal test set
absolute_errors = np.abs(actual - predicted)

error_percentiles = pd.DataFrame(
    {
        "Percentile": [
            "50th percentile (Median)",
            "75th percentile",
            "90th percentile",
            "95th percentile",
            "99th percentile",
            "Maximum",
        ],
        "Absolute error (g/km)": [
            np.percentile(absolute_errors, 50),
            np.percentile(absolute_errors, 75),
            np.percentile(absolute_errors, 90),
            np.percentile(absolute_errors, 95),
            np.percentile(absolute_errors, 99),
            absolute_errors.max(),
        ],
    }
)

plot_upper_limit = np.percentile(
    absolute_errors,
    99.5,
)

fig = plt.figure(figsize=(8, 6))

plt.hist(
    absolute_errors,
    bins=60,
    range=(0, plot_upper_limit),
    edgecolor="white",
    alpha=0.85,
)

plt.axvline(
    np.median(absolute_errors),
    color="green",
    linestyle="--",
    linewidth=2,
    label=(f"Median: " f"{np.median(absolute_errors):.2f} g/km"),
)

plt.axvline(
    np.mean(absolute_errors),
    color="red",
    linestyle="--",
    linewidth=2,
    label=(f"MAE: " f"{np.mean(absolute_errors):.2f} g/km"),
)

plt.xlabel("Absolute prediction error (g/km)")
plt.ylabel("Number of 2025 records")
plt.title("Distribution of Absolute Errors: 2025 Temporal Test")
plt.legend()

plt.tight_layout()
st.pyplot(fig, width=1100)

content_6 = """
Records with absolute error ≤ 5 g/km: 76.89%
Records with absolute error ≤ 10 g/km: 87.93%
Records with absolute error ≤ 20 g/km: 94.27%

## Percentile Distribution of Absolute Prediction Errors

| Percentile          | Absolute error (g/km) |
|---------------------|------------------------|
| 50th percentile (Median) | 1.9700              |
| 75th percentile     | 4.6000                 |
| 90th percentile     | 12.2200                |
| 95th percentile     | 22.8700                |
| 99th percentile     | 48.9100                |
| Maximum             | 141.1200               |

---

## Error Analysis for Seen/Unseen vs. Fuel Type/Fuel Mode

The final 2025 test results were analyzed jointly by configuration novelty and powertrain segment.

A record was labelled as **seen** when its core technical configuration — defined by vehicle mass, engine capacity, engine power, fuel type, and fuel mode — had already occurred in the combined 2024 training data. Otherwise, it was labelled as **unseen**.

| Configuration status | Fuel type     | Fuel mode | Records | Actual mean | Predicted mean | Mean residual | MAE    | RMSE   | Median absolute error | Pct error over 20 |
|----------------------|---------------|-----------|---------|-------------|----------------|---------------|--------|--------|------------------------|-------------------|
| unseen               | petrol/electric | P       | 404     | 44.8005     | 19.7275        | 25.0731       | 25.5461| 33.5587| 31.3815               | 59.9010           |
| unseen               | diesel/electric | P       | 22      | 37.9365     | 17.0100        | 20.9265       | 21.1726| 30.7355| 2.2297                | 45.4545           |
| seen                 | petrol/electric | P       | 2,062   | 34.9670     | 28.7708        | 6.1962        | 6.7481 | 17.4150| 0.4204                | 14.2095           |
| seen                 | diesel/electric | P       | 207     | 21.0521     | 13.2730        | 7.7791        | 8.0978 | 16.1256| 0.3335                | 23.6715           |
| unseen               | diesel          | H       | 240     | 146.5259    | 150.8798       | -4.3540       | 12.6889| 15.5007| 12.2451               | 25.4167           |
| unseen               | diesel          | M       | 496     | 175.8559    | 175.4246       | 0.4313        | 8.2570 | 11.7162| 4.4938                | 11.2903           |
| unseen               | petrol          | M       | 424     | 164.8132    | 163.4730       | 1.3402        | 5.8056 | 9.6622 | 3.3216                | 6.3679            |
| unseen               | petrol          | H       | 812     | 136.8319    | 138.6709       | -1.8390       | 7.3638 | 9.6365 | 6.5544                | 2.7094            |
| seen                 | diesel          | H       | 1,036   | 147.4675    | 149.9493       | -2.4818       | 4.9135 | 7.8278 | 2.6753                | 4.5367            |
| seen                 | petrol          | H       | 3,095   | 129.0048    | 129.7805       | -0.7757       | 3.3165 | 5.1481 | 1.9260                | 0.3877            |

---

## Key Findings: Unseen Petrol/Electric PHEVs Drive Most Large Errors

The joint analysis shows that the high-error tail in the 2025 temporal test set is concentrated overwhelmingly in previously unseen petrol/electric plug-in hybrid (PHEV) configurations.

For unseen petrol/electric vehicles in fuel mode `P`, the model produced a MAE of 25.55 g/km and an RMSE of 33.56 g/km. These vehicles had an average actual CO₂ value of 44.80 g/km, whereas the average prediction was only 19.73 g/km. This corresponds to a mean residual of +25.07 g/km, indicating systematic underprediction.

More than half of these unseen PHEV records (59.90%) had an absolute prediction error above 20 g/km. Their median absolute error was also high at 31.38 g/km, showing that the issue was not caused only by a small number of extreme outliers.

In contrast, seen petrol/electric PHEV configurations had a much lower MAE of 6.75 g/km. However, they still showed a positive mean residual of 6.20 g/km and an RMSE of 17.42 g/km, indicating that PHEVs remain more difficult to predict than conventional petrol and diesel vehicles even when their technical configurations have appeared in the training data.

For conventional petrol and diesel vehicles, prediction performance was substantially more stable. Seen configurations generally achieved MAE values between 2.70 and 4.91 g/km, while unseen conventional configurations achieved MAE values between 5.81 and 12.69 g/km. The largest non-PHEV error occurred for unseen diesel hybrid vehicles, with an MAE of 12.69 g/km and an RMSE of 15.50 g/km.

Therefore, configuration novelty alone does not explain the observed error distribution. The principal challenge is the interaction between configuration novelty and the petrol/electric PHEV segment. New PHEV configurations appear to have CO₂ emission patterns that are not adequately captured by patterns learned from historical data.
"""
st.markdown(content_6, unsafe_allow_html=False)


content_7 = """
## 7. Scientific Conclusions

1. **WLTP CO₂ emissions are highly predictable from vehicle specifications in this dataset.**  
   The strong performance is consistent with the structured physical and regulatory nature of the data, but it does not mean the model has discovered a causal law.

2. **Powertrain regime is central.**  
   Electric range, fuel type and fuel mode separate low-emission electrified vehicles from predominantly higher-emission combustion vehicles and explain much of the model’s predictive power.

3. **Group-aware validation materially improves scientific credibility.**  
   It prevents repeated or near-identical technical configurations from appearing on both sides of evaluation.

4. **The temporal test is essential.**  
   The final score is still excellent in 2025, but configuration overlap means a second benchmark with stricter unseen-model or unseen-family splits would provide a harder test.

5. **The deep-learning result confirms that model-family choice should follow both data structure and evaluation design.**  
   The tested MLP and ExtraTrees showed different error trade-offs, so any final head-to-head decision should use the same split, preprocessing, and reporting metrics.

---

## 8. Business Conclusions and Recommendations

The model can support early CO₂ estimation, quality checks and scenario analysis before official WLTP results are available, if input variables are available at prediction time. An MAE around 5.03 g/km on the reported 2025 test is useful for ranking and estimating vehicles, but it should not be treated as a regulatory replacement without subgroup and uncertainty analysis.

**Recommended deployment position:**

- Use tuned ExtraTrees as the primary prediction model.  
- Retain Random Forest as a benchmark alternative.  
- Keep the strict design-stage model for early product decisions.  
- Use the extended model only when electric-range and IT specifications are genuinely known before prediction.

**Recommended safeguards:**

- Report prediction intervals.  
- Monitor error by fuel mode, manufacturer, emissions range and unseen configuration status.  
- Prohibit post-test variables such as direct target proxies.  
- Retrain as new model years arrive.  
- Keep a transparent feature dictionary so business users understand required inputs.
"""
st.markdown(content_7, unsafe_allow_html=False)

content_8 = """# MODELING PART II: 
## Analysis On Segmented Datasets
## 1. Data Preparation and Experimental Design

The starting cleaned files contained 113,868 2024 records and 88,963 2025 records. Records were grouped by pre-test specification identifiers and technical attributes, and numeric measurements were replaced by registration-weighted means. This reduced the datasets to 15,957 unique 2024 records and 14,735 unique 2025 records.

The target was `Ewltp (g/km)`. The 2024 data were split with `GroupShuffleSplit` using mass, engine capacity, engine power, fuel type and fuel mode as the core group definition. Training and validation groups were disjoint, so validation measured performance on unseen core technical combinations. The 2025 dataset was kept untouched until final evaluation.

A central improvement was replacing a simple random-style benchmark with a group-aware split. Groups were defined from core technical characteristics: mass, engine capacity, engine power, fuel type and fuel mode. The final split had zero exact core-configuration overlap between training and validation. The 2025 set was not used for model selection or tuning.

Preprocessing was fitted on training data only and then applied to validation and test data. It included make target encoding, training-derived frequency maps, IT-code extraction, one-hot fuel encoding and scaling of continuous variables.

The extension notebook separated the data into **Mono-fuel (Fm=M)**, **PHEV (Fm=P)** and **Hybrid (Fm=H)**. Each subset was independently group-split, encoded and scaled. The fuel-mode column was removed because it was constant within each subset.

The 2024/2025 record counts were:

- **Mono:** 8,444 / 6,857  
- **PHEV:** 2,817 / 2,695  
- **Hybrid:** 4,696 / 5,183  

Training/validation counts were:

- **Mono:** 6,816 / 1,628  
- **PHEV:** 2,246 / 571  
- **Hybrid:** 3,745 / 951  

Each final subset model used `ExtraTreesRegressor` with:

- `n_estimators=388`  
- `max_depth=50`  
- `max_features=0.6702995511817258`  
- `min_samples_leaf=2`  
- `min_samples_split=5`  

Each model was then retrained on combined 2024 training and validation data.

---

## 2. Segmented Temporal Evaluation

The 2025 results show that model performance differs strongly by powertrain segment. Mono and Hybrid predictions were generally stable, whereas PHEV predictions were substantially more difficult, especially for unseen configurations.

### Fuel Mode: MONO

| 2025 subset              | Records | MAE    | RMSE   | R²      |
|--------------------------|---------|--------|--------|---------|
| All 2025 records         | 6,857   | 3.8459 | 5.9373 | 0.9746  |
| Seen core configurations | 5,937   | 3.4254 | 5.1633 | 0.9769  |
| Unseen core configurations| 920    | 6.5589 | 9.5233 | 0.9634  |

- Seen configurations: 5,937 / 6,857 (86.58%)  
- Unseen configurations: 920 / 6,857 (13.42%)  



### Fuel Mode: PHEV

| 2025 subset              | Records | MAE     | RMSE    | R²       |
|--------------------------|---------|---------|---------|----------|
| All 2025 records         | 2,695   | 9.6732  | 20.4547 | 0.5044   |
| Seen core configurations | 2,269   | 6.7831  | 17.1844 | 0.6289   |
| Unseen core configurations| 426    | 25.0665 | 32.7719 | -0.0684  |

- Seen configurations: 2,269 / 2,695 (84.19%)  
- Unseen configurations: 426 / 2,695 (15.81%)  



### Fuel Mode: HYBRID

| 2025 subset              | Records | MAE    | RMSE   | R²      |
|--------------------------|---------|--------|--------|---------|
| All 2025 records         | 5,183   | 4.9229 | 7.4179 | 0.9430  |
| Seen core configurations | 4,131   | 4.0925 | 6.3299 | 0.9541  |
| Unseen core configurations| 1,052  | 8.1839 | 10.6659| 0.9125  |

- Seen configurations: 4,131 / 5,183 (79.70%)  
- Unseen configurations: 1,052 / 5,183 (20.30%)  

Seen configurations were core technical configurations present in the corresponding 2024 training data. Coverage was 86.58% for Mono, 84.19% for PHEV and 79.70% for Hybrid. The negative R² for unseen PHEVs indicates poor generalisation to that subset.

---

## 3. Interpretation and Feature Importance

Permutation importance was computed on the 2025 test subsets by repeatedly permuting each feature and measuring the RMSE increase. The dominant features differed by segment.

| Segment | Most influential features        | RMSE increases (g/km)          |
|---------|----------------------------------|--------------------------------|
| Mono    | Mt, ep (KW), ec (cm3)           | 26.3757, 14.6895, 11.7005      |
| PHEV    | Electric range, ep (KW), Mt     | 10.4589, 1.8985, 0.3319        |
| Hybrid  | ep (KW), Mt, ec (cm3)           | 14.6276, 10.9463, 7.5671       |

These are predictive importance measures rather than causal effects. Electric range is strongly associated with PHEV operating and homologation behaviour, but it does not fully describe high-performance PHEVs.

---
"""
st.markdown(content_8, unsafe_allow_html=False)

content_9 = """
## 4. Error Analysis

### Residual Analysis

Residuals were defined as:

```math
{residual} = {actual CO2} - {predicted CO2}
```

Positive residuals indicate underprediction and negative residuals indicate overprediction.

- **Mono** and **Hybrid** residuals were mostly concentrated around zero.  
- **PHEV** residuals showed a wider spread and many large positive values.

This pattern confirms that the model is well calibrated for conventional Mono-fuel and Hybrid vehicles, while PHEV vehicles—especially unseen configurations—exhibit systematic under-prediction and higher uncertainty.

"""
st.markdown(content_9, unsafe_allow_html=False)

actual_mono = df_mono["actual_Ewltp"].to_numpy()
predicted_mono = df_mono["predicted_Ewltp"].to_numpy()

actual_phev = df_phev["actual_Ewltp"].to_numpy()
predicted_phev = df_phev["predicted_Ewltp"].to_numpy()

actual_hybrid = df_hybrid["actual_Ewltp"].to_numpy()
predicted_hybrid = df_hybrid["predicted_Ewltp"].to_numpy()


def plot_residuals_vs_actual_triplet(
    y_test_1,
    y_pred_1,
    label_1,
    y_test_2,
    y_pred_2,
    label_2,
    y_test_3,
    y_pred_3,
    label_3,
    plot_sample_size=20_000,
):
    rng = np.random.default_rng(42)

    def subsample_residuals(y_test, y_pred):
        y_actual = (
            y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
        )
        y_predicted = np.asarray(y_pred)

        if len(y_actual) != len(y_predicted):
            raise ValueError("y_test and y_pred must have the same length.")

        sample_size = min(plot_sample_size, len(y_actual))

        sample_positions = rng.choice(
            len(y_actual),
            size=sample_size,
            replace=False,
        )

        residuals = y_actual - y_predicted

        return (
            y_actual[sample_positions],
            residuals[sample_positions],
        )

    actual_1, residuals_1 = subsample_residuals(y_test_1, y_pred_1)
    actual_2, residuals_2 = subsample_residuals(y_test_2, y_pred_2)
    actual_3, residuals_3 = subsample_residuals(y_test_3, y_pred_3)

    lower_y_limit = min(
        residuals_1.min(),
        residuals_2.min(),
        residuals_3.min(),
    )

    upper_y_limit = max(
        residuals_1.max(),
        residuals_2.max(),
        residuals_3.max(),
    )

    y_padding = (upper_y_limit - lower_y_limit) * 0.05

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=True,
    )

    def plot_axis(ax, actual, residuals, label):
        ax.scatter(
            actual,
            residuals,
            alpha=0.15,
            s=12,
        )

        ax.axhline(
            y=0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Zero residual",
        )

        ax.set_xlabel("Actual WLTP CO₂ emissions (g/km)")
        ax.set_ylabel("Residual: Actual - Predicted (g/km)")
        ax.set_title("2025 Temporal Test: Residuals vs Actual CO₂\n" f"{label}")
        ax.set_ylim(
            lower_y_limit - y_padding,
            upper_y_limit + y_padding,
        )
        ax.legend()

    plot_axis(axes[0], actual_1, residuals_1, label_1)
    plot_axis(axes[1], actual_2, residuals_2, label_2)
    plot_axis(axes[2], actual_3, residuals_3, label_3)

    plt.tight_layout()
    return fig


fig = plot_residuals_vs_actual_triplet(
    actual_mono,
    predicted_mono,
    "Mono",
    actual_phev,
    predicted_phev,
    "PHEV",
    actual_hybrid,
    predicted_hybrid,
    "Hybrid",
)
st.pyplot(fig, width="stretch")

content_10 = """
### Largest Prediction Errors

The largest Mono error was 74.16 g/km: actual emissions were 369.0 g/km and the prediction was 443.16 g/km for a seen configuration. The largest PHEV error was 87.44 g/km for an unseen configuration with actual emissions of 140.0 g/km and a prediction of 52.56 g/km. The largest Hybrid error was 41.95 g/km for a seen configuration with actual emissions of 219.0 g/km and a prediction of 177.05 g/km.

PHEV fuel-type analysis confirmed systematic underprediction. Petrol/electric PHEVs had mean actual emissions of 36.5780 g/km, mean predictions of 27.3795 g/km, mean residual +9.1985 g/km and RMSE 20.6618 g/km. Diesel/electric PHEVs had mean residual +9.0703 g/km and RMSE 18.0744 g/km.
#### Top 20 Errors for Mono

|      | ec (cm3) | ep (KW) | actual_co2_g_km | predicted_co2_g_km | residual_g_km | absolute_error_g_km | configuration_status |
|------|----------|---------|------------------|---------------------|----------------|----------------------|-----------------------|
| 2693 | 3.9214   | 4.8993  | 369.0000         | 443.1600            | -74.1600       | 74.1600              | seen                  |
| 2052 | 3.1085   | 5.4120  | 315.0000         | 263.4400            | 51.5600        | 51.5600              | unseen                |
| 2765 | 2.1534   | 1.9942  | 180.0000         | 230.2100            | -50.2100       | 50.2100              | unseen                |
| 2764 | 2.1534   | 1.9942  | 183.0000         | 229.1600            | -46.1600       | 46.1600              | unseen                |
| 2499 | 4.6122   | 4.2779  | 263.0000         | 308.3700            | -45.3700       | 45.3700              | unseen                |
| 2712 | 0.5225   | 0.9223  | 252.0000         | 210.3000            | 41.7000        | 41.7000              | unseen                |
| 1317 | 4.6122   | 5.3654  | 258.0000         | 294.3800            | -36.3800       | 36.3800              | unseen                |
| 2550 | 3.9214   | 5.2101  | 319.0000         | 284.0700            | 34.9300        | 34.9300              | unseen                |
| 2551 | 3.9214   | 5.2101  | 319.0000         | 284.0700            | 34.9300        | 34.9300              | unseen                |
| 2549 | 3.9214   | 5.2101  | 319.0000         | 284.0700            | 34.9300        | 34.9300              | unseen                |
| 2552 | 3.9214   | 5.2101  | 319.0000         | 284.0700            | 34.9300        | 34.9300              | unseen                |
| 2491 | 5.6225   | 3.0817  | 316.0000         | 281.1700            | 34.8300        | 34.8300              | seen                  |
| 2490 | 5.6225   | 3.0817  | 313.0000         | 278.2600            | 34.7400        | 34.7400              | seen                  |
| 2492 | 5.6225   | 3.0817  | 317.0000         | 282.7400            | 34.2600        | 34.2600              | unseen                |
| 2301 | 0.5995   | 0.1610  | 226.0000         | 194.2300            | 31.7700        | 31.7700              | unseen                |
| 2300 | 0.5995   | 0.1610  | 224.0000         | 192.5000            | 31.5000        | 31.5000              | unseen                |
| 2299 | 0.5995   | 0.1610  | 226.0000         | 194.9000            | 31.1000        | 31.1000              | unseen                |
| 2297 | 0.5995   | 0.1610  | 224.0000         | 193.0900            | 30.9100        | 30.9100              | unseen                |
| 2773 | 0.5225   | 0.3941  | 322.0000         | 291.1500            | 30.8500        | 30.8500              | unseen                |
| 2298 | 0.5995   | 0.1610  | 224.0000         | 193.2000            | 30.8000        | 30.8000              | unseen                |


#### Top 20 Errors for PHEV

|      | ec (cm3) | ep (KW) | Electric range (km) | actual_co2_g_km | predicted_co2_g_km | residual_g_km | absolute_error_g_km | configuration_status |
|------|----------|---------|----------------------|------------------|---------------------|----------------|----------------------|-----------------------|
| 1215 | 3.2791   | 3.8035  | -0.4162              | 140.0000         | 52.5600             | 87.4400        | 87.4400              | unseen                |
| 1216 | 3.2791   | 3.8035  | -0.4162              | 140.0000         | 52.5600             | 87.4400        | 87.4400              | unseen                |
| 1227 | -0.0604  | -0.0894 | -1.1848              | 154.0000         | 68.4100             | 85.5900        | 85.5900              | seen                  |
| 1229 | -0.0604  | -0.0894 | -1.1499              | 152.0000         | 68.5800             | 83.4200        | 83.4200              | seen                  |
| 1228 | -0.0604  | -0.0894 | -1.1149              | 145.0600         | 66.6100             | 78.4500        | 78.4500              | seen                  |
| 1230 | -0.0604  | -0.0894 | -1.1149              | 144.5000         | 66.7700             | 77.7300        | 77.7300              | seen                  |
| 1226 | -0.0604  | -0.0894 | -1.0950              | 143.2900         | 65.9400             | 77.3500        | 77.3500              | seen                  |
| 2344 | 3.2791   | 3.6029  | -0.1454              | 116.0000         | 39.4900             | 76.5100        | 76.5100              | seen                  |
| 1212 | 7.4589   | 5.8236  | -2.2678              | 350.0000         | 273.6500            | 76.3500        | 76.3500              | unseen                |
| 2348 | 3.2791   | 3.6029  | -0.1437              | 115.2000         | 39.5100             | 75.6900        | 75.6900              | seen                  |
| 2349 | 3.2791   | 3.6029  | -0.1367              | 115.0000         | 39.3300             | 75.6700        | 75.6700              | seen                  |
| 2343 | 3.2791   | 3.6029  | -0.1017              | 113.0000         | 38.4800             | 74.5200        | 74.5200              | unseen                |
| 2346 | 3.2791   | 3.6029  | -0.0668              | 111.6700         | 37.4400             | 74.2300        | 74.2300              | seen                  |
| 2345 | 3.2791   | 3.6029  | -0.0668              | 111.0000         | 37.1000             | 73.9000        | 73.9000              | seen                  |
| 2347 | 3.2791   | 3.6029  | -0.0668              | 111.5000         | 37.6800             | 73.8200        | 73.8200              | seen                  |
| 8    | 3.2557   | 5.8503  | -2.3028              | 275.0000         | 201.4200            | 73.5800        | 73.5800              | unseen                |
| 2350 | 3.2791   | 3.6029  | -0.0319              | 109.0000         | 36.3100             | 72.6900        | 72.6900              | seen                  |
| 2660 | 1.6069   | 1.0477  | -1.1149              | 129.0000         | 60.7300             | 68.2700        | 68.2700              | unseen                |
| 1414 | 1.6119   | 2.1179  | 0.3019               | 96.5700          | 28.4300             | 68.1300        | 68.1300              | unseen                |
| 660  | 3.9457   | 3.4557  | -0.4543              | 109.7700         | 41.7900             | 67.9800        | 67.9800              | seen                  |


#### Top 20 Errors for Hybrid

|      | ec (cm3) | ep (KW) | Electric range (km) | actual_co2_g_km | predicted_co2_g_km | residual_g_km | absolute_error_g_km | configuration_status |
|------|----------|---------|----------------------|------------------|---------------------|----------------|----------------------|-----------------------|
| 1974 | 1.7945   | 0.7216  | 0.0000               | 219.0000         | 177.0500            | 41.9500        | 41.9500              | seen                  |
| 1956 | 1.7945   | 0.2024  | 0.0000               | 221.0000         | 179.7400            | 41.2600        | 41.2600              | seen                  |
| 1955 | 1.7945   | 0.2024  | 0.0000               | 221.0000         | 181.5200            | 39.4800        | 39.4800              | seen                  |
| 2463 | 0.2011   | 2.5535  | 0.0000               | 241.0000         | 203.6500            | 37.3500        | 37.3500              | unseen                |
| 2467 | 0.2011   | 2.5535  | 0.0000               | 236.0000         | 199.0500            | 36.9500        | 36.9500              | unseen                |
| 3395 | -0.5798  | -0.2447 | 0.0000               | 102.9600         | 139.9100            | -36.9500       | 36.9500              | unseen                |
| 3394 | -0.5798  | -0.2447 | 0.0000               | 102.0400         | 138.0800            | -36.0400       | 36.0400              | unseen                |
| 2462 | 0.2011   | 2.5535  | 0.0000               | 239.0000         | 203.7600            | 35.2400        | 35.2400              | unseen                |
| 2992 | 1.7961   | 2.8419  | 0.0000               | 278.0000         | 243.2000            | 34.8000        | 34.8000              | unseen                |
| 3392 | -0.5798  | -0.2447 | 0.0000               | 104.9700         | 139.5400            | -34.5600       | 34.5600              | unseen                |
| 3391 | -0.5798  | -0.2447 | 0.0000               | 103.8400         | 138.0800            | -34.2400       | 34.2400              | unseen                |
| 2461 | 0.2011   | 2.5535  | 0.0000               | 238.0000         | 203.8000            | 34.2000        | 34.2000              | unseen                |
| 2460 | 0.2011   | 2.5535  | 0.0000               | 238.0000         | 203.8800            | 34.1200        | 34.1200              | unseen                |
| 1975 | 1.7945   | 0.7216  | 0.0000               | 222.0000         | 188.3000            | 33.7000        | 33.7000              | seen                  |
| 2006 | 1.7945   | 1.8034  | 0.0000               | 238.0000         | 204.5400            | 33.4600        | 33.4600              | unseen                |
| 2002 | 1.7945   | 1.8034  | 0.0000               | 233.5000         | 200.0500            | 33.4500        | 33.4500              | unseen                |
| 2465 | 0.2011   | 2.5535  | 0.0000               | 232.0000         | 198.9800            | 33.0200        | 33.0200              | unseen                |
| 2004 | 1.7945   | 1.8034  | 0.0000               | 234.0000         | 201.2600            | 32.7400        | 32.7400              | unseen                |
| 1976 | 1.7945   | 0.7216  | 0.0000               | 226.0000         | 193.6900            | 32.3100        | 32.3100              | seen                  |
| 2022 | 4.0088   | 4.8180  | 0.0000               | 304.0000         | 272.9700            | 31.0300        | 31.0300              | unseen                |
"""
st.markdown(content_10, unsafe_allow_html=False)

content_11 = """
### Actual vs. predicted CO2 Emissions:
"""
st.markdown(content_11, unsafe_allow_html=False)


def plot_actual_vs_predicted_triplet(
    y_test_1,
    y_pred_1,
    label_1,
    y_test_2,
    y_pred_2,
    label_2,
    y_test_3,
    y_pred_3,
    label_3,
):
    plot_sample_size = 20_000

    rng = np.random.default_rng(42)

    def subsample(y_test, y_pred):
        n = len(y_test)
        size = min(plot_sample_size, n)
        positions = rng.choice(n, size=size, replace=False)
        y_test_array = (
            y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
        )
        y_pred_array = np.asarray(y_pred)
        return y_test_array[positions], y_pred_array[positions]

    s_actual_1, s_pred_1 = subsample(y_test_1, y_pred_1)
    s_actual_2, s_pred_2 = subsample(y_test_2, y_pred_2)
    s_actual_3, s_pred_3 = subsample(y_test_3, y_pred_3)

    lower_limit = min(
        s_actual_1.min(),
        s_pred_1.min(),
        s_actual_2.min(),
        s_pred_2.min(),
        s_actual_3.min(),
        s_pred_3.min(),
    )

    upper_limit = max(
        s_actual_1.max(),
        s_pred_1.max(),
        s_actual_2.max(),
        s_pred_2.max(),
        s_actual_3.max(),
        s_pred_3.max(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _plot_ax(ax, actual, predicted, title):
        ax.scatter(actual, predicted, alpha=0.15, s=12)
        ax.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Perfect prediction",
        )
        ax.set_xlim(lower_limit, upper_limit)
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_xlabel("Actual WLTP CO₂ emissions (g/km)")
        ax.set_ylabel("Predicted WLTP CO₂ emissions (g/km)")
        ax.set_title(title)
        ax.legend()

    _plot_ax(
        axes[0],
        s_actual_1,
        s_pred_1,
        f"2025 Temporal Test: Actual vs Predicted CO₂\n{label_1}",
    )
    _plot_ax(
        axes[1],
        s_actual_2,
        s_pred_2,
        f"2025 Temporal Test: Actual vs Predicted CO₂\n{label_2}",
    )
    _plot_ax(
        axes[2],
        s_actual_3,
        s_pred_3,
        f"2025 Temporal Test: Actual vs Predicted CO₂\n{label_3}",
    )

    plt.tight_layout()
    return fig


fig_actual_vs_predicted = plot_actual_vs_predicted_triplet(
    actual_mono,
    predicted_mono,
    "Mono",
    actual_phev,
    predicted_phev,
    "PHEV",
    actual_hybrid,
    predicted_hybrid,
    "Hybrid",
)
st.pyplot(fig_actual_vs_predicted)


content_12 = """
The scatter plot compares the actual and predicted WLTP CO₂ emissions for the 2025 temporal test set. The red dashed line represents perfect predictions, where the predicted value is equal to the actual value.

The Mono and Hybrid models showed strong agreement between actual and predicted WLTP CO₂ emissions, with most observations concentrated close to the perfect-prediction line. Prediction errors increased slightly for high-emission vehicles, where relatively few observations were available.

The PHEV model showed a wider error distribution and a clear tendency to underpredict CO₂ emissions for some vehicles. Several observations with moderate or high actual emissions were predicted substantially below the perfect-prediction line. This result is consistent with the segment-level error analysis, which identified unseen and high-performance petrol/electric PHEV configurations as the main source of large prediction errors.

## Distribution of Absolute Prediction Errors
"""
st.markdown(content_12, unsafe_allow_html=False)


def plot_absolute_error_distribution_triplet(
    y_test_1,
    y_pred_1,
    label_1,
    y_test_2,
    y_pred_2,
    label_2,
    y_test_3,
    y_pred_3,
    label_3,
    bins=60,
):
    def get_absolute_errors(y_test, y_pred):
        y_actual = (
            y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
        )

        y_predicted = np.asarray(y_pred)

        if len(y_actual) != len(y_predicted):
            raise ValueError("y_test and y_pred must have the same length.")

        return np.abs(y_actual - y_predicted)

    def create_error_summary(absolute_errors, label):
        return pd.DataFrame(
            {
                "Model": [label] * 9,
                "Metric": [
                    "50th percentile (Median)",
                    "75th percentile",
                    "90th percentile",
                    "95th percentile",
                    "99th percentile",
                    "Maximum",
                    "Error ≤ 5 g/km",
                    "Error ≤ 10 g/km",
                    "Error ≤ 20 g/km",
                ],
                "Value": [
                    np.percentile(absolute_errors, 50),
                    np.percentile(absolute_errors, 75),
                    np.percentile(absolute_errors, 90),
                    np.percentile(absolute_errors, 95),
                    np.percentile(absolute_errors, 99),
                    absolute_errors.max(),
                    (absolute_errors <= 5).mean() * 100,
                    (absolute_errors <= 10).mean() * 100,
                    (absolute_errors <= 20).mean() * 100,
                ],
                "Unit": [
                    "g/km",
                    "g/km",
                    "g/km",
                    "g/km",
                    "g/km",
                    "g/km",
                    "%",
                    "%",
                    "%",
                ],
            }
        )

    absolute_errors_1 = get_absolute_errors(y_test_1, y_pred_1)
    absolute_errors_2 = get_absolute_errors(y_test_2, y_pred_2)
    absolute_errors_3 = get_absolute_errors(y_test_3, y_pred_3)

    error_summary = pd.concat(
        [
            create_error_summary(absolute_errors_1, label_1),
            create_error_summary(absolute_errors_2, label_2),
            create_error_summary(absolute_errors_3, label_3),
        ],
        ignore_index=True,
    )

    plot_upper_limit = max(
        np.percentile(absolute_errors_1, 99.5),
        np.percentile(absolute_errors_2, 99.5),
        np.percentile(absolute_errors_3, 99.5),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharex=True,
    )

    def plot_axis(ax, absolute_errors, label):
        median_error = np.median(absolute_errors)
        mae = np.mean(absolute_errors)

        ax.hist(
            absolute_errors,
            bins=bins,
            range=(0, plot_upper_limit),
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            median_error,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_error:.2f} g/km",
        )

        ax.axvline(
            mae,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"MAE: {mae:.2f} g/km",
        )

        ax.set_xlabel("Absolute prediction error (g/km)")
        ax.set_ylabel("Number of 2025 records")
        ax.set_title("Distribution of Absolute Errors\n" f"{label}")
        ax.legend()

    plot_axis(axes[0], absolute_errors_1, label_1)
    plot_axis(axes[1], absolute_errors_2, label_2)
    plot_axis(axes[2], absolute_errors_3, label_3)

    plt.tight_layout()
    return fig


fig = plot_absolute_error_distribution_triplet(
    actual_mono,
    predicted_mono,
    "Mono",
    actual_phev,
    predicted_phev,
    "PHEV",
    actual_hybrid,
    predicted_hybrid,
    "Hybrid",
)

st.pyplot(fig, width="stretch")

content_13 = """

| Model  | Metric                  | Value    | Unit |
|--------|-------------------------|----------|------|
| Mono   | 50th percentile (Median)| 2.4300   | g/km |
| Mono   | 75th percentile         | 5.1300   | g/km |
| Mono   | 90th percentile         | 8.9400   | g/km |
| Mono   | 95th percentile         | 12.4400  | g/km |
| Mono   | 99th percentile         | 20.8900  | g/km |
| Mono   | Maximum                 | 74.1600  | g/km |
| Mono   | Error ≤ 5 g/km          | 74.2700  | %    |
| Mono   | Error ≤ 10 g/km         | 92.2100  | %    |
| Mono   | Error ≤ 20 g/km         | 98.7200  | %    |
| PHEV   | 50th percentile (Median)| 0.5300   | g/km |
| PHEV   | 75th percentile         | 2.1200   | g/km |
| PHEV   | 90th percentile         | 40.7400  | g/km |
| PHEV   | 95th percentile         | 49.4900  | g/km |
| PHEV   | 99th percentile         | 65.4000  | g/km |
| PHEV   | Maximum                 | 87.4400  | g/km |
| PHEV   | Error ≤ 5 g/km          | 77.0300  | %    |
| PHEV   | Error ≤ 10 g/km         | 77.5100  | %    |
| PHEV   | Error ≤ 20 g/km         | 78.0000  | %    |
| Hybrid | 50th percentile (Median)| 2.7200   | g/km |
| Hybrid | 75th percentile         | 6.7700   | g/km |
| Hybrid | 90th percentile         | 13.0900  | g/km |
| Hybrid | 95th percentile         | 16.2400  | g/km |
| Hybrid | 99th percentile         | 23.8700  | g/km |
| Hybrid | Maximum                 | 41.9500  | g/km |
| Hybrid | Error ≤ 5 g/km          | 67.5700  | %    |
| Hybrid | Error ≤ 10 g/km         | 83.8300  | %    |
| Hybrid | Error ≤ 20 g/km         | 97.7600  | %    |
"""
st.markdown(content_13, unsafe_allow_html=False)
