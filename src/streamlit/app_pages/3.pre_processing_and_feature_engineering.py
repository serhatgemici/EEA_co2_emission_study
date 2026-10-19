from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from ziko_st_toc import table_of_contents

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

df_2024_EEA_FR_cleaned = pd.read_csv(BASE_DIR / "data/processed/EU/2024_FR_cleaned.csv")

st.set_page_config(page_title="Pre-processing and Feature Engineering", layout="wide")

image_url = "https://miro.medium.com/max/1920/1*637kim0dxEy5AouQ2chsNQ.jpeg"
st.image(
    image_url, caption="Data Pre-processing and Feature Engineering", width="stretch"
)

with st.sidebar:
    table_of_contents()


content_1 = """
## 3. Data Pre-processing and Feature Engineering

### Initial cleaning
The preprocessing workflow included:

- Loading the datasets with pandas.
- Inspecting schemas and data types.
- Comparing variable availability between years and datasets.
- Removing duplicate records.
- Investigating inconsistent naming conventions.
- Identifying columns with complete missingness.
- Detecting constant and zero-variance columns.
- Inspecting invalid or inconsistent regulatory identifiers.
- Analysing missingness patterns and potential missing-not-at-random behaviour.

In the EEA 2024 data, the following fields were reported as 100% missing and were removed:

- MMS
- Enedc
- Ernedc
- De
- Vf
- ech

Several fields were constant:

- Country
- Status
- year
- r
- Cr

These variables were marked to be removed later because they provided no useful variation for the selected modeling task.

### Vehicle identifiers and regulatory codes
The following fields were carefully examined:

- VFN
- Tan
- T
- Va
- Ve
- RLFI

Although these codes contain useful regulatory information, they have very high cardinality and may cause the model to memorize vehicle families or laboratory configurations. Such memorisation would reduce generalization to new vehicles.

The VFN field was also checked for structural validity. Most records followed an IP-based format, but a small number of records contained irregular values such as missing components, alternative prefixes, or incomplete codes. These records were isolated for quality review rather than automatically treated as valid.

The decision was made to exclude high-cardinality regulatory identifiers from the principal model and focus on physical and meaningful categorical features.

### Manufacturer and model processing
The manufacturer feature was target-encoded to preserve information about average emissions by manufacturer while avoiding a very large one-hot-encoded matrix. This operation must be performed carefully inside the training pipeline to avoid target leakage between training and test data.

Model names were transformed using frequency encoding rather than one-hot encoding. This retained information about model frequency without creating thousands of sparse columns.

### Fuel-related processing
Fuel type and fuel mode were converted into categorical features using one-hot encoding. This approach avoids incorrectly assigning an ordinal relationship to categories.

The principal retained fuel categories were:

- Petrol.
- Petrol/electric.
- Petrol/diesel
- Diesel.

Low-frequency categories such as diesel/electric, E85, and LPG were removed in the modeling preparation stage because their small representation could produce unstable estimates.

The **ng** (natural gas) and **LPG** fuel categories were also removed because they represented a very small proportion of the data and lacked corresponding fuel-consumption information.

### Missing-value treatment
Missingness was not treated uniformly. The analysis recognized that some missing values are structural rather than accidental:

- Electric range and electric consumption are naturally absent for many combustion vehicles.
- Electric power fields are primarily relevant to electric or hybrid vehicles.
- Pollutant fields may be absent because testing requirements vary according to powertrain type.

Missing electric range and electric-consumption values were set to zero where zero represented “not applicable for a non-electric vehicle.” This assumption should be documented clearly because zero and unknown are not always equivalent.

### Target leakage prevention
Two variables were marked suspicious, because they provided information too directly related to the target:

- **Fuel consumption**: It has a near-deterministic physical relationship with CO₂ emissions and would make the prediction problem artificially easy.
- **Erwltp**: It represents an adjustment or regulatory value closely connected to WLTP emissions.

Removing these variables forces the model to learn from more fundamental vehicle characteristics such as mass, engine displacement, power, fuel type, and manufacturer.

### Train-test split
The data was initially divided using an 80/20 train-test split with random_state=42. The split was performed before scaling and outlier filtering to ensure that the test data remained unseen during model preparation.

For a stronger evaluation in the modeling phase, it was noted that it would also be useful to consider:

- Grouped splitting by vehicle model or manufacturer.
- Time-based splitting if multiple registration years become available.
- Evaluation on manufacturers or models not present in the training set.

### Standardization
Numerical variables were standardized using **StandardScaler**. The scaler was fitted only on the training data and then applied to the test data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization is useful for models that are sensitive to feature scale, including linear regression, support-vector machines, k-nearest neighbours, and neural networks. Tree-based models generally require less scale preparation, but using a consistent preprocessing pipeline remains good practice when comparing multiple algorithms.

### Dimension reduction
No mandatory dimension-reduction technique was required for the initial feature set because high-cardinality identifiers were removed or encoded before modeling. If the final encoded feature space becomes large, dimension reduction could be considered for exploratory purposes or for models that are sensitive to sparse, high-dimensional input.

---

## 4. Visualizations and Statistical Analysis

### Relationships between explanatory variables
The exploratory analysis identified several important relationships:

- Vehicle mass is positively associated with emissions.
- Engine power and engine displacement are positively associated with emissions.
- Mass and WLTP test mass are strongly related.
- Fuel type produces clear separation between low- and high-emission groups.
- Electric range and electric consumption are concentrated among electrified vehicles.
- Manufacturer and model distributions are highly concentrated.
- Some regulatory codes are strongly associated with specific vehicle families and manufacturers.

The strong relationship between **m (kg)** and **Mt** creates a multicollinearity concern. 
Although **Mt** is closely linked to laboratory testing, **m (kg)** is more appropriate for future prediction because curb mass may be available before the vehicle has completed official testing. Consequently, **m (kg)** was marked to be retained and **Mt** was marked to be excluded.
"""
st.markdown(content_1, unsafe_allow_html=False)

numeric_cols = df_2024_EEA_FR_cleaned.select_dtypes(include="number").columns.tolist()
corr_matrix = df_2024_EEA_FR_cleaned[numeric_cols].corr()

fig = plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation matrix for numerical features")
plt.xticks(rotation=40, ha="right")
plt.yticks(rotation=40, ha="right")
st.pyplot(fig, width=1100)

fig = plt.figure(figsize=(6, 4))
sns.regplot(
    data=df_2024_EEA_FR_cleaned[df_2024_EEA_FR_cleaned["Erwltp (g/km)"] > 0].sample(
        2000, random_state=42
    ),
    x="Erwltp (g/km)",
    y="Ewltp (g/km)",
    color="darkred",
    scatter_kws={"alpha": 0.3},
)
plt.title("Erwltp Leaks Post-Test Target Adjustments", fontsize=13, fontweight="bold")
plt.xlabel("Eco-Innovation Emissions Reduction (Erwltp)")
plt.ylabel("Target Emissions (Ewltp)")
st.pyplot(fig, width=1100)


content_2 = """
### Target distribution
The EEA target, **Ewltp (g/km)**, has:

- Mean of approximately 102.17 g/km.
- Median of approximately 127.00  g/km.
- Values ranging from 0 to approximately 456.00 g/km.
- A distribution containing a pronounced low-emission or zero-emission group and a higher-emission combustion group.

The zero values are consistent with the treatment of electric and hydrogen vehicles in the dataset. This creates a bimodal or zero-inflated target distribution rather than a simple symmetric distribution.

"""
st.markdown(content_2, unsafe_allow_html=False)

target_col = "Ewltp (g/km)"

if target_col not in df_2024_EEA_FR_cleaned.columns:
    st.error(f"Column '{target_col}' not found in the dataset.")
else:
    y = df_2024_EEA_FR_cleaned[target_col].dropna()

    # Basic stats
    mean_val = y.mean()
    median_val = y.median()
    min_val = y.min()
    max_val = y.max()

    with st.expander("Target statistics", expanded=False):
        st.write(f"Mean: {mean_val:.2f} g/km")
        st.write(f"Median: {median_val:.2f} g/km")
        st.write(f"Min: {min_val:.2f} g/km")
        st.write(f"Max: {max_val:.2f} g/km")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram + KDE
    sns.histplot(
        y,
        bins=80,
        kde=True,
        ax=ax,
        color="#4da6ff",
        alpha=0.8,
        line_kws={"color": "darkblue", "linewidth": 2},
    )

    ax.set_xlabel("Ewltp (g/km)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Distribution of WLTP CO₂ Emissions (Ewltp)",
        fontsize=14,
        fontweight="bold",
    )

    # A vertical line to draw attention to the zero
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero emissions")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig, width=1100)

content_3 = """
### Fuel-type relationships
The presentation identifies substantial differences between fuel categories:

- All-electric and hydrogen vehicles have an average tailpipe value close to 0 g/km.
- Diesel/electric plug-in hybrids have a reported average of approximately 14.78 g/km.
- Petrol/electric vehicles have a reported average of approximately 27.51 g/km.
- Petrol vehicles have a reported average of approximately 130.74 g/km.
- Diesel vehicles have a reported average of approximately 149.83 g/km.
- Unknown fuel entries have a reported average of approximately 214 g/km, although they represent a very small part of the dataset.

These differences indicate that fuel type is likely to be one of the most influential categorical predictors.

"""
st.markdown(content_3, unsafe_allow_html=False)

# Select only petrol and diesel vehicles;
# Keep the three columns needed for the chart and remove the missing rows
plot_df = (
    df_2024_EEA_FR_cleaned.loc[
        df_2024_EEA_FR_cleaned["Ft"].isin(["petrol", "diesel"]),
        ["Ft", "Mt", "Ewltp (g/km)"],
    ]
    .dropna()
    .copy()
)

# Take an equal number (10,000) of random observations from each fuel type
plot_df = (
    plot_df.groupby("Ft", group_keys=False)
    .sample(n=10_000, random_state=42)
    .reset_index(drop=True)
)

# Scatter plot + regression line for each fuel type
g = sns.lmplot(
    data=plot_df,
    x="Mt",
    y="Ewltp (g/km)",
    hue="Ft",
    scatter_kws={"alpha": 0.12, "s": 12},
    line_kws={"linewidth": 2},
    height=6,
    aspect=1.5,
)

g.set_axis_labels("WLTP test mass (kg)", "WLTP CO2 emissions (g/km)")

g.fig.suptitle("CO2 trend by WLTP test mass: petrol vs diesel", y=1.02)
st.pyplot(g.fig, width=1100)


plt.figure(figsize=(6, 4))
combustion_df = df_2024_EEA_FR_cleaned[df_2024_EEA_FR_cleaned["Fuel consumption"] > 0]

sns.scatterplot(
    data=combustion_df.sample(5000, random_state=42),
    x="Fuel consumption",
    y="Ewltp (g/km)",
    hue="Ft",
    alpha=0.4,
)
plt.title(
    "Near-Deterministic Relationship: Fuel Consumption vs CO2",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Fuel Consumption (l/100km)")
plt.ylabel("Ewltp (g/km)")
st.pyplot(plt.gcf(), width=1100)

content_4 = """
### Numerical distributions and outliers
The numerical variables have different distribution shapes:

- **ep (KW)** is strongly right-skewed, with skewness of approximately 2.24.
- **ec (cm3)** is right-skewed, with skewness of approximately 1.45.
- **Electric range (km)** has a skewness of approximately 1.41.
- **z (Wh/km)** has skewness of approximately 1.08.
- **m (kg)** has moderate right skewness, approximately 0.85.
- **Ewltp (g/km)** is comparatively close to symmetric at approximately -0.34, although its zero-emission group creates a meaningful mixture structure.
- **Fuel consumption** is also comparatively well behaved after appropriate filtering.

For **vehicle mass**, the mean is approximately 1,558.77 kg, the median is approximately 1,483 kg, and the range is approximately 513–4,800 kg. An IQR-based rule identifies many statistical outliers, but only a small number of records fall outside more conservative physical boundaries. This distinction is important: a statistical outlier is not automatically an erroneous vehicle.

For **engine power**, the mean is approximately 116.18 kW, while the maximum is approximately 1,352 kW. Very low or extremely high values require domain-specific validation rather than automatic deletion.
"""
st.markdown(content_4, unsafe_allow_html=False)


num_cols = [
    "m (kg)",
    "Electric range (km)",
    "Ewltp (g/km)",
    "ec (cm3)",
    "ep (KW)",
    "z (Wh/km)",
    "Fuel consumption",
    "Erwltp (g/km)",
]

# Keep only columns that actually exist in your dataset to avoid errors
existing_num_cols = [col for col in num_cols if col in df_2024_EEA_FR_cleaned.columns]

# Visual inspection of distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2x4 grid
axes = axes.ravel()

for i, col in enumerate(existing_num_cols):
    df_2024_EEA_FR_cleaned[col].hist(bins=30, ax=axes[i], edgecolor="black")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Hide unused subplots if fewer than 8 columns
for j in range(len(existing_num_cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Distribution of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

st.pyplot(fig, width="stretch")

# Print Skewness to verify outlier presence
st.markdown("#### Feature Skewness")

skew_df = df_2024_EEA_FR_cleaned[existing_num_cols].skew().to_frame(name="Skewness")
with st.expander("Feature Skewness", expanded=False):
    st.dataframe(skew_df, width="stretch")


content_5 = """
### Statistical analysis
The project used or prepared the following statistical methods:

- Descriptive statistics, including count, mean, median, standard deviation, minimum, maximum, and quartiles.
- Missing-value percentage analysis.
- Frequency and cardinality analysis for categorical variables.
- Skewness calculations for numerical columns.
- IQR-based outlier detection.
- Grouped mean comparisons by fuel type and manufacturer.
- Correlation analysis between numerical predictors and emissions.
- Distribution plots and histograms.
- Scatter plots for relationships such as fuel consumption and emissions.
- Comparative analysis between ADEME and EEA datasets.

The relationship between fuel consumption and WLTP emissions was found to be almost deterministic. This statistical and visual evidence supported the decision to remove fuel consumption from the model as a target-leaking feature.

For the modeling phase, the statistical analysis should be complemented by:

- Pearson or Spearman correlation tests for numerical associations.
- Group comparison tests across fuel categories.
- Confidence intervals for grouped emission means.
- Residual analysis after model training.
- Cross-validation and error analysis by fuel type, manufacturer, and vehicle category.

---

## 5. Conclusions Made For The Modeling Phase

The exploratory and preprocessing work supports the following modeling decisions:

- **Use physical vehicle characteristics as the principal predictors.**  
  Mass, engine displacement, power, fuel type, fuel mode, electric range, and energy consumption contain meaningful information about emissions.

- **Treat fuel type as a major predictor.**  
  The target distribution differs substantially between combustion, hybrid, and electric powertrains.

- **Avoid regulatory identifiers as direct model features.**  
  VFN, Tan, RLFI, T, Va, and Ve can have high cardinality, weak portability, and a high risk of memorization or target leakage.

- **Prefer curb mass over test mass for future prediction.**  
  Curb mass is more likely to be available before official testing, while test mass is closely tied to the laboratory procedure.

- **Exclude fuel consumption from the main predictive feature set.**  
  Although it is highly predictive, it provides an almost direct physical proxy for the target and would undermine the intended prediction task.

- **Handle electric vehicles carefully.**  
  Zero-emission vehicles create a distinct subpopulation. Separate modelling by powertrain or a two-stage modeling strategy may be appropriate.

- **Use a leakage-safe preprocessing pipeline.**  
  Target encoding, standardization, imputation, and feature selection must be fitted only on training data.

- **Evaluate generalization, not only random-split accuracy.**  
  Because vehicle models and manufacturers can repeat across the dataset, a random split may produce optimistic results. Grouped or time-based validation should be considered.

- **Report limitations alongside model performance.**  
  Predictions will estimate official WLTP-type emissions from available technical specifications; they will not represent real-world driving emissions or life-cycle carbon impact.

- **Preserve reproducibility.**  
  The final project should retain the data-cleaning rules, feature definitions, validation decisions, preprocessing pipeline, model parameters, and evaluation results.

Overall, the project has established a suitable foundation for supervised machine-learning experiments. The data exploration shows that the prediction problem is technically feasible, but the modeling phase must be designed carefully to manage strong categorical effects, zero-inflated electric features, missing-not-at-random data, manufacturer concentration, and leakage from regulatory or post-test variables.
"""
st.markdown(content_5, unsafe_allow_html=False)
