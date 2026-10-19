from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from ziko_st_toc import table_of_contents

with st.sidebar:
    table_of_contents()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
# load EEA sample dataset
df_2023_EEA_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2023_EEA_head_100.csv"
)
df_2024_EEA_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2024_EEA_head_100.csv"
)
df_2025_EEA_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2025_EEA_head_100.csv"
)

# load  ADEME sample dataset
df_2015_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2015_ADEME_head_100.csv"
)
df_2014_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2014_ADEME_head_100.csv"
)
df_2013_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2013_ADEME_head_100.csv"
)
df_2012_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2012_ADEME_head_100.csv"
)
df_2011_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2011_ADEME_head_100.csv"
)
df_2010_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2010_ADEME_head_100.csv"
)
df_2009_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2009_ADEME_head_100.csv"
)
df_2008_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2008_ADEME_head_100.csv"
)
df_2007_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2007_ADEME_head_100.csv"
)
df_2006_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2006_ADEME_head_100.csv"
)
df_2005_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2005_ADEME_head_100.csv"
)
df_2004_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2004_ADEME_head_100.csv"
)
df_2003_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2003_ADEME_head_100.csv"
)
df_2002_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2002_ADEME_head_100.csv"
)
df_2001_ADEME_head_100 = pd.read_csv(
    BASE_DIR / "data/streamlit_samples/df_2001_ADEME_head_100.csv"
)

df_2024_EEA_FR_cleaned = pd.read_csv(BASE_DIR / "data/processed/EU/2024_FR_cleaned.csv")

st.set_page_config(page_title="Data Exploration", layout="wide")

st.title("Data Exploration & Feature Engineering")
image_url = (
    "https://iqaccountant.com/wp-content/uploads/2024/05/data-exploration-1024x576.jpg"
)
st.image(image_url, caption="Data Exploration", width="stretch")

content1 = """
## 1. Understanding and Manipulation of Data

### Data framework
The project uses two principal data sources:

**European Environment Agency data**
- Data concerning CO₂ emissions from newly registered passenger vehicles in Europe.
- Submitted by EU member states under Regulation (EU) 2019/631.
- The main analytical dataset is the 2024 release.
- The presentation reports approximately 10.78 million rows and 29 country codes before filtering.
- The dataset includes technical vehicle specifications, fuel information, manufacturer information, registration information, and official WLTP emissions.
"""
st.markdown(content1, unsafe_allow_html=False)

eu_dataset_name = st.selectbox(
    "Select Year to Display Sample", ("2023", "2024", "2025")
)
if eu_dataset_name == "2023":
    st.dataframe(df_2023_EEA_head_100.head(5))
elif eu_dataset_name == "2024":
    st.dataframe(df_2024_EEA_head_100.head(5))
elif eu_dataset_name == "2025":
    st.dataframe(df_2025_EEA_head_100.head(5))

content2 = """
**French ADEME data**
- French governmental data concerning vehicles marketed in France.
- Covers historical years from 2001 to 2015.
- The schema changes considerably over time, ranging from approximately 12 to 30 columns.
- The 2015 dataset contains approximately 20,880 observations and detailed technical, emissions, pollutant, and administrative variables.
"""
st.markdown(content2, unsafe_allow_html=False)
ademe_dataset_name = st.selectbox(
    "Select Year to Display Sample",
    (
        "2015",
        "2014",
        "2013",
        "2012",
        "2011",
        "2010",
        "2009",
        "2008",
        "2007",
        "2006",
        "2005",
        "2004",
        "2003",
        "2002",
        "2001",
    ),
)
if ademe_dataset_name == "2015":
    st.dataframe(df_2015_ADEME_head_100.head(5))
elif ademe_dataset_name == "2014":
    st.dataframe(df_2014_ADEME_head_100.head(5))
elif ademe_dataset_name == "2013":
    st.dataframe(df_2013_ADEME_head_100.head(5))
elif ademe_dataset_name == "2012":
    st.dataframe(df_2012_ADEME_head_100.head(5))
elif ademe_dataset_name == "2011":
    st.dataframe(df_2011_ADEME_head_100.head(5))
elif ademe_dataset_name == "2010":
    st.dataframe(df_2010_ADEME_head_100.head(5))
elif ademe_dataset_name == "2009":
    st.dataframe(df_2009_ADEME_head_100.head(5))
elif ademe_dataset_name == "2008":
    st.dataframe(df_2008_ADEME_head_100.head(5))
elif ademe_dataset_name == "2007":
    st.dataframe(df_2007_ADEME_head_100.head(5))
elif ademe_dataset_name == "2006":
    st.dataframe(df_2006_ADEME_head_100.head(5))
elif ademe_dataset_name == "2005":
    st.dataframe(df_2005_ADEME_head_100.head(5))
elif ademe_dataset_name == "2004":
    st.dataframe(df_2004_ADEME_head_100.head(5))
elif ademe_dataset_name == "2003":
    st.dataframe(df_2003_ADEME_head_100.head(5))
elif ademe_dataset_name == "2002":
    st.dataframe(df_2002_ADEME_head_100.head(5))
elif ademe_dataset_name == "2001":
    st.dataframe(df_2001_ADEME_head_100.head(5))

content_3 = """
The datasets are publicly available through governmental and European data portals. The relevant sources identified in the presentation are the European Environment Agency and data.gouv.fr. The data is therefore not proprietary to the project team, although its collection, publication, and regulatory ownership remain with the relevant public institutions.

### Dataset volume
The EEA 2024 dataset is very large, with more than 10 million records in its original form. The presentation later describes a cleaned French 2024 dataset containing approximately 119,380 rows and 18 columns for the modelling preparation stage.

The ADEME data is considerably smaller but historically more heterogeneous. The 2015 ADEME dataset contains approximately 20,880 rows, while the French EEA comparison dataset contains approximately 28,051 records.

The large EEA dataset creates practical challenges involving memory use, processing time, categorical cardinality, duplicate handling, and efficient exploratory analysis.

---

## 2. Relevance of the Data

### Target variable
The principal target variable is:

- **Ewltp (g/km)**: official WLTP CO₂ emissions.

For the ADEME dataset, the corresponding target is:

- **co2_mixte**: combined CO₂ emissions in grams per kilometre.

The target represents an official laboratory or homologation value. It should not be interpreted as a direct measurement of real-world emissions under all driving conditions.

### Relevant variables
The most relevant candidate predictors include:

**For the EU dataset, important variables include:**
- **m**: vehicle mass in kilograms.
- **Mt**: WLTP test mass.
- **ec**: engine displacement in cubic centimetres.
- **ep**: maximum engine power in kilowatts.
- **Ft**: fuel type.
- **Fm**: fuel mode.
- **Electric range**: electric driving range.
- **z**: electric energy consumption in Wh/km.
- **Ct**: type-approval vehicle category.
- **Mk**: vehicle make or brand.
- **IT**: eco-innovation technology indicators.

**For the ADEME dataset, important variables include:**
- Vehicle mass.
- Maximum engine power.
- Combined fuel consumption.
- Fuel type.
- Hybrid status.
- Transmission type.
- Engine and pollutant measurements.
- Manufacturer and model information.

### Important dataset characteristics
Several characteristics of the data are particularly important:

- The EEA dataset is strongly imbalanced by country. Germany, France, Italy, and Spain account for approximately two-thirds of the records.
- The 2024 dataset mainly represents passenger vehicles. Approximately 98.79% of vehicle categories are reported as M1, while M1G represents approximately 1.20%.
- Fuel categories are unevenly represented. Petrol, diesel, all-electric, and petrol/electric vehicles make up most records.
- Electric variables are highly zero-inflated because they are generally not applicable to conventional internal-combustion vehicles.
- Manufacturer and model fields have high cardinality and require careful treatment.
- Some regulatory identifiers can indirectly encode test conditions or emissions information, making them unsuitable as direct model inputs.
- The EU EEA dataset is huge. Only for the year 2024 alone is 2.53 GB, containing millions of entries. In order to be able to carry our study, we will be working on a subset of 2024 data filtered for Country=France.  

### Data limitations
The following limitations were identified:

- The data represents newly registered vehicles, not the complete European vehicle fleet.
- Registrations are not equivalent to sales volume because the `r` field is constant at 1.
- Countries are unevenly represented.
- WLTP values are laboratory-based and do not capture traffic, weather, driving behaviour, maintenance, or road conditions.
- Electric-vehicle tailpipe CO₂ does not represent total life-cycle emissions.
- Country of registration does not necessarily indicate country of manufacture or actual country of use.
- Manufacturer and model names are not fully standardised.
- Some variables are systematically missing depending on powertrain type.
- The EEA dataset does not provide all historical fields available in the ADEME source.
- ADEME is limited to historical data and was discontinued after 2015.
"""
st.markdown(content_3, unsafe_allow_html=False)
# Complete 26-Variable comparison mapping
mapping_dict = {
    # Shared / Equivalent Concepts (11)
    "Make / Brand": (1, 1),
    "Commercial Name": (1, 1),
    "Fuel Type": (1, 1),
    "CO2 Emissions": (1, 1),
    "Vehicle Mass": (1, 1),
    "Engine Power (kW)": (1, 1),
    "Engine Capacity (cc)": (1, 1),
    "Type / Variant Code": (1, 1),
    "Gearbox / Transmission": (1, 1),
    "Electric Consumption/Range": (1, 1),
    "Manufacturer Name": (1, 1),
    # ADEME Exclusive (15)
    "NOx Emissions": (1, 0),
    "Particulate Matter (PM)": (1, 0),
    "CO Emissions": (1, 0),
    "Hydrocarbons (HC)": (1, 0),
    "Urban Fuel Consumption": (1, 0),
    "Extra-Urban Fuel Conso.": (1, 0),
    "Combined Fuel Conso.": (1, 0),
    "French Fiscal Power (CV)": (1, 0),
    "CNIT Code": (1, 0),
    "Market Segment (Gamme)": (1, 0),
    "Body Type (Carrosserie)": (1, 0),
    "Directive Code (V9)": (1, 0),
    "Hybrid Flag": (1, 0),
    "HC+NOx Total": (1, 0),
    "ADEME Update Date": (1, 0),
    # EEA Exclusive (15)
    "Wheelbase (w)": (0, 1),
    "Front Track Width (at1)": (0, 1),
    "Rear Track Width (at2)": (0, 1),
    "Registration Volume (r)": (0, 1),
    "Eco-Innovation Credit (Er)": (0, 1),
    "Type Approval No. (TAN)": (0, 1),
    "Vehicle Family Name (VFN)": (0, 1),
    "Member State (MS)": (0, 1),
    "EU Pool Name (Mp)": (0, 1),
    "National Make Name (Mhn)": (0, 1),
    "EEA Version Code (Ve)": (0, 1),
    "EEA Variant Code (Va)": (0, 1),
    "Reporting Status": (0, 1),
    "Reporting Year": (0, 1),
    "EEA Entry ID": (0, 1),
}

# Create DataFrame
df_all_vars = pd.DataFrame(
    [v[0] for v in mapping_dict.values()],
    index=mapping_dict.keys(),
    columns=["ADEME 2015"],
)
df_all_vars["EU EEA 2015 (FR)"] = [v[1] for v in mapping_dict.values()]

# Plot Full Heatmap
fig, ax = plt.subplots(figsize=(6, 8))

sns.heatmap(
    df_all_vars,
    annot=True,
    cbar=False,
    cmap=["#ffe6e6", "#4da6ff"],
    linewidths=1,
    linecolor="white",
    fmt="d",
    ax=ax,
)

ax.set_title(
    "Full Variable Matrix Across Both 26-Column Files\n(1 = Present, 0 = Missing)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

plt.yticks()
plt.xticks(fontweight="bold")
plt.tight_layout()

st.pyplot(fig, width="stretch")


content_4 = """
After comparative analysis between EEA and ADEME datasets, for reasons detailed in the data limitations section **we have decided to drop the ADEME dataset** and continue our study using **EEA dataset.** 

Furthermore, considering the file size of the EEA dataset, we have decided to continue our study by filtering the EEA dataset by **France** and limit our study to use **2024** final dataset for **training and validation** and **2025** provisional dataset for **testing.**

__Steps below outline the data filtering and preparation process for the EEA dataset filtered by **France** and organized by year.__

---
"""
st.markdown(content_4, unsafe_allow_html=False)
