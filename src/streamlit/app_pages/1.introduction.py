import streamlit as st

from ziko_st_toc import table_of_contents

with st.sidebar:
    table_of_contents()

st.set_page_config(page_title="CO₂ Emission Estimation", layout="wide")

st.title("European Vehicle CO₂ Emission Estimation Project")
image_url = "https://s45727.pcdn.co/wp-content/uploads/2024/11/Emissions-plane-CO2-Nov-2024.jpg.optimal.jpg"

st.image(image_url, caption="CO₂ emissions and transport", width="stretch")

content = """
## 1. Context

### Business integration
This project analyses European vehicle-registration and technical data to understand the relationship between vehicle characteristics and official WLTP CO₂ emissions. The project team does not work directly in the automotive sector; however, the subject is relevant to organizations involved in sustainability reporting, environmental monitoring, vehicle classification, regulatory compliance, and transport planning.

The intended outcome is a reusable data-analysis and machine-learning framework that can estimate CO₂ emissions from vehicle specifications before complete emissions information is available. Such a framework could support preliminary environmental assessments, comparisons between vehicle categories, and data-driven decision-making during product or fleet analysis. It should be understood as an analytical support tool rather than a replacement for official homologation tests.

### Technical perspective
From a technical perspective, the project involves working with large, heterogeneous and imperfect real-world datasets. The main activities include data acquisition, schema comparison, data cleaning, missing-value analysis, data validation, exploratory data analysis, feature engineering, categorical encoding, leakage prevention, outlier analysis, standardization, and preparation for machine-learning modelling.

Particular attention was given to vehicle-family identifiers, type-approval codes, manufacturer names, fuel categories, and variables with high cardinality. Several identifier and regulatory fields were excluded because they could cause overfitting or target leakage. The workflow was implemented using Python-based data-science tools, including pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, and category-encoding techniques.

### Economic perspective
A reliable emission-estimation framework could reduce the time and manual effort required to analyze large vehicle datasets. It could also help organisations identify emission patterns at an early stage, compare manufacturers or fuel technologies, and prioritize more detailed investigations.

The project does not claim to replace laboratory testing or official regulatory procedures. Instead, its economic value lies in enabling scalable preliminary analysis using existing data. By automating repetitive processing and providing reproducible analytical steps, the approach could support future reporting, monitoring, benchmarking, or decision-support applications.

### Scientific perspective
Scientifically, the project combines statistical analysis, machine learning, environmental data analysis, software engineering, and physical interpretation of vehicle emissions. It investigates how mass, engine displacement, engine power, fuel type, fuel mode, vehicle category, and manufacturer-related variables are associated with WLTP CO₂ emissions.

The interdisciplinary nature of the team is important. Dr. Ana Valencia Garcia contributes a physics background through a PhD in Theoretical Physics, while Serhat Gemici contributes expertise in computer engineering and software quality. The collaboration aims to produce a transparent, reproducible, and scientifically defensible approach to emission prediction.

---

## 2. Objectives

The principal objective is to examine the relationship between the technical specifications of newly registered vehicles in Europe and their official WLTP CO₂ emissions, and to prepare a model capable of estimating emissions from relevant vehicle characteristics.

The project has the following specific objectives:

- Analyze European vehicle-registration data and identify its structure, quality issues, limitations, and potential biases.
- Identify the technical and categorical variables most relevant to CO₂ emissions.
- Compare the French ADEME dataset with the European Environment Agency dataset.
- Clean and standardize the data so that it can be used reliably in machine-learning workflows.
- Detect missing values, inconsistent codes, duplicate records, extreme values, and structural anomalies.
- Engineer usable features from categorical and numerical variables.
- Prevent target leakage caused by variables that directly encode or reproduce the target value.
- Prepare a reproducible training and evaluation pipeline for future modelling.
- Investigate whether vehicle emissions can be estimated accurately from specifications available before official testing.


## Team expertise

| Team member          | Background and relevant expertise                                                                                                                                 |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ana Valencia Garcia  | PhD in Theoretical Physics. Brings expertise in scientific reasoning, physical interpretation, quantitative analysis, and the evaluation of environmental relationships. |
| Serhat Gemici        | BSc in Computer Engineering and MSc in Engineering Management. Brings experience in software quality, data processing, Python, test automation, and reproducible technical implementation. |

The team's combined expertise covers both the physical interpretation of emissions and the computational implementation required to process large datasets and prepare predictive models.


## Business-expert interaction

The project was developed primarily from publicly available regulatory and governmental data sources. No direct automotive-industry expert interaction is documented in the presentation. Consequently, the underlying problem definition was refined through the official data documentation, European emissions regulations, variable descriptions, exploratory analysis, and discussions with the project mentor.

If the project is extended, interviews with automotive homologation specialists, fleet managers, regulatory consultants, or emissions engineers would be valuable. Such interactions could help validate the interpretation of fields such as VFN, RLFI, type-approval identifiers, test mass, fuel mode, and eco-innovation codes.


## Similar projects and surrounding work

The project uses concepts commonly found in vehicle-emission analysis, regulatory reporting, fleet benchmarking, and environmental machine learning. However, no specific comparable internal company project is identified in the presentation. The current work contributes by combining two complementary perspectives:

- A historical French dataset from ADEME, useful for understanding earlier vehicle-data structures and data-quality challenges.
- A large European Environment Agency dataset containing 2024 new-registration data, used as the primary analytical basis.

The comparison between these sources helps distinguish historical French catalogue information from more recent European registration data and clarifies which variables can be compared across datasets.
"""

st.markdown(content, unsafe_allow_html=False)
