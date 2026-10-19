import streamlit as st

st.set_page_config(
    page_title="CO2 Project Live Report",
    page_icon="🚙",
    layout="wide",
)

home_page = st.Page(
    "app_pages/1.introduction.py",
    title="1. Introduction",
    icon="🏠",
)

exploration_page = st.Page(
    "app_pages/2.data_exploration.py",
    title="2. Data Exploration",
    icon="🔍",
)

visualization_page = st.Page(
    "app_pages/3.pre_processing_and_feature_engineering.py",
    title="3. Feature Engineering and Pre-processing",
    icon="📊",
)

modeling_page = st.Page(
    "app_pages/4.modeling.py",
    title="4. Model Selection and Fine Tuning",
    icon="👾",
)

conclusions_page = st.Page(
    "app_pages/5.conclusions.py",
    title="5. Conclusions",
    icon="📝",
)

try_for_yourself_page = st.Page(
    "app_pages/6.try_for_yourself.py",
    title="6. Demonstration",
    icon="🧩",
)

references_page = st.Page(
    "app_pages/7.references.py",
    title="7. References",
    icon="📚",
)

about_page = st.Page(
    "app_pages/8.about.py",
    title="8. About",
    icon="👤",
)

navigation = st.navigation(
    {
        "CO2 Project Live Report": [
            home_page,
            exploration_page,
            visualization_page,
            modeling_page,
            conclusions_page,
            try_for_yourself_page,
            references_page,
            about_page,
        ],
    },
    position="sidebar",
)

navigation.run()
