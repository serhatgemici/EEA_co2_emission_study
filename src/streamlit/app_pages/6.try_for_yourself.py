import streamlit as st
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


from src.models.helpers import (
    EXTENDED_FEATURES_ALL,
    STRICT_DESIGN_FEATURES_ALL,
    EXTENDED_FEATURES_MONO,
    EXTENDED_FEATURES_PHEV,
    EXTENDED_FEATURES_HYBRID,
    STRICT_DESIGN_FEATURES_MONO,
    STRICT_DESIGN_FEATURES_PHEV,
    STRICT_DESIGN_FEATURES_HYBRID,
    train_final_model,
    build_prediction_input,
)

SPLIT_DIR = BASE_DIR / "data" / "processed" / "eu" / "splits" / "final"


# load data
@st.cache_data(show_spinner="Loading prepared data...")
def load_segment_splits(split_dir: Path, segment: str) -> dict:
    """
    Load **encoded (not scaled)** feature splits and target splits for one vehicle segment.
    """
    target_col = "Ewltp (g/km)"

    return {
        "X_train": pd.read_parquet(
            split_dir / f"X_train_2024_{segment}_encoded_final.parquet"
        ),
        "X_val": pd.read_parquet(
            split_dir / f"X_val_2024_{segment}_encoded_final.parquet"
        ),
        "X_test": pd.read_parquet(
            split_dir / f"X_test_2025_{segment}_encoded_final.parquet"
        ),
        "y_train": pd.read_parquet(split_dir / f"y_train_2024_{segment}.parquet")[
            target_col
        ],
        "y_val": pd.read_parquet(split_dir / f"y_val_2024_{segment}.parquet")[
            target_col
        ],
        "y_test": pd.read_parquet(split_dir / f"y_test_2025_{segment}.parquet")[
            target_col
        ],
    }


@st.cache_data(show_spinner="Loading complete dataset splits...")
def load_complete_splits(split_dir_str: str) -> dict:
    split_dir = Path(split_dir_str)
    target_col = "Ewltp (g/km)"

    return {
        "X_train": pd.read_parquet(split_dir / "X_train_2024_encoded_final.parquet"),
        "X_val": pd.read_parquet(split_dir / "X_val_2024_encoded_final.parquet"),
        "X_test": pd.read_parquet(split_dir / "X_test_2025_encoded_final.parquet"),
        "y_train": pd.read_parquet(split_dir / "y_train.parquet")[target_col],
        "y_val": pd.read_parquet(split_dir / "y_val.parquet")[target_col],
        "y_test": pd.read_parquet(split_dir / "y_test.parquet")[target_col],
    }


# Segmented Dataset Splits
mono_data = load_segment_splits(SPLIT_DIR, "Mono")
phev_data = load_segment_splits(SPLIT_DIR, "Phev")
hybrid_data = load_segment_splits(SPLIT_DIR, "Hybrid")


# Complete Dataset Splits
complete_data = load_complete_splits(str(SPLIT_DIR))


X_train_encoded = complete_data["X_train"]
X_val_encoded = complete_data["X_val"]
X_test_encoded = complete_data["X_test"]

y_train = complete_data["y_train"]
y_val = complete_data["y_val"]
y_test = complete_data["y_test"]


st.set_page_config(page_title="Try for Yourself", layout="wide")

st.markdown(
    """
    <style>
    .st-key-title_badge {
        transform: translate(-40px, -12px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(
    horizontal=True,
    vertical_alignment="center",
    gap="small",
):
    st.title("CO₂ Emission Prediction", width="content")

    with st.container(key="title_badge", width="content"):
        st.badge("Try for yourself", color="green")

st.divider()

st.subheader("Dataset, Feature Set, and Model Selection")
st.caption(
    "Select a dataset, feature set and model, then enter vehicle specifications "
    "to prepare an estimated WLTP CO₂ emission prediction."
)


# -------------------------
# Configuration options
# -------------------------
DATASET_RULES = {
    "Complete": {
        "fuel_types": [
            "petrol",
            "diesel",
            "petrol/electric",
            "diesel/electric",
        ],
        "fuel_modes": ["M", "P", "H"],
        "fixed_fuel_mode": None,
    },
    "Mono only": {
        "fuel_types": ["petrol", "diesel"],
        "fuel_modes": ["M"],
        "fixed_fuel_mode": "M",
    },
    "PHEV only": {
        "fuel_types": ["petrol/electric", "diesel/electric"],
        "fuel_modes": ["P"],
        "fixed_fuel_mode": "P",
    },
    "Hybrid only": {
        "fuel_types": ["petrol", "diesel"],
        "fuel_modes": ["H"],
        "fixed_fuel_mode": "H",
    },
}


FUEL_MODE_LABELS = {
    "M": "Mono",
    "P": "PHEV",
    "H": "Hybrid",
}


IT_CODE_LABELS = {
    "IT_28": "IT_28: Efficient 12V Alternator",
    "IT_29": "IT_29: Efficient LED Lighting",
    "IT_32": "IT_32: Thermal Insulation Glass / Solar Roof",
    "IT_33": "IT_33: Active Grille Shutter / Engine Compartment Encapsulation",
    "IT_37": "IT_37: 48V Motor-Generator",
    "IT_38": "IT_38: Smart Fuel Heater",
}


# -------------------------
# Feature-set selection map
# -------------------------
FEATURE_SET_MAP = {
    ("Complete", "Strict"): STRICT_DESIGN_FEATURES_ALL,
    ("Complete", "Extended"): EXTENDED_FEATURES_ALL,
    ("Mono only", "Strict"): STRICT_DESIGN_FEATURES_MONO,
    ("Mono only", "Extended"): EXTENDED_FEATURES_MONO,
    ("PHEV only", "Strict"): STRICT_DESIGN_FEATURES_PHEV,
    ("PHEV only", "Extended"): EXTENDED_FEATURES_PHEV,
    ("Hybrid only", "Strict"): STRICT_DESIGN_FEATURES_HYBRID,
    ("Hybrid only", "Extended"): EXTENDED_FEATURES_HYBRID,
}


# -------------------------
# Cached model training
# -------------------------
@st.cache_resource(show_spinner="Training selected model...")
def get_trained_model(
    model_name: str,
    dataset_name: str,
    feature_set_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
):
    feature_set = FEATURE_SET_MAP[(dataset_name, feature_set_name)]

    final_model, final_features, y_pred_2025 = train_final_model(
        model_name=model_name,
        feature_set=feature_set,
        X_train_encoded=X_train,
        X_val_encoded=X_val,
        X_test_encoded=X_test,
        y_train=y_train,
        y_val=y_val,
    )

    return final_model, final_features, y_pred_2025


# -------------------------
# Model configuration
# -------------------------
config_container = st.container(key="config_row")

with config_container:
    config_col1, config_col2, config_col3 = st.columns(3, gap="medium")

    with config_col1:
        dataset_option = st.radio(
            "Available datasets",
            options=[
                "Complete",
                "Mono only",
                "PHEV only",
                "Hybrid only",
            ],
            captions=[
                "Complete dataset containing all fuel modes",
                "Filtered dataset for mono vehicles",
                "Filtered dataset for PHEV vehicles",
                "Filtered dataset for hybrid vehicles",
            ],
            key="dataset_option",
        )

    with config_col2:
        feature_set_option = st.radio(
            "Available feature sets",
            options=["Strict", "Extended"],
            captions=[
                "Strict design features: m(kg), ec (cm3), ep (kWh), Ft, Fm",
                "Strict design features plus Electric range IT",
            ],
            key="feature_set_option",
        )
        if feature_set_option == "Strict":
            st.markdown("""
            ```python
            STRICT_DESIGN_FEATURES = [
                "m (kg)", 
                "ec (cm3)",
                "ep (KW)",
                "Ft",
                "Fm",
            ]
            ```""")
        elif feature_set_option == "Extended":
            st.markdown("""
                        ```python
                        EXTENDED_DESIGN_FEATURES = [
                            "m (kg)", 
                            "ec (cm3)",
                            "ep (KW)",
                            "Ft",
                            "Fm",
                            "Electric range",
                            "IT",
                        ]
                        ```""")

    with config_col3:
        model_option = st.radio(
            "Available models",
            options=[
                "ExtraTreesRegressor",
                "RandomForestRegressor",
            ],
            key="model_option",
        )
        if model_option == "ExtraTreesRegressor":
            st.markdown("""
            ```json
                best_params: {
                'max_depth': None, 
                'max_features': 0.82680517164919, 
                'min_samples_leaf': 3, 
                'min_samples_split': 2, 
                'n_estimators': 678
                }""")
        elif model_option == "RandomForestRegressor":
            st.markdown("""
                        ```json
                            best_params: {
                            'max_depth': 25, 
                            'max_features': 0.41235069657748147, 
                            'min_samples_leaf': 2, 
                            'min_samples_split': 9, 
                            'n_estimators': 791}""")


st.markdown(
    """
    <style>
    .st-key-config_row div[data-testid="stColumn"]:nth-child(-n+3) {
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

rules = DATASET_RULES[dataset_option]

if dataset_option == "Complete":
    selected_data = complete_data
elif dataset_option == "Mono only":
    selected_data = mono_data
elif dataset_option == "PHEV only":
    selected_data = phev_data
else:
    selected_data = hybrid_data


# -------------------------
# Selected configuration
# -------------------------
st.subheader("Selected configuration")

selected_col1, selected_col2, selected_col3 = st.columns(3, gap="medium")

with selected_col1:
    st.metric("Dataset", dataset_option, icon="🗂️", border=True)

with selected_col2:
    st.metric("Feature set", feature_set_option, icon="🧩", border=True)

with selected_col3:
    st.metric("Model", model_option, icon="🧠", border=True)

st.divider()


# -------------------------
# Vehicle input form
# -------------------------
st.subheader("Vehicle specifications")
st.caption("Enter the vehicle specifications ")
feature_dictionary = {
    "m": "Vehicle mass (kg)",
    "ec": "Engine capacity (cm³)",
    "ep": "Engine power (kW)",
    "Ft": "Fuel type",
    "Fm": "Fuel mode",
    "IT": "Eco innovative technologies",
}
with st.expander("View dictionary", expanded=False):
    st.dataframe(
        data=pd.DataFrame(
            list(feature_dictionary.items()),
            columns=["Feature", "Description"],
        ),
        width="stretch",
        hide_index=True,
    )

input_col1, input_col2, input_col3 = st.columns(3, gap="medium")

with input_col1:
    m = st.text_input(
        "Vehicle mass (m (kg))",
        value="1500.00",
        placeholder="e.g. 1500",
        key="vehicle_mass",
    )

with input_col2:
    ec = st.text_input(
        "Engine capacity (ec (cm³))",
        value="2000.00",
        placeholder="e.g. 2000",
        key="engine_capacity",
    )

with input_col3:
    ep = st.text_input(
        "Engine power (ep (kW))",
        value="100.00",
        placeholder="e.g. 100",
        key="engine_power",
    )


# -------------------------
# Fuel and electric-range inputs
# -------------------------
fuel_col1, fuel_col2, fuel_col3 = st.columns(3, gap="medium")

with fuel_col1:
    fuel_type = st.selectbox(
        "Fuel type (Ft)",
        options=rules["fuel_types"],
        key=f"fuel_type_{dataset_option}",
    )

with fuel_col2:
    if rules["fixed_fuel_mode"] is None:
        fuel_mode = st.selectbox(
            "Fuel mode (Fm)",
            options=rules["fuel_modes"],
            format_func=lambda value: FUEL_MODE_LABELS[value],
            key="fuel_mode_complete",
        )
    else:
        fuel_mode = rules["fixed_fuel_mode"]

        st.text_input(
            "Fuel mode (Fm)",
            value=FUEL_MODE_LABELS[fuel_mode],
            disabled=True,
            key=f"fixed_fuel_mode_{dataset_option}",
        )

with fuel_col3:
    if dataset_option == "Mono only":
        electric_range = None

    elif feature_set_option == "Extended":
        electric_range = st.text_input(
            "Electric range (km)",
            value="0.00",
            placeholder="e.g. 0 or 60",
            key=f"electric_range_{dataset_option}",
        )

    else:
        electric_range = None

        st.text_input(
            "Electric range (km)",
            value="Not included in Strict feature set",
            disabled=True,
            key=f"electric_range_disabled_{dataset_option}",
        )


# -------------------------
# Multiple IT-code checkboxes
# -------------------------
if feature_set_option == "Extended":
    st.subheader("Eco-innovative technologies")

    st.caption(
        "Select every applicable IT code. If none are selected, "
        "the model input will use IT_none."
    )

    it_col1, it_col2, it_col3 = st.columns(3, gap="medium")

    with it_col1:
        it_28 = st.checkbox(
            IT_CODE_LABELS["IT_28"],
            key=f"IT_28_{dataset_option}",
        )

        it_29 = st.checkbox(
            IT_CODE_LABELS["IT_29"],
            key=f"IT_29_{dataset_option}",
        )

    with it_col2:
        it_32 = st.checkbox(
            IT_CODE_LABELS["IT_32"],
            key=f"IT_32_{dataset_option}",
        )

        it_33 = st.checkbox(
            IT_CODE_LABELS["IT_33"],
            key=f"IT_33_{dataset_option}",
        )

    with it_col3:
        it_37 = st.checkbox(
            IT_CODE_LABELS["IT_37"],
            key=f"IT_37_{dataset_option}",
        )

        it_38 = st.checkbox(
            IT_CODE_LABELS["IT_38"],
            key=f"IT_38_{dataset_option}",
        )

    selected_it_codes = [
        code
        for code, selected in {
            "IT_28": it_28,
            "IT_29": it_29,
            "IT_32": it_32,
            "IT_33": it_33,
            "IT_37": it_37,
            "IT_38": it_38,
        }.items()
        if selected
    ]

    it_none = int(len(selected_it_codes) == 0)

else:
    selected_it_codes = []


# -------------------------
# Validation preview only
# -------------------------
if st.button("Validate input values"):
    try:
        vehicle_mass = float(m)
        engine_capacity = float(ec)
        engine_power = float(ep)

        electric_range_value = (
            float(electric_range) if electric_range is not None else None
        )

        if vehicle_mass <= 0:
            st.error("Vehicle mass must be greater than 0.")
            st.stop()

        if engine_capacity < 0:
            st.error("Engine capacity cannot be negative.")
            st.stop()

        if engine_power < 0:
            st.error("Engine power cannot be negative.")
            st.stop()

        if electric_range_value is not None and electric_range_value < 0:
            st.error("Electric range cannot be negative.")
            st.stop()

        st.success("Input values are valid.")

        preview_col1, preview_col2, preview_col3 = st.columns(3, gap="medium")

        with preview_col1:
            st.write("**Mass**")
            st.write(f"{vehicle_mass:,.2f} kg")

            st.write("**Engine capacity**")
            st.write(f"{engine_capacity:,.2f} cm³")

        with preview_col2:
            st.write("**Engine power**")
            st.write(f"{engine_power:,.2f} kW")

            st.write("**Electric range**")

            if dataset_option == "Mono only":
                st.write("Not applicable for Mono vehicles")
            elif electric_range_value is not None:
                st.write(f"{electric_range_value:,.2f} km")
            else:
                st.write("Not included in Strict feature set")

        with preview_col3:
            st.write("**Fuel type**")
            st.write(fuel_type)

            st.write("**Fuel mode**")
            st.write(FUEL_MODE_LABELS[fuel_mode])

            st.write("**Eco-innovation IT codes**")

            if feature_set_option == "Strict":
                st.write("Not included in Strict feature set")
            elif selected_it_codes:
                st.write(", ".join(selected_it_codes))
            else:
                st.write("None selected (IT_none = 1)")

        feature_set = FEATURE_SET_MAP[(dataset_option, feature_set_option)]

        preview_features = [
            column
            for column in feature_set
            if column in selected_data["X_train"].columns
            and column in selected_data["X_val"].columns
            and column in selected_data["X_test"].columns
        ]

        if not preview_features:
            st.error("None of the selected features exist in the loaded training data.")
            st.stop()

        model_input = build_prediction_input(
            final_features=preview_features,
            vehicle_mass=vehicle_mass,
            engine_capacity=engine_capacity,
            engine_power=engine_power,
            fuel_type=fuel_type,
            fuel_mode=fuel_mode,
            electric_range=electric_range_value,
            selected_it_codes=selected_it_codes,
        )

        with st.expander("View prepared model input", expanded=False):
            st.dataframe(model_input, width="stretch")

    except ValueError:
        st.error(
            "Please enter valid numeric values. " "For example: 1500, 2000, 100, or 60."
        )


st.divider()
# -------------------------
# Prediction
# -------------------------
if st.button("Predict CO₂ emission", type="primary", width="stretch"):
    try:
        vehicle_mass = float(m)
        engine_capacity = float(ec)
        engine_power = float(ep)

        electric_range_value = (
            float(electric_range) if electric_range is not None else None
        )

        if vehicle_mass <= 0:
            st.error("Vehicle mass must be greater than 0.")
            st.stop()

        if engine_capacity < 0:
            st.error("Engine capacity cannot be negative.")
            st.stop()

        if engine_power < 0:
            st.error("Engine power cannot be negative.")
            st.stop()

        if electric_range_value is not None and electric_range_value < 0:
            st.error("Electric range cannot be negative.")
            st.stop()

        model, final_features, _ = get_trained_model(
            model_name=model_option,
            dataset_name=dataset_option,
            feature_set_name=feature_set_option,
            X_train=selected_data["X_train"],
            X_val=selected_data["X_val"],
            X_test=selected_data["X_test"],
            y_train=selected_data["y_train"],
            y_val=selected_data["y_val"],
        )

        model_input = build_prediction_input(
            final_features=final_features,
            vehicle_mass=vehicle_mass,
            engine_capacity=engine_capacity,
            engine_power=engine_power,
            fuel_type=fuel_type,
            fuel_mode=fuel_mode,
            electric_range=electric_range_value,
            selected_it_codes=selected_it_codes,
        )

        predicted_co2 = float(model.predict(model_input)[0])

        previous_co2 = st.session_state.get("last_predicted_co2")

        delta_co2 = predicted_co2 - previous_co2 if previous_co2 is not None else None

        st.metric(
            label="Estimated WLTP CO₂ emission:",
            value=f"{predicted_co2:.2f} g/km",
            delta=f"{delta_co2:+.2f} g/km" if delta_co2 is not None else None,
            delta_color="inverse",
            border=True,
            icon=":material/data_check:",
        )

        st.session_state["last_predicted_co2"] = predicted_co2

        st.snow()

    except ValueError:
        st.error(
            "Please enter valid numeric values. " "For example: 1500, 2000, 100, or 60."
        )
