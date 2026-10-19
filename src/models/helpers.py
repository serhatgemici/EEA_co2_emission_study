from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pandas as pd

EXTENDED_FEATURES_ALL = [
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

STRICT_DESIGN_FEATURES_ALL = [
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

EXTENDED_FEATURES_MONO = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
    "IT_28",
    "IT_29",
    "IT_32",
    "IT_33",
    "IT_37",
    "IT_38",
    "IT_none",
]

EXTENDED_FEATURES_PHEV = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_petrol/electric",
    "Ft_diesel/electric",
    "Electric range (km)",
    "IT_28",
    "IT_29",
    "IT_32",
    "IT_33",
    "IT_37",
    "IT_38",
    "IT_none",
]

EXTENDED_FEATURES_HYBRID = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
    "Electric range (km)",
    "IT_28",
    "IT_29",
    "IT_32",
    "IT_33",
    "IT_37",
    "IT_38",
    "IT_none",
]

STRICT_DESIGN_FEATURES_MONO = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
]

STRICT_DESIGN_FEATURES_PHEV = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_petrol/electric",
    "Ft_diesel/electric",
]

STRICT_DESIGN_FEATURES_HYBRID = [
    "m (kg)",
    "ec (cm3)",
    "ep (KW)",
    "Ft_diesel",
    "Ft_petrol",
]


def train_final_model(
    model_name: str,
    feature_set: list[str],
    X_train_encoded: pd.DataFrame,
    X_val_encoded: pd.DataFrame,
    X_test_encoded: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
):
    final_features = [
        column
        for column in feature_set
        if column in X_train_encoded.columns
        and column in X_val_encoded.columns
        and column in X_test_encoded.columns
    ]

    if not final_features:
        raise ValueError(
            "None of the selected features exist in the loaded training data."
        )

    X_trainval_final = pd.concat(
        [
            X_train_encoded[final_features],
            X_val_encoded[final_features],
        ],
        axis=0,
    )

    y_trainval_final = pd.concat(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    X_test_final = X_test_encoded[final_features].copy()

    if model_name == "ExtraTreesRegressor":
        final_model = ExtraTreesRegressor(
            max_depth=None,
            max_features=0.82680517164919,
            min_samples_leaf=3,
            min_samples_split=2,
            n_estimators=678,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "RandomForestRegressor":
        final_model = RandomForestRegressor(
            max_depth=25,
            max_features=0.41235069657748147,
            min_samples_leaf=2,
            min_samples_split=9,
            n_estimators=791,
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    final_model.fit(X_trainval_final, y_trainval_final)

    y_pred_2025 = final_model.predict(X_test_final)

    return final_model, final_features, y_pred_2025


def build_prediction_input(
    final_features: list[str],
    vehicle_mass: float,
    engine_capacity: float,
    engine_power: float,
    fuel_type: str,
    fuel_mode: str,
    electric_range: float | None,
    selected_it_codes: list[str],
) -> pd.DataFrame:
    """
    Build one encoded input row using the selected model feature order.
    """
    row = {feature: 0 for feature in final_features}

    numeric_values = {
        "m (kg)": vehicle_mass,
        "ec (cm3)": engine_capacity,
        "ep (KW)": engine_power,
        "Electric range (km)": (electric_range if electric_range is not None else 0.0),
    }

    for feature_name, value in numeric_values.items():
        if feature_name in row:
            row[feature_name] = value

    fuel_feature = f"Ft_{fuel_type}"
    if fuel_feature in row:
        row[fuel_feature] = 1

    fuel_mode_feature = f"Fm_{fuel_mode}"
    if fuel_mode_feature in row:
        row[fuel_mode_feature] = 1

    it_features = [
        "IT_28",
        "IT_29",
        "IT_32",
        "IT_33",
        "IT_37",
        "IT_38",
    ]

    for it_feature in it_features:
        if it_feature in row:
            row[it_feature] = int(it_feature in selected_it_codes)

    if "IT_none" in row:
        row["IT_none"] = int(len(selected_it_codes) == 0)

    return pd.DataFrame([row], columns=final_features)
