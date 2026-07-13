import pandas as pd
from itertools import combinations

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib 


# =====================================================
# Load Data
# =====================================================

data = pd.read_csv("posture_data_train.csv")

# First 1000 rows = Upright
# Next 1000 rows = Slouched

data["label"] = 0
data.loc[1000:1999, "label"] = 1

feature_cols = [
    "intershoulder_distance",
    "necklength_y",
    "slouchangle",
    "angle_of_rotation",
    "normalization_factor_distance",
    "nose_normalization_factor"
]

X = data[feature_cols]
y = data["label"]


# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# =====================================================
# Feature Search
# =====================================================

def find_best_feature_subset(
    model,
    X_train,
    y_train,
    feature_cols,
    model_name
):

    best_auc = 0
    best_features = None
    results = []

    for r in range(1, len(feature_cols) + 1):

        for subset in combinations(feature_cols, r):

            X_sub = X_train[list(subset)]

            auc = cross_val_score(
                model,
                X_sub,
                y_train,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            ).mean()

            results.append({
                "Model": model_name,
                "Features": subset,
                "Num Features": len(subset),
                "CV_AUC": auc
            })

            if auc > best_auc:
                best_auc = auc
                best_features = subset

    return best_features, best_auc, pd.DataFrame(results)


# =====================================================
# Models
# =====================================================

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
}


# =====================================================
# Run Models
# =====================================================

import joblib

best_overall_acc = 0
best_overall_model = None
best_overall_features = None
best_overall_name = None
best_overall_auc = 0



all_results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    best_features, best_cv_auc, result_df = find_best_feature_subset(
        model,
        X_train,
        y_train,
        feature_cols,
        model_name
    )

    all_results.append(result_df)

    print(f"Best Features: {best_features}")
    print(f"Best CV AUC : {best_cv_auc:.4f}")

    # -----------------------------------------
    # Fit final model on training set
    # -----------------------------------------

    model.fit(
        X_train[list(best_features)],
        y_train
    )

    # -----------------------------------------
    # Test set evaluation
    # -----------------------------------------

    y_pred = model.predict(
        X_test[list(best_features)]
    )

    y_prob = model.predict_proba(
        X_test[list(best_features)]
    )[:, 1]

    test_acc = accuracy_score(
        y_test,
        y_pred
    )

    test_auc = roc_auc_score(
        y_test,
        y_prob
    )

    
    if test_acc > best_overall_acc and test_auc > best_overall_auc:
        best_overall_acc = test_acc
        best_overall_model = model
        best_overall_features = best_features
        best_overall_name = model_name
        best_overall_auc = test_auc


    print("\nTest Results")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"AUC      : {test_auc:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

# =====================================================
# Save All Feature Search Results
# =====================================================

all_results = pd.concat(
    all_results,
    ignore_index=True
)

all_results = all_results.sort_values(
    "CV_AUC",
    ascending=False
)



print("\n")
print("=" * 70)
print("TOP 20 FEATURE COMBINATIONS")
print("=" * 70)

print(
    all_results.head(20)
)

all_results.to_csv(
    "feature_subset_results.csv",
    index=False
)

print("\nSaved to feature_subset_results.csv")

joblib.dump(
    {
        "model": best_overall_model,
        "features": list(best_overall_features),
        "accuracy": best_overall_acc,
        "auc": best_overall_auc
    },
    "best_posture_model.pkl"
)

print("\nSaved model to:")
print("best_posture_model.pkl")