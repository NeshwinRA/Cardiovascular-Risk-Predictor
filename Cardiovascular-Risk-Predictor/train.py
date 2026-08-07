import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path="data/framingham.csv"):
    df = pd.read_csv(path)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def build_features(df):
    X = df.drop(["TenYearCHD", "education"], axis=1)
    y = df["TenYearCHD"]
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

    print("=== Logistic Regression ===")
    print(classification_report(y_test, lr_preds, digits=4))
    print(f"ROC-AUC: {lr_auc:.4f}\n")

    print("=== Random Forest ===")
    print(classification_report(y_test, rf_preds, digits=4))
    print(f"ROC-AUC: {rf_auc:.4f}\n")

    if rf_auc >= lr_auc:
        best_model = rf
        best_name = "Random Forest"
        best_auc = rf_auc
    else:
        best_model = lr
        best_name = "Logistic Regression"
        best_auc = lr_auc

    return scaler, best_model, best_name, best_auc, lr_auc, rf_auc


def save_artifacts(scaler, model, model_name, best_auc, lr_auc, rf_auc):
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(model, "models/best_model.pkl")
    info = {
        "selected_model": model_name,
        "best_auc": best_auc,
        "logistic_regression_auc": lr_auc,
        "random_forest_auc": rf_auc,
    }
    with open("models/model_info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"Saved scaler to models/scaler.pkl")
    print(f"Saved best model ({model_name}) to models/best_model.pkl")
    print("Saved model metadata to models/model_info.json")


def main():
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler, best_model, best_name, best_auc, lr_auc, rf_auc = train_and_evaluate(
        X_train_resampled, X_test, y_train_resampled, y_test
    )
    save_artifacts(scaler, best_model, best_name, best_auc, lr_auc, rf_auc)


if __name__ == "__main__":
    main()
