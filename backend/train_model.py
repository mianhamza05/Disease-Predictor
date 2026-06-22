# """
# train_model.py
# Train a Random Forest classifier on the disease prediction dataset.
# Saves model and symptom list as pickle files.
# """
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder

# def train():
#     # Load data
#     train_df = pd.read_csv('Training.csv')
#     test_df = pd.read_csv('Testing.csv')

#     # Drop any unnamed columns (common in this dataset)
#     train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
#     test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

#     # Fill NaN with 0
#     train_df = train_df.fillna(0)
#     test_df = test_df.fillna(0)

#     # Features & target
#     feature_cols = [c for c in train_df.columns if c != 'prognosis']
#     X_train = train_df[feature_cols].values
#     y_train = train_df['prognosis'].values
#     X_test = test_df[feature_cols].values
#     y_test = test_df['prognosis'].values

#     # Encode labels
#     le = LabelEncoder()
#     y_train_enc = le.fit_transform(y_train)
#     y_test_enc = le.transform(y_test)

#     # Train Random Forest
#     print("Training Random Forest...")
#     model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train_enc)

#     # Evaluate
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test_enc, y_pred)
#     print(f"Test Accuracy: {acc:.4f}")

#     # Save model, encoder, and feature columns
#     with open('model.pkl', 'wb') as f:
#         pickle.dump(model, f)
#     with open('label_encoder.pkl', 'wb') as f:
#         pickle.dump(le, f)
#     with open('feature_cols.pkl', 'wb') as f:
#         pickle.dump(feature_cols, f)

#     print(f"Saved model.pkl, label_encoder.pkl, feature_cols.pkl")
#     print(f"Diseases: {list(le.classes_)}")
#     return acc

# if __name__ == '__main__':
#     train()


"""
train_model.py
Train and compare 7 classifiers on the disease prediction dataset.
Selects the best model based on accuracy, classification report, and generalization.
Saves the best model and supporting files as pickle files.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost. Install with: pip install xgboost")


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
    return models


def train():
    # Load data
    train_df = pd.read_csv('Training.csv')
    test_df  = pd.read_csv('Testing.csv')

    # Drop unnamed index columns if present
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    test_df  = test_df.loc[:,  ~test_df.columns.str.contains('^Unnamed')]

    # Fill missing values
    train_df = train_df.fillna(0)
    test_df  = test_df.fillna(0)

    # Split features and target
    feature_cols = [c for c in train_df.columns if c != 'prognosis']
    X_train = train_df[feature_cols].values
    y_train = train_df['prognosis'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['prognosis'].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    class_names = list(le.classes_)

    models = get_models()

    results = {}

    separator = "-" * 80

    print(separator)
    print("DISEASE PREDICTION - MODEL COMPARISON")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Classes          : {len(class_names)}")
    print(separator)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        print(separator)

        model.fit(X_train, y_train_enc)

        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_acc = accuracy_score(y_train_enc, y_train_pred)
        test_acc  = accuracy_score(y_test_enc,  y_test_pred)
        overfit_gap = train_acc - test_acc

        report = classification_report(y_test_enc, y_test_pred, target_names=class_names)

        print(f"Train Accuracy   : {train_acc:.4f}")
        print(f"Test Accuracy    : {test_acc:.4f}")
        print(f"Overfit Gap      : {overfit_gap:.4f}  (train - test, lower is better)")
        print(f"\nClassification Report on Test Set:\n")
        print(report)
        print(separator)

        results[name] = {
            "model":        model,
            "train_acc":    train_acc,
            "test_acc":     test_acc,
            "overfit_gap":  overfit_gap,
            "report":       report,
        }

    # Select best model
    # Primary  : highest test accuracy
    # Tiebreak : lowest overfit gap (better generalization)
    best_name = max(
        results,
        key=lambda n: (round(results[n]["test_acc"], 4), -round(results[n]["overfit_gap"], 4))
    )
    best = results[best_name]

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Overfit Gap':>12}"
    print(header)
    print("-" * 60)
    for name, r in sorted(results.items(), key=lambda x: x[1]["test_acc"], reverse=True):
        marker = "  <-- BEST" if name == best_name else ""
        print(f"{name:<25} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {r['overfit_gap']:>12.4f}{marker}")

    print("\n" + "=" * 80)
    print(f"BEST MODEL : {best_name}")
    print(f"  Test Accuracy  : {best['test_acc']:.4f}")
    print(f"  Train Accuracy : {best['train_acc']:.4f}")
    print(f"  Overfit Gap    : {best['overfit_gap']:.4f}")
    print("=" * 80)

    # Save best model and supporting artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump(best["model"], f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\nSaved  model.pkl          ({best_name})")
    print(f"Saved  label_encoder.pkl")
    print(f"Saved  feature_cols.pkl")
    print(f"\nDiseases ({len(class_names)}): {class_names}")

    return best_name, best["test_acc"]


if __name__ == '__main__':
    best_name, best_acc = train()
    print(f"\nDone. Best model: {best_name}  |  Test accuracy: {best_acc:.4f}")