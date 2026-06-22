"""
generate_graphs.py
Trains 7 classifiers on the disease prediction dataset and produces
4 publication-quality graphs matching the ML research paper:
  Fig 1 - Accuracy Comparison
  Fig 2 - ROC Curve (Logistic Regression)
  Fig 3 - Confusion Matrix (Logistic Regression)
  Fig 4 - Precision, Recall, F1 Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize


# ── colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#185FA5",
    "teal":   "#1D9E75",
    "coral":  "#D85A30",
    "purple": "#534AB7",
    "amber":  "#BA7517",
    "gray":   "#5F5E5A",
    "green":  "#639922",
    "light_blue": "#B5D4F4",
    "light_teal": "#9FE1CB",
}
MODEL_COLORS = [
    "#185FA5", "#D85A30", "#1D9E75", "#534AB7",
    "#BA7517", "#5F5E5A", "#639922",
]
HIGHLIGHT = "#185FA5"   # best model accent

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E8E8E4",
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})


# ── load & prep data ─────────────────────────────────────────────────────────
def load_data(train_path="Training.csv", test_path="Testing.csv"):
    train = pd.read_csv(train_path).loc[:, lambda df: ~df.columns.str.contains("^Unnamed")].fillna(0)
    test  = pd.read_csv(test_path).loc[:,  lambda df: ~df.columns.str.contains("^Unnamed")].fillna(0)
    feat  = [c for c in train.columns if c != "prognosis"]
    le    = LabelEncoder()
    y_tr  = le.fit_transform(train["prognosis"].values)
    y_te  = le.transform(test["prognosis"].values)
    return train[feat].values, y_tr, test[feat].values, y_te, le, feat


# ── model zoo ────────────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic\nRegression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision\nTree":       DecisionTreeClassifier(random_state=42),
        "Random\nForest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":                  SVC(kernel="rbf", random_state=42, probability=True),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
        "Naive\nBayes":         GaussianNB(),
        "Gradient\nBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


# ── train & collect metrics ───────────────────────────────────────────────────
def train_all(X_tr, y_tr, X_te, y_te, models):
    results = {}
    for name, mdl in models.items():
        print(f"  Training {name.replace(chr(10),' ')}…")
        mdl.fit(X_tr, y_tr)
        yp   = mdl.predict(X_te)
        yp_tr= mdl.predict(X_tr)
        results[name] = {
            "model":     mdl,
            "test_acc":  accuracy_score(y_te,  yp),
            "train_acc": accuracy_score(y_tr,  yp_tr),
            "prec":      precision_score(y_te, yp, average="weighted", zero_division=0),
            "rec":       recall_score(y_te,    yp, average="weighted", zero_division=0),
            "f1":        f1_score(y_te,        yp, average="weighted", zero_division=0),
            "cm":        confusion_matrix(y_te, yp),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Accuracy Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_accuracy(results, save_path="fig1_accuracy_comparison.png"):
    names      = list(results.keys())
    test_vals  = [results[n]["test_acc"]  for n in names]
    train_vals = [results[n]["train_acc"] for n in names]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - w/2, test_vals,  w, label="Test accuracy",
                   color=MODEL_COLORS, edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, train_vals, w, label="Train accuracy",
                   color=[c + "55" for c in MODEL_COLORS],
                   edgecolor=MODEL_COLORS, linewidth=1.2, linestyle="--",
                   zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.93, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Fig 1: Model performance comparison — accuracy", fontsize=13, fontweight="bold", pad=12)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                f"{h*100:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#888780",       label="Test accuracy (solid)"),
        mpatches.Patch(facecolor="#88878055", edgecolor="#888780",
                       linestyle="--", linewidth=1.2, label="Train accuracy (hatched)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — ROC Curve (Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_roc(X_tr, y_tr, X_te, y_te, le, results,
             save_path="fig2_roc_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    n_cls   = len(le.classes_)

    roc_models = {
        "Logistic\nRegression": results["Logistic\nRegression"]["model"],
        "Random\nForest":       results["Random\nForest"]["model"],
        "SVM":                  results["SVM"]["model"],
        "Naive\nBayes":         results["Naive\nBayes"]["model"],
    }
    palette = [COLORS["blue"], COLORS["green"], COLORS["coral"], COLORS["purple"]]
    y_bin   = label_binarize(y_te, classes=np.arange(n_cls))

    for (name, mdl), col in zip(roc_models.items(), palette):
        prob = mdl.predict_proba(X_te)
        fpr_all, tpr_all = [], []
        for i in range(n_cls):
            if y_bin[:, i].sum() == 0:
                continue
            fp, tp, _ = roc_curve(y_bin[:, i], prob[:, i])
            fpr_all.append(fp); tpr_all.append(tp)
        all_fpr   = np.unique(np.concatenate(fpr_all))
        mean_tpr  = np.mean([np.interp(all_fpr, f, t) for f, t in zip(fpr_all, tpr_all)], axis=0)
        roc_auc_v = auc(all_fpr, mean_tpr)
        lw = 2.5 if name == "Logistic\nRegression" else 1.5
        ax.plot(all_fpr, mean_tpr, color=col, lw=lw,
                label=f"{name.replace(chr(10),' ')} (AUC = {roc_auc_v:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline (AUC = 0.50)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("Fig 2: ROC curves — micro-average per model", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Confusion Matrix (Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion(X_te, y_te, le, results,
                   save_path="fig3_confusion_matrix.png"):
    mdl  = results["Logistic\nRegression"]["model"]
    yp   = mdl.predict(X_te)
    cm   = confusion_matrix(y_te, yp)
    classes = [c[:18] for c in le.classes_]   # truncate long names

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.3, linecolor="#E8E8E4",
        cbar_kws={"shrink": 0.7},
        ax=ax
    )
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
    ax.set_ylabel("True label",      fontsize=12, labelpad=8)
    ax.set_title("Fig 3: Confusion matrix — Logistic Regression (test set)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Precision · Recall · F1 Score
# ═══════════════════════════════════════════════════════════════════════════════
def plot_metrics(results, save_path="fig4_precision_recall_f1.png"):
    names = list(results.keys())
    prec  = [results[n]["prec"] for n in names]
    rec   = [results[n]["rec"]  for n in names]
    f1    = [results[n]["f1"]   for n in names]

    x  = np.arange(len(names))
    w  = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - w,   prec, w, label="Precision", color=COLORS["blue"],   edgecolor="white", zorder=3)
    b2 = ax.bar(x,       rec,  w, label="Recall",    color=COLORS["teal"],   edgecolor="white", zorder=3)
    b3 = ax.bar(x + w,   f1,   w, label="F1 score",  color=COLORS["coral"],  edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.93, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title("Fig 4: Precision · Recall · F1 score — all models", fontsize=13, fontweight="bold", pad=12)

    for group in [b1, b2, b3]:
        for bar in group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data…")
    X_tr, y_tr, X_te, y_te, le, feat = load_data()
    print(f"  Train {X_tr.shape} | Test {X_te.shape} | Classes {len(le.classes_)}")

    print("\nTraining models…")
    models  = get_models()
    results = train_all(X_tr, y_tr, X_te, y_te, models)

    print("\nGenerating graphs…")
    plot_accuracy(results)
    plot_roc(X_tr, y_tr, X_te, y_te, le, results)
    plot_confusion(X_te, y_te, le, results)
    plot_metrics(results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<22} {'Test Acc':>9} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-"*60)
    for n, r in sorted(results.items(), key=lambda x: x[1]["test_acc"], reverse=True):
        label = n.replace("\n", " ")
        print(f"{label:<22} {r['test_acc']:>9.4f} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f}")

    best = max(results, key=lambda n: results[n]["test_acc"])
    print(f"\nBest model: {best.replace(chr(10),' ')}  |  Test accuracy: {results[best]['test_acc']:.4f}")
    print("\nAll 4 graphs saved successfully.")


if __name__ == "__main__":
    main()
