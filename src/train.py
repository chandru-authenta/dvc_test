import json, os, pickle, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import yaml

# # Optional experiment trackers
USE_WANDB = bool(os.environ.get("WANDB_API_KEY"))
USE_MLFLOW = bool(os.environ.get("MLFLOW_TRACKING_URI"))

if USE_WANDB:
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "dvc-ml"), config={})

if USE_MLFLOW:
    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "dvc-ml"))

# ---- Load params & data ----
params = yaml.safe_load(open("params.yaml"))
p = params["train"]

df = pd.read_csv("data/iris.csv")  # tracked by DVC
X = df.drop("species", axis=1)
y = df["species"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=p["test_size"], random_state=p["random_state"]
)

# ---- Train ----
if p["model"] == "logistic_regression":
    model = LogisticRegression(max_iter=p["max_iter"], n_jobs=None)
else:
    raise ValueError("Unsupported model")

model.fit(Xtr, ytr)
pred = model.predict(Xte)
acc = accuracy_score(yte, pred)
report = classification_report(yte, pred, output_dict=True)

# ---- Save model & artifacts ----
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Small table artifact for plots
cm_df = (
    pd.crosstab(yte, pred, rownames=["actual"], colnames=["predicted"])
    .reset_index()
)
cm_df.to_csv("artifacts/confusion_matrix.csv", index=False)

# ---- Metrics (for DVC) ----
metrics = {"accuracy": float(acc), "macro_f1": float(report["macro avg"]["f1-score"])}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---- Trackers (optional) ----
if USE_WANDB:
    wandb.config.update(p)
    wandb.log(metrics)
    wandb.save("artifacts/model.pkl")

if USE_MLFLOW:
    with mlflow.start_run():
        for k, v in p.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact("artifacts/model.pkl")
        mlflow.log_artifact("artifacts/confusion_matrix.csv")
        mlflow.log_artifact("metrics.json")

print(f"Accuracy: {acc:.4f}")
