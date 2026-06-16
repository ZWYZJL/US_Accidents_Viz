import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

DATA_PATH = "src/data/accidents_cleaned_sample.csv"
MODEL_PATH = "models/severity_predictor.joblib"

df = pd.read_csv(DATA_PATH)
df["Start_Time"] = pd.to_datetime(df["Start_Time"])

df["Hour"] = df["Start_Time"].dt.hour
df["Month"] = df["Start_Time"].dt.month
df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["RushHour"] = df["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

target = "Severity"

numeric_features = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Hour",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "RushHour",
]

categorical_features = [
    "State",
    "Weather_Condition",
    "Wind_Direction",
]

binary_features = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
]

features = numeric_features + categorical_features + binary_features

df = df.dropna(subset=[target])
X = df[features].copy()
y = df[target].astype(int)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

binary_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", binary_transformer, binary_features),
    ]
)

model = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"模型已保存到：{MODEL_PATH}")