import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("customer.csv")
df = df[df["Churn"].notna()]

x = df.drop(["Churn", "CustomerID"], axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=42, stratify=y
)

num_attrbs = ['Age', 'Tenure', 'Usage Frequency',
              'Support Calls', 'Payment Delay',
              'Total Spend', 'Last Interaction']

cat_attrbs = ["Gender", "Subscription Type", "Contract Length"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipe = ColumnTransformer([
    ("num", num_pipe, num_attrbs),
    ("cat", cat_pipe, cat_attrbs)
])

rf_pipe = Pipeline([
    ("process", full_pipe),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_pipe.fit(x_train, y_train)

y_pred = rf_pipe.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

pickle.dump(rf_pipe, open("customer_churn_model.pkl", "wb"))
pickle.dump(full_pipe,open("preprocess.pkl","wb"))
