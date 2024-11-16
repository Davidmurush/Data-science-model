
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# realistic student data
np.random.seed(42)
n_samples = 1000  # Number of students

data = {
    "GPA": np.round(np.random.normal(3.0, 0.5, n_samples), 2),  # Normal distribution
    "Attendance_Rate": np.random.randint(50, 100, n_samples),  # Attendance percentage
    "Socioeconomic_Status": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.3, 0.5, 0.2]),
    "Program_Type": np.random.choice(["STEM", "Humanities", "Business", "Arts"], n_samples),
    "Age": np.random.randint(17, 25, n_samples),
    "Enroll": np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
}
#DataFrame
df = pd.DataFrame(data)

df["GPA"] = df["GPA"].clip(0.0, 4.0)

df = pd.get_dummies(df, columns=["Socioeconomic_Status", "Program_Type"], drop_first=True)

X = df.drop("Enroll", axis=1)
y = df["Enroll"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and optimize the model
model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# model Evaluation
y_pred = best_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Save model
joblib.dump(best_model, "enhanced_student_enrollment_model.pkl")
print("\nModel saved as 'enhanced_student_enrollment_model.pkl'")
