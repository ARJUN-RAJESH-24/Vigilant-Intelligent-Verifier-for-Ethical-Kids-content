import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def main():
    # Load features
    text = pd.read_csv("features/text_features.csv")
    audio = pd.read_csv("features/audio_features.csv")
    video = pd.read_csv("features/video_features.csv")
    labels = pd.read_csv("data/labels.csv")

    df = text.merge(audio, on="id", how="left").merge(video, on="id", how="left").merge(labels, on="id", how="left")
    df = df.fillna(0)

    X = df.drop(columns=["id", "label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "SVM": SVC(kernel="linear", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "NaiveBayes": GaussianNB()
    }

    os.makedirs("models/trained_models", exist_ok=True)
    os.makedirs("models/results", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"\n✅ {name} Results:")
        print(classification_report(y_test, y_pred))

        # Save model
        joblib.dump(model, f"models/trained_models/{name}.pkl")

        # Save report
        pd.DataFrame(report).transpose().to_csv(f"models/results/{name}_report.csv", index=True)

if __name__ == "__main__":
    main()
