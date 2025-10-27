import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

video = pd.read_csv('features/video_features.csv')
text = pd.read_csv('features/text_features.csv')
labels = pd.read_csv('data/labels.csv')
df = video.merge(text, on='id').merge(labels, on='id')
X = df.drop(['id','label'], axis=1)
y = df['label']

models = {
    'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced', n_estimators=200),
    'XGB': xgb.XGBClassifier(scale_pos_weight=y.value_counts()[0]/y.value_counts()[1])
}

scoring = ['precision','recall','f1','roc_auc']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    print(f"\\n{name}")
    for s in scoring:
        print(f"{s}: {res['test_'+s].mean():.3f}")
