import numpy as np, pandas as pd, joblib, json, os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import folium

ROOT = Path.cwd()
PROC = ROOT/'data'/'processed'
OUTMODELS = ROOT/'models'; OUTMODELS.mkdir(parents=True, exist_ok=True)
OUTRESULTS = ROOT/'results'; OUTRESULTS.mkdir(parents=True, exist_ok=True)
OUTPLOTS = ROOT/'plots'; OUTPLOTS.mkdir(parents=True, exist_ok=True)

N_ESTIMATORS = 100  
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("Loading sequences (bycity)...")
a = np.load(PROC/'data_train_T7_bycity.npz')
X = a['X']; y = a['y']
print("X,y shapes:", X.shape, y.shape)
n,T,F = X.shape
Xf = X.reshape((n, T*F))

# quick train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

print("Training RandomForest (n_estimators=%d) ..." % N_ESTIMATORS)
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced')
clf.fit(X_train, y_train)

# predictions and metrics
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
report = classification_report(y_test, y_pred, digits=4)
auc_score = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n--- RandomForest Results ---")
print(report)
print("ROC AUC:", round(float(auc_score),4))
print("Confusion matrix:\n", cm)

# save model + metrics
joblib.dump(clf, OUTMODELS/'randomforest.pkl')
metrics = {'roc_auc': float(auc_score), 'confusion_matrix': cm.tolist(), 'report': report}
with open(OUTRESULTS/'rf_metrics.json','w') as f:
    json.dump(metrics, f, indent=2)
print("Saved model to", OUTMODELS/'randomforest.pkl')
print("Saved metrics to", OUTRESULTS/'rf_metrics.json')

importances = clf.feature_importances_

imp_mat = importances.reshape(T, F)
plt.figure(figsize=(8,4))
plt.imshow(imp_mat, aspect='auto')
plt.colorbar(label='importance')
plt.xlabel('feature index (0..%d)'%(F-1))
plt.ylabel('time step (0..%d)'%(T-1))
plt.title('RF feature importances (time x feature)')
plt.savefig(OUTPLOTS/'rf_feature_importance_matrix.png', dpi=150)
plt.close()
print("Saved feature importance matrix to", OUTPLOTS/'rf_feature_importance_matrix.png')

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC={auc_score:.4f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
plt.savefig(OUTPLOTS/'rf_roc_curve.png', dpi=150)
plt.close()
print("Saved ROC curve to", OUTPLOTS/'rf_roc_curve.png')

# Quick risk map: predict probability on meta and plot
meta_df = pd.read_csv(PROC/'data_train_T7_bycity_meta.csv')
if len(meta_df) != len(Xf):
    print("Meta length and X length differ: meta_len=%d, X_len=%d" % (len(meta_df), len(Xf)))
probas_all = clf.predict_proba(Xf)[:,1]
meta_df = meta_df.copy()
meta_df['rf_proba'] = probas_all
# create folium map centered on Uttarakhand approx
center = [29.2, 79.0]
m = folium.Map(location=center, zoom_start=7)
# add circle markers colored by risk
for _, row in meta_df.iterrows():
    lat = row['lat']; lon = row['lon']
    p = float(row['rf_proba'])
    import colorsys
    # use red for high risk: color from green (low) to red (high)
    r = int(255 * p); g = int(255 * (1-p)); b = 0
    folium.CircleMarker(location=[lat, lon], radius=4, color=None, fill=True, fill_opacity=0.7,
                        fill_color=f'#{r:02x}{g:02x}{b:02x}', popup=f"{row.get('city',row.get('cell_id',''))} {row.get('date','')}: {p:.3f}"
                    ).add_to(m)
# save map
map_path = OUTRESULTS/'rf_risk_map.html'
m.save(map_path)
print("Saved quick risk map to", map_path)

print("\nDone. Please open the JSON/PNG/HTML files in the results/ and plots/ folders.")
