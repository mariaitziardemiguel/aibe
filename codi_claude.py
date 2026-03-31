"""
MIMIC-IV ED — Predicción de Reingreso (Admisión) en 90 días
=============================================================
Target: ¿El paciente vuelve a urgencias y es ADMITIDO en los 90 días siguientes?

NOTA SOBRE UCI: Las tablas ED no contienen datos de UCI directamente.
  - `hadm_id` presente = el paciente fue admitido al hospital (proxy de UCI/planta)
  - Para distinguir UCI específicamente se necesitaría `icustays.csv` de MIMIC-IV core.
  - Usamos "reingreso con admisión hospitalaria en 90 días" como target clínicamente válido.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, average_precision_score,
                             precision_recall_curve)


# En este proyecto los CSV están en la misma carpeta que este script.
# Ajustamos la ruta para ser portable tanto en macOS como en entornos de servidor.
DATA_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════
print("📂 Cargando datos reales...")

ed       = pd.read_csv(DATA_DIR / "edstays.csv",   parse_dates=["intime","outtime"])
triage   = pd.read_csv(DATA_DIR / "triage.csv")
vitals   = pd.read_csv(DATA_DIR / "vitalsign.csv", parse_dates=["charttime"])
diag     = pd.read_csv(DATA_DIR / "diagnosis.csv")
medrecon = pd.read_csv(DATA_DIR / "medrecon.csv",  parse_dates=["charttime"])
pyxis    = pd.read_csv(DATA_DIR / "pyxis.csv",     parse_dates=["charttime"])

print(f"  edstays: {ed.shape} | {ed['subject_id'].nunique()} pacientes únicos")
print(f"  triage:  {triage.shape} | vitalsign: {vitals.shape}")
print(f"  diag:    {diag.shape}   | medrecon: {medrecon.shape} | pyxis: {pyxis.shape}")

# ══════════════════════════════════════════════════════════════
# 2. CONSTRUCCIÓN DEL TARGET
# ══════════════════════════════════════════════════════════════
print("\n🎯 Construyendo target (reingreso con admisión ≤90 días)...")

ed = ed.sort_values(["subject_id", "intime"]).reset_index(drop=True)
ed["next_intime"]      = ed.groupby("subject_id")["intime"].shift(-1)
ed["next_disposition"] = ed.groupby("subject_id")["disposition"].shift(-1)
ed["days_to_next"]     = (ed["next_intime"] - ed["outtime"]).dt.total_seconds() / 86400

ed["readmit_90d"] = (
    (ed["days_to_next"] <= 90) &
    (ed["next_disposition"] == "ADMITTED")
).astype(float)

# Última visita de cada paciente → sin follow-up → excluir
last_mask = ed.groupby("subject_id")["intime"].transform("max") == ed["intime"]
ed.loc[last_mask, "readmit_90d"] = np.nan

labeled = ed.dropna(subset=["readmit_90d"]).copy()
labeled["readmit_90d"] = labeled["readmit_90d"].astype(int)

print(f"  Visitas con target: {len(labeled)}")
print(f"  Positivos (reingreso): {labeled['readmit_90d'].sum()} ({labeled['readmit_90d'].mean():.1%})")

# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n🔧 Feature engineering...")

# ── 3a. Features de triage ──
triage["pain_num"] = pd.to_numeric(triage["pain"], errors="coerce")
triage_feats = triage.copy()

# ── 3b. Agregados de vitalsign por stay ──
vitals["pain_num"] = pd.to_numeric(vitals["pain"], errors="coerce")
vit_agg = vitals.groupby("stay_id").agg(
    hr_mean      = ("heartrate",   "mean"),
    hr_max       = ("heartrate",   "max"),
    hr_min       = ("heartrate",   "min"),
    hr_std       = ("heartrate",   "std"),
    o2_min       = ("o2sat",       "min"),
    o2_mean      = ("o2sat",       "mean"),
    sbp_min      = ("sbp",         "min"),
    sbp_mean     = ("sbp",         "mean"),
    temp_max     = ("temperature", "max"),
    resp_max     = ("resprate",    "max"),
    resp_mean    = ("resprate",    "mean"),
    pain_max_v   = ("pain_num",    "max"),
    n_vitals     = ("charttime",   "count"),
    has_afib     = ("rhythm",      lambda x: x.str.contains("FIBRIL|AFIB", case=False,
                                                              na=False).any().astype(int)),
).reset_index()

# ── 3c. Features de diagnosis ──
diag_agg = diag.groupby("stay_id").agg(
    n_diagnoses  = ("icd_code",  "count"),
    has_cardiac  = ("icd_title", lambda x: x.str.contains(
                      "HEART|CARDIAC|INFARCT|CORONARY|ARRHYTH|FIBRILL",
                      case=False, na=False).any().astype(int)),
    has_resp     = ("icd_title", lambda x: x.str.contains(
                      "PULMON|PNEUM|BRONCH|ASTHMA|COPD|RESPIR",
                      case=False, na=False).any().astype(int)),
    has_neuro    = ("icd_title", lambda x: x.str.contains(
                      "CEREBR|STROKE|HEMORRH|SEIZURE|ENCEPH",
                      case=False, na=False).any().astype(int)),
    has_sepsis   = ("icd_title", lambda x: x.str.contains(
                      "SEPSIS|SEPTIC", case=False, na=False).any().astype(int)),
    primary_icd  = ("icd_code",  "first"),
).reset_index()

# ── 3d. Features de medicamentos ──
med_agg = medrecon.groupby("stay_id").agg(
    n_meds_recon     = ("name",           "count"),
    n_meds_unique    = ("name",           "nunique"),
    has_anticoag     = ("etcdescription", lambda x: x.str.contains(
                          "ANTICOAG|HEPARIN|WARFARIN", case=False, na=False).any().astype(int)),
    has_diuretic     = ("etcdescription", lambda x: x.str.contains(
                          "DIURETIC", case=False, na=False).any().astype(int)),
).reset_index()

pyxis_agg = pyxis.groupby("stay_id").agg(
    n_pyxis_meds  = ("name",   "count"),
    n_pyxis_uniq  = ("name",   "nunique"),
).reset_index()

# ── 3e. Historial del paciente (visitas previas) ──
# CLAVE para reingreso: cuántas veces ha venido antes
ed_sorted = ed.sort_values(["subject_id", "intime"])
ed_sorted["visit_number"]    = ed_sorted.groupby("subject_id").cumcount() + 1
ed_sorted["n_prior_visits"]  = ed_sorted["visit_number"] - 1
ed_sorted["prior_admitted"]  = ed_sorted.groupby("subject_id").apply(
    lambda g: (g["disposition"] == "ADMITTED").shift(1).fillna(False).cumsum()
).reset_index(level=0, drop=True).astype(int)

# LOS de visita actual
ed_sorted["los_hours"] = (ed_sorted["outtime"] - ed_sorted["intime"]).dt.total_seconds() / 3600
ed_sorted["hour_in"]   = ed_sorted["intime"].dt.hour
ed_sorted["is_night"]  = ed_sorted["hour_in"].isin(range(0, 8)).astype(int)
ed_sorted["is_weekend"] = ed_sorted["intime"].dt.dayofweek.isin([5, 6]).astype(int)

# ── 3f. Encodings categóricos ──
le = LabelEncoder()
for col in ["gender", "race", "arrival_transport"]:
    ed_sorted[col + "_enc"] = le.fit_transform(ed_sorted[col].astype(str))

# Disposition actual codificada
ed_sorted["dispo_enc"] = le.fit_transform(ed_sorted["disposition"].astype(str))

# Chief complaint → top categorías como dummies
top_cc = triage_feats["chiefcomplaint"].str.lower().str.strip().value_counts().head(10).index
triage_feats["cc_clean"] = triage_feats["chiefcomplaint"].str.lower().str.strip()
triage_feats["cc_clean"] = triage_feats["cc_clean"].where(
    triage_feats["cc_clean"].isin(top_cc), other="other")
cc_dummies = pd.get_dummies(triage_feats["cc_clean"], prefix="cc").astype(int)
triage_feats = pd.concat([triage_feats, cc_dummies], axis=1)

# ══════════════════════════════════════════════════════════════
# 4. MERGE → TABLA MAESTRA
# ══════════════════════════════════════════════════════════════
print("🔗 Merging tablas...")

df = (labeled
      .merge(triage_feats.drop(columns=["subject_id","chiefcomplaint","pain",
                                         "cc_clean"], errors="ignore"),
             on="stay_id", how="left")
      .merge(vit_agg,   on="stay_id", how="left")
      .merge(diag_agg.drop(columns=["primary_icd"], errors="ignore"),
             on="stay_id", how="left")
      .merge(med_agg,   on="stay_id", how="left")
      .merge(pyxis_agg, on="stay_id", how="left")
      .merge(ed_sorted[["stay_id","visit_number","n_prior_visits","prior_admitted",
                         "los_hours","hour_in","is_night","is_weekend",
                         "gender_enc","race_enc","arrival_transport_enc","dispo_enc"]],
             on="stay_id", how="left")
)

# Rellenar ceros en features de conteo
zero_cols = ["n_diagnoses","has_cardiac","has_resp","has_neuro","has_sepsis",
             "n_meds_recon","n_meds_unique","has_anticoag","has_diuretic",
             "n_pyxis_meds","n_pyxis_uniq","n_vitals","has_afib","prior_admitted"]
for c in zero_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

print(f"  Tabla maestra: {df.shape}")

# ══════════════════════════════════════════════════════════════
# 5. SELECCIÓN DE FEATURES Y SPLIT
# ══════════════════════════════════════════════════════════════
cc_cols = [c for c in df.columns if c.startswith("cc_")]

FEATURES = (
    # Vitales triage
    ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain_num", "acuity"] +
    # Vitales serie temporal
    ["hr_mean","hr_max","hr_min","hr_std","o2_min","o2_mean",
     "sbp_min","sbp_mean","temp_max","resp_max","resp_mean","pain_max_v","n_vitals","has_afib"] +
    # Diagnósticos
    ["n_diagnoses","has_cardiac","has_resp","has_neuro","has_sepsis"] +
    # Medicamentos
    ["n_meds_recon","n_meds_unique","has_anticoag","has_diuretic",
     "n_pyxis_meds","n_pyxis_uniq"] +
    # Historial y contexto
    ["n_prior_visits","prior_admitted","los_hours",
     "hour_in","is_night","is_weekend"] +
    # Categóricas codificadas
    ["gender_enc","race_enc","arrival_transport_enc","dispo_enc"] +
    # Chief complaint dummies
    cc_cols
)
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES].copy()
y = df["readmit_90d"]

print(f"  Features usadas: {len(FEATURES)}")
print(f"  Muestra final: {len(X)} | Positivos: {y.sum()} ({y.mean():.1%})")

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ══════════════════════════════════════════════════════════════
# 6. MODELOS
# ══════════════════════════════════════════════════════════════
print("\n🤖 Entrenando modelos...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.5, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                       class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,
                                           max_depth=3, subsample=0.8,
                                           random_state=42)),
    ]),
    "Extra Trees": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                     class_weight="balanced",
                                     random_state=42, n_jobs=-1)),
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1).mean()
    results[name] = {"pipe": pipe, "y_prob": y_prob, "y_pred": y_pred,
                     "auc": auc, "ap": ap, "cv_auc": cv_auc}
    print(f"  {name:25s}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}  CV-AUC={cv_auc:.3f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n🏆 Mejor modelo: {best_name}  (ROC-AUC={best['auc']:.3f})")
print("\n" + classification_report(y_test, best["y_pred"],
                                   target_names=["No reingreso","Reingreso"]))

# ══════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE (nativa del modelo)
# ══════════════════════════════════════════════════════════════
print("🔍 Calculando feature importance...")
clf = best["pipe"].named_steps["clf"]
shap_df = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(15)

# ══════════════════════════════════════════════════════════════
# 8. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════
print("📊 Generando dashboard...")

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
BG     = "#F8FAFC"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     BG,
    "figure.facecolor":   BG,
})

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# ── P1: Target distribution ──
ax1 = fig.add_subplot(gs[0, 0])
counts = y.value_counts().sort_index()
bars = ax1.bar(["No reingreso\n(0)", "Reingreso\n(1)"],
               counts.values, color=[GREEN, RED], width=0.5, edgecolor="white")
for b, v in zip(bars, counts.values):
    ax1.text(b.get_x()+b.get_width()/2, v+0.5, f"{v}\n({v/len(y):.0%})",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_title("Distribución del Target\nReingreso ≤90 días (admisión)",
              fontweight="bold", fontsize=11)
ax1.set_ylabel("Nº visitas"); ax1.set_ylim(0, counts.max()*1.2)

# ── P2: Reingreso por disposición ──
ax2 = fig.add_subplot(gs[0, 1])
d = labeled.groupby("disposition")["readmit_90d"].agg(["mean","count"]).reset_index()
d = d[d["count"] >= 3].sort_values("mean", ascending=True)
bars2 = ax2.barh(d["disposition"], d["mean"]*100,
                 color=[RED if v > 0.35 else BLUE for v in d["mean"]])
ax2.axvline(labeled["readmit_90d"].mean()*100, color=GRAY, ls="--",
            alpha=0.7, label=f"Global {labeled['readmit_90d'].mean():.0%}")
ax2.set_xlabel("% Reingreso"); ax2.set_xlim(0, 60)
ax2.set_title("% Reingreso por Disposición\n(visita índice)", fontweight="bold", fontsize=11)
for b, (_, row) in zip(bars2, d.iterrows()):
    ax2.text(row["mean"]*100+0.5, b.get_y()+b.get_height()/2,
             f"n={int(row['count'])}", va="center", fontsize=8)
ax2.legend(fontsize=8)

# ── P3: Visitas previas vs reingreso ──
ax3 = fig.add_subplot(gs[0, 2])
vp_data = labeled.merge(ed_sorted[['stay_id','n_prior_visits']], on='stay_id', how='left')
vp = vp_data.groupby("n_prior_visits")["readmit_90d"].agg(["mean","count"]).reset_index()
vp = vp[vp["count"] >= 3]
sc = ax3.scatter(vp["n_prior_visits"], vp["mean"]*100,
                 s=vp["count"]*8, color=PURPLE, alpha=0.7, edgecolors="white")
ax3.set_xlabel("Nº visitas previas a urgencias")
ax3.set_ylabel("% Reingreso")
ax3.set_title("Historial de Visitas\nvs Tasa de Reingreso", fontweight="bold", fontsize=11)
# Línea de tendencia
z = np.polyfit(vp["n_prior_visits"], vp["mean"]*100, 1)
p = np.poly1d(z)
ax3.plot(sorted(vp["n_prior_visits"]), p(sorted(vp["n_prior_visits"])),
         color=RED, ls="--", alpha=0.6)

# ── P4: Acuity vs reingreso ──
ax4 = fig.add_subplot(gs[1, 0])
ac_data = labeled.merge(df[["stay_id","acuity"]].drop_duplicates(), on="stay_id", how="left")
ac = ac_data.dropna(subset=["acuity"]).groupby("acuity")["readmit_90d"].agg(["mean","count"]).reset_index()
colors4 = [RED, RED, AMBER, GREEN, BLUE]
bars4 = ax4.bar(ac["acuity"].astype(int), ac["mean"]*100,
                color=colors4[:len(ac)], edgecolor="white")
ax4.set_xticks(ac["acuity"].astype(int))
ax4.set_xticklabels([f"ESI-{int(a)}" for a in ac["acuity"]])
ax4.set_ylabel("% Reingreso"); ax4.set_title("% Reingreso por Acuidad ESI",
                                              fontweight="bold", fontsize=11)
for b, (_, row) in zip(bars4, ac.iterrows()):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
             f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

# ── P5: LOS vs reingreso boxplot ──
ax5 = fig.add_subplot(gs[1, 1])
los_data = labeled.merge(ed_sorted[['stay_id','los_hours']], on='stay_id', how='left')
los_pos = los_data[los_data["readmit_90d"]==1]["los_hours"].dropna()
los_neg = los_data[los_data["readmit_90d"]==0]["los_hours"].dropna()
bp = ax5.boxplot([los_neg, los_pos], patch_artist=True,
                 medianprops={"color":"white","lw":2.5}, notch=False)
bp["boxes"][0].set_facecolor(GREEN); bp["boxes"][1].set_facecolor(RED)
ax5.set_xticklabels(["No reingreso", "Reingreso"])
ax5.set_ylabel("LOS en urgencias (horas)")
ax5.set_title("Duración Estancia (LOS)\npor Grupo", fontweight="bold", fontsize=11)

# ── P6: Distribución días hasta reingreso ──
ax6 = fig.add_subplot(gs[1, 2])
days_pos = labeled[labeled["readmit_90d"]==1]["days_to_next"].dropna()
ax6.hist(days_pos, bins=15, color=RED, edgecolor="white", alpha=0.85)
ax6.axvline(days_pos.median(), color=AMBER, ls="--", lw=2,
            label=f"Mediana: {days_pos.median():.0f} días")
ax6.set_xlabel("Días hasta reingreso")
ax6.set_ylabel("Frecuencia")
ax6.set_title("Distribución Tiempo\nhasta Reingreso (positivos)", fontweight="bold", fontsize=11)
ax6.legend(fontsize=9)

# ── P7: ROC curves ──
ax7 = fig.add_subplot(gs[2, :2])
pal = [BLUE, GREEN, RED, PURPLE]
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax7.plot(fpr, tpr, color=pal[i], lw=2.5,
             label=f"{name}  AUC={res['auc']:.3f}  CV={res['cv_auc']:.3f}")
ax7.plot([0,1],[0,1], "k--", alpha=0.4, lw=1)
ax7.fill_between([0,1],[0,1], alpha=0.03, color="gray")
ax7.set_title("Curvas ROC — Predicción Reingreso 90 días",
              fontweight="bold", fontsize=12)
ax7.set_xlabel("Tasa Falsos Positivos (1 - Especificidad)")
ax7.set_ylabel("Tasa Verdaderos Positivos (Sensibilidad)")
ax7.legend(fontsize=9)

# ── P8: Confusion matrix ──
ax8 = fig.add_subplot(gs[2, 2])
cm  = confusion_matrix(y_test, best["y_pred"])
ax8.imshow(cm, cmap="RdYlGn", vmin=0)
labels = [["TN","FP"],["FN","TP"]]
for i in range(2):
    for j in range(2):
        ax8.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color="black")
ax8.set_xticks([0,1]); ax8.set_yticks([0,1])
ax8.set_xticklabels(["Pred: No reingreso","Pred: Reingreso"], fontsize=9)
ax8.set_yticklabels(["Real: No reingreso","Real: Reingreso"], fontsize=9)
ax8.set_title(f"Matriz de Confusión\n{best_name}", fontweight="bold", fontsize=11)

# ── P9: SHAP feature importance ──
ax9 = fig.add_subplot(gs[3, :])
colors9 = [RED if "prior" in f or "n_visit" in f or "dispo" in f
           else BLUE for f in shap_df.index]
bars9 = ax9.bar(range(len(shap_df)), shap_df.values, color=colors9, edgecolor="white")
ax9.set_xticks(range(len(shap_df)))
ax9.set_xticklabels(shap_df.index, rotation=35, ha="right", fontsize=9)
ax9.set_ylabel("SHAP mean |value|")
ax9.set_title(f"Top 15 Features — SHAP Importance ({best_name})\n"
              f"🔴 Historial del paciente   🔵 Clínicas",
              fontweight="bold", fontsize=12)
for b, v in zip(bars9, shap_df.values):
    ax9.text(b.get_x()+b.get_width()/2, v+0.0005, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7)

fig.suptitle(
    "MIMIC-IV ED — Predicción de Reingreso Hospitalario en 90 días\n"
    f"N={len(labeled)} visitas | {labeled['readmit_90d'].sum()} reingresos ({labeled['readmit_90d'].mean():.0%}) | "
    f"Mejor modelo: {best_name} (AUC={best['auc']:.3f})",
    fontsize=13, fontweight="bold", y=0.99
)

OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
out = OUTPUT_DIR / "mimic_readmit_90d.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✅ Dashboard guardado: {out}")

# ══════════════════════════════════════════════════════════════
# 9. RESUMEN FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("RESUMEN COMPARATIVO DE MODELOS")
print("═"*55)
print(f"{'Modelo':<26} {'ROC-AUC':>8} {'PR-AUC':>7} {'CV-AUC':>8}")
print("─"*55)
for name, res in sorted(results.items(), key=lambda x: -x[1]['auc']):
    mark = " ◄ MEJOR" if name == best_name else ""
    print(f"{name:<26} {res['auc']:>8.3f} {res['ap']:>7.3f} {res['cv_auc']:>8.3f}{mark}")

print(f"\nTop 5 features más importantes (SHAP):")
for i, (feat, val) in enumerate(shap_df.head(5).items(), 1):
    print(f"  {i}. {feat:<30} {val:.4f}")

print("\n⚠️  NOTA CLÍNICA:")
print("  Target = reingreso a urgencias con ADMISIÓN hospitalaria en 90 días.")
print("  Para target específico 'reingreso UCI' se necesita icustays.csv (MIMIC-IV core).")
print("  El hadm_id en edstays incluye ingresos a UCI y a planta general.")
