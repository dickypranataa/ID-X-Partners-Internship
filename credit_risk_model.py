

# 0. IMPORT LIBRARY
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (aman untuk server/script)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance

# Gaya plot global
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (10, 5)})


# 1. KONFIGURASI
DATA_PATH       = 'loan_data_2007_2014.csv'   
BEST_THRESHOLD  = 0.30                         
RANDOM_STATE    = 42                           
OUTPUT_DIR      = '.'                          


# 2. FITUR YANG DIPILIH `
FEATURES_FINAL = [
    # Informasi pinjaman
    'loan_amnt',          # Jumlah pinjaman yang diajukan
    'term',               # Tenor pinjaman (36 / 60 bulan)
    'int_rate',           # Suku bunga yang diberikan
    'installment',        # Cicilan bulanan
    'purpose',            # Tujuan penggunaan pinjaman

    # Informasi nasabah
    'emp_length',         # Lama bekerja
    'home_ownership',     # Status kepemilikan rumah
    'annual_inc',         # Pendapatan tahunan
    'verification_status',# Status verifikasi pendapatan
    'dti',                # Debt-to-Income ratio

    # Riwayat kredit
    'delinq_2yrs',                  # Keterlambatan dalam 2 tahun terakhir
    'earliest_cr_line',             # Tanggal kredit pertama kali dibuka
    'inq_last_6mths',               # Jumlah inquiry kredit 6 bulan terakhir
    'mths_since_last_delinq',       # Bulan sejak keterlambatan terakhir
    'mths_since_last_record',       # Bulan sejak record publik terakhir
    'open_acc',                     # Jumlah akun kredit aktif
    'pub_rec',                      # Jumlah record publik negatif
    'revol_bal',                    # Saldo kredit revolving
    'revol_util',                   # Utilisasi kredit revolving (%)
    'total_acc',                    # Total akun kredit sepanjang sejarah
    'collections_12_mths_ex_med',   # Koleksi tunggakan 12 bulan (non-medis)
    'acc_now_delinq',               # Jumlah akun yang saat ini menunggak
    'tot_cur_bal',                  # Total saldo kredit saat ini
    'total_rev_hi_lim',             # Total limit kredit revolving
    'addr_state',                   # Negara bagian alamat nasabah
]

TARGET = 'target'


# 3. LOAD & FILTER DATA
def load_and_filter_data(path: str) -> pd.DataFrame:
    """
    Muat data mentah, filter hanya pinjaman dengan status final
    (Fully Paid = Baik, Charged Off / Default / Late = Buruk).
    Exclude 'Current' untuk menghindari label noise.
    """
    print("=" * 65)
    print("  STEP 1: MEMUAT DATA")
    print("=" * 65)

    df_raw = pd.read_csv(path, low_memory=False)
    print(f"  Data asli  : {len(df_raw):,} baris x {df_raw.shape[1]} kolom")
    print()
    print("  Distribusi loan_status (sebelum filter):")
    print(df_raw['loan_status'].value_counts().to_string())
    print()

    good_status = {
        'Fully Paid',
        'Does not meet the credit policy. Status:Fully Paid'
    }
    bad_status = {
        'Charged Off',
        'Default',
        'Late (31-120 days)',
        'Does not meet the credit policy. Status:Charged Off'
    }

    df = df_raw[df_raw['loan_status'].isin(good_status | bad_status)].copy()
    df[TARGET] = df['loan_status'].apply(lambda x: 0 if x in good_status else 1)

    bad_rate = df[TARGET].mean()
    print(f"  Data setelah filter : {len(df):,} baris")
    print(f"  Nasabah Baik  (0)   : {(df[TARGET]==0).sum():,}")
    print(f"  Nasabah Buruk (1)   : {(df[TARGET]==1).sum():,}")
    print(f"  Default Rate        : {bad_rate:.1%}")

    return df


# 4. PREPROCESSING & FEATURE ENGINEERING
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data, melakukan feature engineering,
    dan encoding variabel kategorikal.
    """
    print()
    print("=" * 65)
    print("  STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 65)

    df_model = df[FEATURES_FINAL + [TARGET]].copy()

    # 1. TERM → integer (36 atau 60)
    df_model['term'] = df_model['term'].str.extract(r'(\d+)').astype(float)

    # 2. EMP_LENGTH → angka (0-10)
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4,  '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8,  '9 years': 9, '10+ years': 10
    }
    df_model['emp_length'] = df_model['emp_length'].map(emp_map)
    df_model['emp_length'].fillna(df_model['emp_length'].median(), inplace=True)

    # 3. EARLIEST_CR_LINE → Usia kredit dalam tahun (credit_age_years)
    REF_YEAR = 2015

    def parse_cr_year(date_str):
        try:
            return int(str(date_str).split('-')[1])
        except Exception:
            return np.nan

    df_model['credit_age_years'] = df_model['earliest_cr_line'].apply(parse_cr_year)
    df_model['credit_age_years'] = REF_YEAR - df_model['credit_age_years']
    df_model['credit_age_years'] = df_model['credit_age_years'].clip(0, 80)
    df_model.drop(columns=['earliest_cr_line'], inplace=True)

    # 4. FITUR RASIO BARU (domain knowledge perbankan)
    df_model['loan_to_income']       = df_model['loan_amnt'] / (df_model['annual_inc'] + 1)
    df_model['installment_to_income'] = df_model['installment'] / (df_model['annual_inc'] / 12 + 1)
    df_model['revol_bal_to_limit']    = df_model['revol_bal'] / (df_model['total_rev_hi_lim'] + 1)

    # 5. MTHS_SINCE_LAST_DELINQ: NaN = tidak pernah menunggak → isi 999
    df_model['mths_since_last_delinq'].fillna(999, inplace=True)
    df_model['mths_since_last_record'].fillna(999, inplace=True)

    # 6. Kolom numerik lainnya → isi NaN dengan median
    numeric_cols = df_model.select_dtypes(include='number').columns
    for col in numeric_cols:
        if df_model[col].isnull().sum() > 0:
            df_model[col].fillna(df_model[col].median(), inplace=True)

    # 7. ENCODING KATEGORIKAL → One-Hot Encoding
    cat_cols = [
        c for c in ['term', 'purpose', 'emp_length', 'home_ownership',
                     'verification_status', 'addr_state']
        if c in df_model.columns
        and str(df_model[c].dtype) in ['object', 'string', 'category', 'str']
    ]
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    print(f"  Fitur setelah encoding : {df_model.shape[1] - 1}")
    print(f"  Jumlah baris           : {len(df_model):,}")

    missing_check = df_model.isnull().sum()
    if missing_check.sum() > 0:
        print("  ⚠️  Missing values tersisa:")
        print(missing_check[missing_check > 0])
    else:
        print("  ✅ Tidak ada missing values.")

    return df_model


# 5. SPLIT DATA
def split_data(df_model: pd.DataFrame):
    """Split data: 60% train / 20% validation / 20% test (stratified)."""
    print()
    print("=" * 65)
    print("  STEP 3: MEMBAGI DATA (60/20/20 STRATIFIED)")
    print("=" * 65)

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp)

    print(f"  Train set : {len(X_train):,} baris | Default rate: {y_train.mean():.1%}")
    print(f"  Val set   : {len(X_val):,} baris  | Default rate: {y_val.mean():.1%}")
    print(f"  Test set  : {len(X_test):,} baris  | Default rate: {y_test.mean():.1%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# 6. TRAINING MODEL
def train_model(X_train, y_train):
    """
    Latih HistGradientBoostingClassifier sebagai base model.
    HGB dipilih karena:
      - Handle missing values secara native
      - Sangat efisien untuk dataset besar
      - Performa mendekati XGBoost/LightGBM tanpa dependensi tambahan
    """
    print()
    print("=" * 65)
    print("  STEP 4: MELATIH BASE MODEL")
    print("=" * 65)
    print("  Model: HistGradientBoostingClassifier")
    print("  Parameter:")
    print("    - max_iter=300, max_depth=6, learning_rate=0.05")
    print("    - min_samples_leaf=50, l2_regularization=0.1")
    print()

    base_model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=RANDOM_STATE
    )
    base_model.fit(X_train, y_train)
    print("  ✅ Base model selesai dilatih.")
    return base_model


# 7. KALIBRASI PROBABILITAS (PLATT SCALING)
def calibrate_model(base_model, X_val, y_val):
    """
    Kalibrasi probabilitas menggunakan Platt Scaling.
    FrozenEstimator digunakan agar base model tidak di-retrain
    (best practice scikit-learn >= 1.4).
    """
    print()
    print("=" * 65)
    print("  STEP 5: KALIBRASI PROBABILITAS (PLATT SCALING)")
    print("=" * 65)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method='sigmoid'   # Platt Scaling
    )
    calibrated_model.fit(X_val, y_val)
    print("  ✅ Kalibrasi selesai (Platt Scaling via FrozenEstimator).")
    return calibrated_model


# 8. EVALUASI MODEL
def evaluate_model(base_model, calibrated_model, X_val, X_test, y_val, y_test):
    """Evaluasi model pada test set dan cetak metrik utama."""
    print()
    print("=" * 65)
    print("  STEP 6: EVALUASI MODEL")
    print("=" * 65)

    # Prediksi probabilitas
    y_val_proba_base = base_model.predict_proba(X_val)[:, 1]
    y_val_proba_cal  = calibrated_model.predict_proba(X_val)[:, 1]
    y_test_proba     = calibrated_model.predict_proba(X_test)[:, 1]

    # Metrik
    auc_val  = roc_auc_score(y_val,  y_val_proba_base)
    auc_test = roc_auc_score(y_test, y_test_proba)

    fpr_v, tpr_v, _ = roc_curve(y_val,  y_val_proba_base)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_test_proba)
    ks_val  = max(tpr_v - fpr_v)
    ks_test = max(tpr_t - fpr_t)

    print(f"  Base Model (Validation)  : AUC = {auc_val:.4f}  |  KS = {ks_val:.2%}")
    print(f"  Calibrated (Test Set)    : AUC = {auc_test:.4f}  |  KS = {ks_test:.2%}")

    return (y_val_proba_base, y_val_proba_cal, y_test_proba,
            fpr_t, tpr_t, auc_test, ks_test)


# 9. ANALISIS THRESHOLD BISNIS
def threshold_analysis(y_test, y_test_proba) -> pd.DataFrame:
    """Analisis precision-recall-F1 pada berbagai threshold keputusan."""
    print()
    print("=" * 65)
    print("  STEP 7: ANALISIS THRESHOLD BISNIS")
    print("=" * 65)

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []

    for thresh in thresholds:
        y_pred = (y_test_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        results.append({
            'Threshold'              : thresh,
            'Precision (%)'          : round(precision * 100, 1),
            'Recall (%)'             : round(recall * 100, 1),
            'F1 Score'               : round(f1, 4),
            'Jumlah Ditolak'         : fp + tp,
            'Nasabah Buruk Tertangkap': tp,
            'Nasabah Baik Salah Tolak': fp
        })

    df_thresh = pd.DataFrame(results)
    print(df_thresh.to_string(index=False))
    return df_thresh


# 10. EVALUASI FINAL DENGAN THRESHOLD TERPILIH
def final_evaluation(y_test, y_test_proba, auc_final, ks_final, threshold=0.30):
    """Cetak laporan evaluasi akhir dengan threshold yang dipilih."""
    print()
    print("=" * 65)
    print(f"  STEP 8: EVALUASI FINAL (Threshold = {threshold})")
    print("=" * 65)

    y_pred_final = (y_test_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()

    print(classification_report(y_test, y_pred_final,
          target_names=['Nasabah Baik (0)', 'Nasabah Buruk (1)']))

    print(f"  ROC-AUC Score : {auc_final:.4f}")
    print(f"  KS Statistic  : {ks_final:.2%}")
    print()
    print("─" * 65)
    print("  INTERPRETASI BISNIS")
    print("─" * 65)
    total_bad  = tp + fn
    total_good = tn + fp
    print(f"  Dari {total_bad:,} Nasabah Buruk (sebenarnya):")
    print(f"    → BERHASIL DITOLAK  : {tp:,} ({tp/total_bad:.1%})  ✅ NPL terselamatkan")
    print(f"    → Lolos ke approval  : {fn:,} ({fn/total_bad:.1%})  ❌ Risiko NPL tersisa")
    print()
    print(f"  Dari {total_good:,} Nasabah Baik (sebenarnya):")
    print(f"    → BERHASIL DISETUJUI : {tn:,} ({tn/total_good:.1%})  ✅ Profit bunga masuk")
    print(f"    → Salah DITOLAK      : {fp:,} ({fp/total_good:.1%})  ⚠️  Opportunity cost")

    return tn, fp, fn, tp


# 11. PERMUTATION FEATURE IMPORTANCE
def compute_feature_importance(base_model, X_val, y_val):
    """Hitung permutation importance (metode yang valid untuk HGB)."""
    print()
    print("=" * 65)
    print("  STEP 9: FEATURE IMPORTANCE (PERMUTATION METHOD)")
    print("=" * 65)
    print("  Menghitung permutation importance (n_repeats=10)...")

    perm = permutation_importance(
        base_model, X_val, y_val,
        n_repeats=10,
        scoring='roc_auc',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    feat_imp = pd.DataFrame({
        'Fitur'     : X_val.columns,
        'Importance': perm.importances_mean,
        'Std'       : perm.importances_std
    }).sort_values('Importance', ascending=False)

    print("\n  Top 10 Faktor Risiko Kredit:")
    print(feat_imp.head(10).to_string(index=False))
    return feat_imp


# 12. VISUALISASI
def plot_all(df, y_val, y_test, y_val_proba_base, y_val_proba_cal,
             y_test_proba, fpr_t, tpr_t, auc_final, ks_final,
             df_thresh, tn, fp, fn, tp, feat_imp, threshold,
             output_dir='.'):
    """Generate semua plot dan simpan sebagai PNG."""
    print()
    print("=" * 65)
    print("  STEP 10: MEMBUAT VISUALISASI")
    print("=" * 65)

    # ── Plot 1: Distribusi Target & Default per Grade ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df[TARGET].value_counts()
    axes[0].pie(counts, labels=['Nasabah Baik (0)', 'Nasabah Buruk (1)'],
                autopct='%1.1f%%', colors=['#2196F3', '#F44336'],
                startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[0].set_title('Proporsi Target (Default vs Tidak)', fontweight='bold')

    grade_default = df.groupby('grade')[TARGET].mean().sort_index()
    colors_grade  = ['#4CAF50','#8BC34A','#FFC107','#FF9800','#F44336','#B71C1C','#880E4F']
    axes[1].bar(grade_default.index, grade_default.values * 100, color=colors_grade)
    axes[1].set_title('Default Rate per Grade Pinjaman (%)', fontweight='bold')
    axes[1].set_xlabel('Grade (A=Terbaik, G=Terburuk)')
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(grade_default.values):
        axes[1].text(i, v * 100 + 0.3, f'{v:.1%}', ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/plot_1_distribusi_target.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plot_1_distribusi_target.png")

    # ── Plot 2: ROC Curve & Calibration Curve ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fpr_t, tpr_t, color='#1565C0', lw=2,
                 label=f'Model (AUC = {auc_final:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    axes[0].fill_between(fpr_t, tpr_t, alpha=0.1, color='#1565C0')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (Test Set)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    frac_pos_base, mean_pred_base = calibration_curve(y_val, y_val_proba_base, n_bins=10)
    frac_pos_cal,  mean_pred_cal  = calibration_curve(y_val, y_val_proba_cal,  n_bins=10)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal')
    axes[1].plot(mean_pred_base, frac_pos_base, 'rs-', label='Sebelum Kalibrasi')
    axes[1].plot(mean_pred_cal,  frac_pos_cal,  'g^-', label='Setelah Kalibrasi (Platt)')
    axes[1].set_xlabel('Probabilitas Prediksi')
    axes[1].set_ylabel('Fraksi Aktual Default')
    axes[1].set_title('Calibration Curve', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/plot_2_roc_calibration.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plot_2_roc_calibration.png")

    # ── Plot 3: Analisis Threshold 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_thresh['Threshold'], df_thresh['Recall (%)'],    'ro-', label='Recall',     lw=2)
    axes[0].plot(df_thresh['Threshold'], df_thresh['Precision (%)'], 'bs-', label='Precision',  lw=2)
    axes[0].plot(df_thresh['Threshold'], df_thresh['F1 Score'] * 100,'g^--',label='F1 Score',   lw=1.5)
    axes[0].axvline(x=threshold, color='purple', linestyle='--', alpha=0.7, label=f'Threshold={threshold}')
    axes[0].set_xlabel('Decision Threshold')
    axes[0].set_ylabel('Nilai (%)')
    axes[0].set_title('Trade-off Precision vs Recall', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_thresh['Threshold'], df_thresh['Jumlah Ditolak'],         'ko-', label='Total Ditolak', lw=2)
    axes[1].plot(df_thresh['Threshold'], df_thresh['Nasabah Buruk Tertangkap'],'rs-', label='Nasabah Buruk Tertangkap', lw=2)
    axes[1].plot(df_thresh['Threshold'], df_thresh['Nasabah Baik Salah Tolak'],'b^--',label='Nasabah Baik Salah Tolak', lw=1.5)
    axes[1].axvline(x=threshold, color='purple', linestyle='--', alpha=0.7, label=f'Threshold={threshold}')
    axes[1].set_xlabel('Decision Threshold')
    axes[1].set_ylabel('Jumlah Nasabah')
    axes[1].set_title('Dampak Threshold terhadap Volume Penolakan', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Analisis Business Threshold', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/plot_3_threshold_analysis.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plot_3_threshold_analysis.png")

    #  Plot 4: Confusion Matrix 
    fig, ax = plt.subplots(figsize=(7, 5))
    cm_labels = np.array([
        [f'TN\n{tn:,}\nNasabah Baik\nBenar Disetujui', f'FP\n{fp:,}\nNasabah Baik\nSalah Ditolak'],
        [f'FN\n{fn:,}\nNasabah Buruk\nLolos Approval',  f'TP\n{tp:,}\nNasabah Buruk\nBerhasil Ditolak']
    ])
    cm_values = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm_values, interpolation='nearest', cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Prediksi: Baik (0)', 'Prediksi: Buruk (1)'], fontsize=10)
    ax.set_yticklabels(['Aktual: Baik (0)', 'Aktual: Buruk (1)'], fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_labels[i, j], ha='center', va='center', fontsize=9,
                    color='white' if cm_values[i, j] > cm_values.max() / 2 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Confusion Matrix (Threshold = {threshold})', fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/plot_4_confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plot_4_confusion_matrix.png")

    #  Plot 5: Feature Importance 
    top15 = feat_imp.head(15).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
    ax.barh(top15['Fitur'], top15['Importance'], xerr=top15['Std'],
            color=colors_fi[::-1], capsize=3, error_kw={'linewidth': 1})
    ax.set_xlabel('Penurunan ROC-AUC saat Fitur Diacak')
    ax.set_title('Top 15 Faktor Risiko Kredit (Permutation Importance)', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/plot_5_feature_importance.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: plot_5_feature_importance.png")

    print("\n  ✅ Semua visualisasi berhasil disimpan.")


# 13. RINGKASAN FINAL
def print_summary(auc_final, ks_final, X_train, X_test, feat_imp, threshold):
    """Cetak ringkasan akhir model."""
    print()
    print("=" * 65)
    print("  RINGKASAN AKHIR MODEL CREDIT RISK")
    print("=" * 65)
    print()
    print(f"  Algoritma        : HistGradientBoostingClassifier")
    print(f"  Kalibrasi        : Platt Scaling (FrozenEstimator)")
    print(f"  ROC-AUC (Test)   : {auc_final:.4f}")
    print(f"  KS Statistic     : {ks_final:.2%}")
    print(f"  Threshold Bisnis : {threshold}")
    print()
    print(f"  Jumlah Fitur     : {X_train.shape[1]}")
    print(f"  Data Training    : {len(X_train):,} baris")
    print(f"  Data Test        : {len(X_test):,} baris")
    print()
    print("  Top 3 Faktor Risiko Kredit (Permutation Importance):")
    for _, row in feat_imp.head(3).iterrows():
        print(f"    {row['Fitur']}: AUC drop {row['Importance']:.4f} ± {row['Std']:.4f}")
    print()
    print("  ✅ Model siap digunakan sebagai decision support tool")
    print("  ⚠️  Threshold final sebaiknya dikalibrasi bersama tim bisnis")
    print("      berdasarkan analisis biaya NPL vs opportunity cost")
    print("=" * 65)


# MAIN — Entry Point
if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  CREDIT RISK SCORING MODEL")
    print("  HistGradientBoosting + Platt Scaling")
    print("  Project: ID/X Partners Internship")
    print("=" * 65)

    # Pipeline utama
    df            = load_and_filter_data(DATA_PATH)
    df_model      = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_model)
    base_model    = train_model(X_train, y_train)
    cal_model     = calibrate_model(base_model, X_val, y_val)

    (y_val_proba_base, y_val_proba_cal, y_test_proba,
     fpr_t, tpr_t, auc_final, ks_final) = evaluate_model(
        base_model, cal_model, X_val, X_test, y_val, y_test)

    df_thresh = threshold_analysis(y_test, y_test_proba)
    tn, fp, fn, tp = final_evaluation(
        y_test, y_test_proba, auc_final, ks_final, threshold=BEST_THRESHOLD)
    feat_imp  = compute_feature_importance(base_model, X_val, y_val)

    plot_all(df, y_val, y_test, y_val_proba_base, y_val_proba_cal,
             y_test_proba, fpr_t, tpr_t, auc_final, ks_final,
             df_thresh, tn, fp, fn, tp, feat_imp,
             threshold=BEST_THRESHOLD, output_dir=OUTPUT_DIR)

    print_summary(auc_final, ks_final, X_train, X_test, feat_imp, BEST_THRESHOLD)
