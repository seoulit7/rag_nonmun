"""Ablation Study Performance Visualization Dashboard.

PDF 테스트 결과서 기반 7개 시각화:
  1. RAGAS 메트릭 비교 (조건별 F / AR / CP)
  2. 환각 감소 효과
  3. 에스컬레이션 패턴 분석 (Tier 분포)
  4. 수준 분류기 성능
  5. 자가 교정 루프 수렴
  6. 구성 요소 기여도 (Δk)
  7. 계산 효율성 (처리 시간)
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

logger = logging.getLogger(__name__)

# ── 상수 ───────────────────────────────────────────────────────────────────────
_COND_ORDER  = ["A", "B", "C", "D", "E"]
_COND_LABELS = {
    "A": "A — Full System",
    "B": "B — No Self-Correction",
    "C": "C — No Multi-Tier",
    "D": "D — No Level Classifier",
    "E": "E — Baseline",
}
_COND_SHORT = {
    "A": "Full\nSystem",
    "B": "No\nSelf-Corr",
    "C": "No\nMulti-Tier",
    "D": "No\nLevel-Cls",
    "E": "Baseline",
}
_COND_COLORS = {
    "A": "#2ecc71",
    "B": "#e67e22",
    "C": "#e74c3c",
    "D": "#9b59b6",
    "E": "#95a5a6",
}
_THRESHOLD    = 0.80
_AR_THRESHOLD = 0.80

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_data() -> pd.DataFrame:
    import psycopg2
    import config.settings as s
    sql = """
        SELECT
            request_id, ablation_condition, user_level, query_level_label,
            final_tier, loop_number, is_final, self_correction_count,
            ragas_f, ragas_ar, ragas_cp, q_total,
            hallucination_detected, hallucination_count,
            is_escalated, is_fallback,
            execution_time_ms, fk_grade, created_at
        FROM public.rag_audit_log
        WHERE ablation_condition IS NOT NULL
        ORDER BY created_at, loop_number
    """
    try:
        conn = psycopg2.connect(s.SUPABASE_DB_URL)
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        logger.error("Data load failed: %s", e)
        return pd.DataFrame()

    for col in ["ragas_f", "ragas_ar", "ragas_cp", "q_total", "fk_grade"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["loop_number"]            = df["loop_number"].fillna(1).astype(int)
    df["self_correction_count"]  = df["self_correction_count"].fillna(0).astype(int)
    df["final_tier"]             = df["final_tier"].fillna(0).astype(int)
    df["is_final"]               = df["is_final"].fillna(False).astype(bool)
    df["is_escalated"]           = df["is_escalated"].fillna(False).astype(bool)
    df["is_fallback"]            = df["is_fallback"].fillna(False).astype(bool)
    df["hallucination_detected"] = df["hallucination_detected"].fillna(False).astype(bool)
    df["hallucination_count"]    = df["hallucination_count"].fillna(0).astype(int)
    return df


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────
def _fig(w=10, h=5.5):
    sns.set_theme(style="whitegrid", font_scale=1.05)
    return plt.subplots(figsize=(w, h), dpi=150)

def _savebuf(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def _final(df):
    return df[df["is_final"]].copy()

def _present_conds(df):
    fin = _final(df)
    return [c for c in _COND_ORDER if c in fin["ablation_condition"].values]


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAGAS 메트릭 비교
# ══════════════════════════════════════════════════════════════════════════════
def _plot_ragas_comparison(df):
    fin   = _final(df)
    conds = _present_conds(df)
    metrics = [
        ("ragas_f",  "Faithfulness (F)",      "#2980b9"),
        ("ragas_ar", "Answer Relevance (AR)", "#e67e22"),
        ("ragas_cp", "Context Precision (CP)","#27ae60"),
    ]
    x     = np.arange(len(conds))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = _fig(11, 6)
    for i, (col, lbl, color) in enumerate(metrics):
        means, errs = [], []
        for c in conds:
            vals = fin[fin["ablation_condition"] == c][col].dropna().values
            means.append(vals.mean() if len(vals) else 0)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) * 1.96 if len(vals) > 1 else 0)
        bars = ax.bar(x + offsets[i], means, width, label=lbl,
                      color=color, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(x + offsets[i], means, yerr=errs, fmt="none",
                    color="#2c3e50", capsize=4, linewidth=1.3, zorder=4)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.axhline(_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5,
               alpha=0.75, label=f"Threshold ({_THRESHOLD})", zorder=2)
    ax.set_title("Table 1. RAGAS Metric Comparison — Ablation 5 Conditions (Mean ± 95% CI)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([_COND_SHORT[c] for c in conds], fontsize=10)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_ylim(0.45, 1.15)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_ragas(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = fin[fin["ablation_condition"] == c]
        def ms(col):
            v = sub[col].dropna()
            return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else "—"
        rows.append({
            "조건": _COND_LABELS.get(c, c),
            "F (충실도)": ms("ragas_f"),
            "AR (답변관련성)": ms("ragas_ar"),
            "CP (컨텍스트정밀도)": ms("ragas_cp"),
            "Q_total": ms("q_total"),
            "건수": len(sub),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 환각 감소 효과
# ══════════════════════════════════════════════════════════════════════════════
def _plot_hallucination(df):
    fin   = _final(df)
    conds = _present_conds(df)
    rates = []
    for c in conds:
        sub = fin[fin["ablation_condition"] == c]
        rates.append(sub["hallucination_detected"].mean() * 100 if len(sub) else 0)

    baseline = rates[conds.index("E")] if "E" in conds else 0

    fig, ax = _fig(9, 5.5)
    colors = [_COND_COLORS.get(c, "#95a5a6") for c in conds]
    bars = ax.bar([_COND_SHORT[c] for c in conds], rates,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, rate, c in zip(bars, rates, conds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        if c != "E" and baseline > 0:
            red = (baseline - rate) / baseline * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    rate / 2,
                    f"↓{red:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_title("Table 2. Hallucination Rate Comparison — Ablation 5 Conditions",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax.set_ylim(0, (max(rates) if rates else 50) * 1.4)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_hallucination(df):
    fin = _final(df)
    baseline_sub = fin[fin["ablation_condition"] == "E"]
    baseline = baseline_sub["hallucination_detected"].mean() * 100 if len(baseline_sub) else 0
    rows = []
    for c in _COND_ORDER:
        sub = fin[fin["ablation_condition"] == c]
        rate = sub["hallucination_detected"].mean() * 100 if len(sub) else 0
        red  = (baseline - rate) / baseline * 100 if baseline > 0 and c != "E" else None
        rows.append({
            "조건": _COND_LABELS.get(c, c),
            "환각 비율": f"{rate:.1f}%",
            "Baseline 대비 감소율": f"{red:.1f}%" if red is not None else "—",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 에스컬레이션 패턴 (Tier 분포 — 조건 A)
# ══════════════════════════════════════════════════════════════════════════════
def _plot_tier_distribution(df):
    fin = _final(df)
    sub = fin[fin["ablation_condition"] == "A"]
    if sub.empty:
        return None

    tier_counts = sub["final_tier"].value_counts().sort_index()
    tier_labels = {0: "Tier 0\n(VectorDB)", 1: "Tier 1\n(LLM)", 2: "Tier 2\n(Web)"}
    labels  = [tier_labels.get(t, f"Tier {t}") for t in tier_counts.index]
    colors  = ["#27ae60", "#f39c12", "#e74c3c"][:len(tier_counts)]
    total   = tier_counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)

    wedges, texts, autotexts = ax1.pie(
        tier_counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax1.set_title("Tier Distribution (Condition A)", fontsize=12, fontweight="bold", pad=12)

    bars = ax2.bar(labels, tier_counts.values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    for bar, cnt in zip(bars, tier_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{cnt}건\n({cnt/total*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Query Count by Tier", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_ylim(0, tier_counts.max() * 1.35)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle("Table 3. Tier Distribution — Escalation Pattern Analysis (Condition A)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _savebuf(fig)


def _table_tier_distribution(df):
    fin = _final(df)
    sub = fin[fin["ablation_condition"] == "A"]
    if sub.empty:
        return pd.DataFrame()

    tier_names = {0: "Tier 0 (MSD 매뉴얼)", 1: "Tier 1 (LLM 지식)", 2: "Tier 2 (웹 검색)"}
    pro   = sub[sub["user_level"] == "Professional"]
    cons  = sub[sub["user_level"] == "Consumer"]
    total = sub

    n_pro  = max(len(pro),   1)
    n_cons = max(len(cons),  1)
    n_tot  = max(len(total), 1)

    rows = []
    for tier in sorted(sub["final_tier"].unique()):
        cp = len(pro[pro["final_tier"]   == tier])
        cc = len(cons[cons["final_tier"] == tier])
        ct = len(total[total["final_tier"] == tier])
        rows.append({
            "티어":        tier_names.get(tier, f"Tier {tier}"),
            "전문가 쿼리": f"{cp} ({cp/n_pro*100:.0f}%)",
            "일반인 쿼리": f"{cc} ({cc/n_cons*100:.0f}%)",
            "전체":        f"{ct} ({ct/n_tot*100:.0f}%)",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 수준 분류기 성능
# ══════════════════════════════════════════════════════════════════════════════
def _compute_classifier_metrics(df):
    sub = df[
        df["is_final"] &
        df["ablation_condition"].isin(["A", "B", "C"]) &
        df["query_level_label"].notna() &
        df["user_level"].notna()
    ].drop_duplicates(subset=["request_id"]).copy()
    if sub.empty:
        return None

    label_map = {"P": "Professional", "C": "Consumer"}
    sub["expected"]  = sub["query_level_label"].map(label_map)
    sub["predicted"] = sub["user_level"]
    valid = sub[
        sub["expected"].isin(["Professional", "Consumer"]) &
        sub["predicted"].isin(["Professional", "Consumer"])
    ]
    if len(valid) == 0:
        return None

    y_true = (valid["expected"]  == "Professional").astype(int)
    y_pred = (valid["predicted"] == "Professional").astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": len(valid)}


def _plot_classifier(m):
    keys    = ["Accuracy", "Precision", "Recall", "F1"]
    values  = [m[k] for k in keys]
    colors  = ["#2980b9", "#27ae60", "#e67e22", "#9b59b6"]

    fig, ax = _fig(8, 5)
    bars = ax.bar(keys, values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.2, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    ax.axhline(0.9, color="#e74c3c", linestyle="--", linewidth=1.5,
               alpha=0.75, label="90% 기준선", zorder=2)
    ax.set_title("Table 4. Level Classifier Performance (Conditions A / B / C)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.6, 1.12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4-b. FK Grade — 수준 분류기 간접 검증
# ══════════════════════════════════════════════════════════════════════════════
_FK_CONSUMER_MAX     = 9.0   # Consumer 목표: Grade ≤ 9 (고등학생 수준 이하)
_FK_PROFESSIONAL_MIN = 12.0  # Professional 기준: Grade ≥ 12 (대학 수준)


def _plot_fk_grade(df):
    fin = _final(df)
    sub = fin[fin["fk_grade"].notna() & fin["user_level"].isin(["Professional", "Consumer"])]
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # ── 왼쪽: 박스플롯 (user_level별) ────────────────────────────────────────
    ax1 = axes[0]
    palette = {"Consumer": "#3498db", "Professional": "#e74c3c"}
    sns.boxplot(
        data=sub, x="user_level", y="fk_grade",
        palette=palette, width=0.45, linewidth=1.5,
        order=["Consumer", "Professional"], ax=ax1,
    )
    ax1.axhline(_FK_CONSUMER_MAX,     color="#3498db", linestyle="--",
                linewidth=1.4, alpha=0.8, label=f"Consumer target (≤{_FK_CONSUMER_MAX})")
    ax1.axhline(_FK_PROFESSIONAL_MIN, color="#e74c3c", linestyle="--",
                linewidth=1.4, alpha=0.8, label=f"Professional target (≥{_FK_PROFESSIONAL_MIN})")
    for level, color in palette.items():
        mean_val = sub[sub["user_level"] == level]["fk_grade"].mean()
        if not pd.isna(mean_val):
            ax1.text(
                0 if level == "Consumer" else 1,
                mean_val + 0.3,
                f"mean={mean_val:.2f}",
                ha="center", fontsize=10, fontweight="bold", color=color,
            )
    ax1.set_title("FK Grade by User Level", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("User Level", fontsize=11)
    ax1.set_ylabel("FK Grade", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")

    # ── 오른쪽: 조건별 평균 막대 ─────────────────────────────────────────────
    ax2 = axes[1]
    conds = _present_conds(df)
    levels = ["Consumer", "Professional"]
    x     = np.arange(len(conds))
    width = 0.32
    for i, (level, color) in enumerate(palette.items()):
        means = []
        for c in conds:
            vals = fin[
                (fin["ablation_condition"] == c) &
                (fin["user_level"] == level)
            ]["fk_grade"].dropna().values
            means.append(vals.mean() if len(vals) else 0)
        offset = -width / 2 if i == 0 else width / 2
        bars = ax2.bar(x + offset, means, width, label=level,
                       color=color, alpha=0.8, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, means):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.15,
                         f"{val:.1f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")

    ax2.axhline(_FK_CONSUMER_MAX,     color="#3498db", linestyle="--",
                linewidth=1.3, alpha=0.7)
    ax2.axhline(_FK_PROFESSIONAL_MIN, color="#e74c3c", linestyle="--",
                linewidth=1.3, alpha=0.7)
    ax2.set_title("Mean FK Grade by Condition & Level", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([_COND_SHORT[c] for c in conds], fontsize=10)
    ax2.set_ylabel("Mean FK Grade", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle("FK Grade — Level Classifier Indirect Validation (is_final=True only)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _savebuf(fig)


def _table_fk_grade(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = fin[fin["ablation_condition"] == c]
        for level in ["Consumer", "Professional"]:
            vals = sub[sub["user_level"] == level]["fk_grade"].dropna()
            target = f"≤ {_FK_CONSUMER_MAX}" if level == "Consumer" else f"≥ {_FK_PROFESSIONAL_MIN}"
            if len(vals) == 0:
                rows.append({"Condition": _COND_LABELS.get(c, c), "User Level": level,
                             "Mean FK Grade": "—", "Std Dev": "—", "Target": target,
                             "Within Target": "—", "n": 0})
                continue
            mean_v = vals.mean()
            within = (
                (vals <= _FK_CONSUMER_MAX).mean() * 100 if level == "Consumer"
                else (vals >= _FK_PROFESSIONAL_MIN).mean() * 100
            )
            rows.append({
                "Condition":    _COND_LABELS.get(c, c),
                "User Level":   level,
                "Mean FK Grade": f"{mean_v:.2f}",
                "Std Dev":      f"{vals.std(ddof=1):.2f}" if len(vals) > 1 else "—",
                "Target":       target,
                "Within Target": f"{within:.1f}%",
                "n":            len(vals),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 자가 교정 루프 수렴
# ══════════════════════════════════════════════════════════════════════════════
def _plot_loop_convergence(df):
    sub = df[df["ablation_condition"] == "A"].copy()
    if sub.empty:
        return None

    grp = (
        sub[sub["q_total"].notna()]
        .groupby("loop_number")["q_total"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if grp.empty:
        return None
    grp["ci"] = 1.96 * grp["std"] / np.sqrt(grp["count"])

    fig, ax = _fig(8, 5.5)
    ax.plot(grp["loop_number"], grp["mean"],
            marker="o", color="#2980b9", linewidth=2.5, markersize=9,
            zorder=3, label="Mean Q_total")
    ax.fill_between(grp["loop_number"],
                    grp["mean"] - grp["ci"], grp["mean"] + grp["ci"],
                    alpha=0.18, color="#2980b9", label="95% CI")
    for _, row in grp.iterrows():
        ax.annotate(f"{row['mean']:.3f}",
                    (row["loop_number"], row["mean"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold", color="#2c3e50")

    # convergence ratio (Q_total >= threshold)
    for _, row in grp.iterrows():
        loop_sub = sub[sub["loop_number"] == row["loop_number"]]["q_total"].dropna()
        if len(loop_sub):
            conv = (loop_sub >= _THRESHOLD).mean() * 100
            ax.annotate(f"Conv. {conv:.0f}%",
                        (row["loop_number"], row["mean"] - grp["ci"].max() - 0.04),
                        ha="center", fontsize=8.5, color="#7f8c8d")

    ax.axhline(_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"τ_Q = {_THRESHOLD}")
    ax.set_title("Table 5. Self-Correction Loop Convergence — Q_total (Condition A)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Loop Number (Evaluation Count)", fontsize=11)
    ax.set_ylabel("Mean Q_total Score", fontsize=11)
    ax.set_xticks(grp["loop_number"].tolist())
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_loop(df):
    sub = df[df["ablation_condition"] == "A"].copy()
    rows = []
    for lc in sorted(sub["loop_number"].unique()):
        vals = sub[sub["loop_number"] == lc]["q_total"].dropna().values
        if len(vals) == 0:
            continue
        conv = (vals >= _THRESHOLD).mean() * 100
        rows.append({
            "Loop": int(lc),
            "Mean Q_total": round(float(vals.mean()), 3),
            "Std Dev": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0, 3),
            "Conv. Rate (τ≥0.8)": f"{conv:.1f}%",
            "n": len(vals),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 6. 구성 요소 기여도 (Δk)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_contribution(df):
    fin = _final(df)
    metrics   = ["ragas_f", "ragas_ar", "ragas_cp", "q_total"]
    m_labels  = ["ΔF", "ΔAR", "ΔCP", "ΔQ"]
    ablations = [("B", "Self-Correction (SC)"), ("C", "Multi-Tier (MT)"), ("D", "Level-Classifier (LC)")]

    a_vals = {}
    for m in metrics:
        v = fin[fin["ablation_condition"] == "A"][m].dropna().values
        a_vals[m] = v.mean() if len(v) else None

    rows = []
    for cond, name in ablations:
        row = {"Component": name, "Ablated Cond.": cond}
        for m, lbl in zip(metrics, m_labels):
            v = fin[fin["ablation_condition"] == cond][m].dropna().values
            b = v.mean() if len(v) else None
            row[lbl] = round(a_vals[m] - b, 4) if (a_vals[m] is not None and b is not None) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_contribution(delta_df):
    if delta_df.empty:
        return None
    m_labels = ["ΔF", "ΔAR", "ΔCP", "ΔQ"]
    colors   = ["#2980b9", "#e67e22", "#27ae60", "#9b59b6"]
    names    = delta_df["Component"].tolist()
    x        = np.arange(len(names))
    width    = 0.18
    offsets  = np.linspace(-1.5 * width, 1.5 * width, 4)

    fig, ax = _fig(10, 5.5)
    for i, (lbl, color, offset) in enumerate(zip(m_labels, colors, offsets)):
        vals = delta_df[lbl].fillna(0).tolist()
        bars = ax.bar(x + offset, vals, width, label=lbl,
                      color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, delta_df[lbl].tolist()):
            if val is not None and not pd.isna(val):
                label_text = f"+{val:.3f}" if val >= 0 else f"{val:.3f}"
                if val >= 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            label_text, ha="center", va="bottom",
                            fontsize=8, fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() - 0.003,
                            label_text, ha="center", va="top",
                            fontsize=8, fontweight="bold", color="#c0392b")

    ax.set_title("Table 6. Ablation Component Contribution  Δk = Full(A) − Ablated",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Δ Metric  (higher = more contribution)", fontsize=11)
    ax.axhline(0, color="#2c3e50", linewidth=0.8)
    y_min = min(delta_df[m_labels].min().min() - 0.03, -0.02)
    y_max = max(delta_df[m_labels].max().max() + 0.03, 0.22)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 계산 효율성 (처리 시간)
# ══════════════════════════════════════════════════════════════════════════════
def _plot_efficiency(df):
    fin   = _final(df)
    conds = _present_conds(df)
    times, errs = [], []
    for c in conds:
        vals = fin[fin["ablation_condition"] == c]["execution_time_ms"].dropna().values / 1000
        times.append(vals.mean() if len(vals) else 0)
        errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) * 1.96 if len(vals) > 1 else 0)

    fig, ax = _fig(9, 5.5)
    colors = [_COND_COLORS.get(c, "#95a5a6") for c in conds]
    bars = ax.bar([_COND_SHORT[c] for c in conds], times,
                  yerr=errs, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.2, capsize=5, zorder=3,
                  error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"})
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Table 7. Processing Time per Query — Ablation 5 Conditions",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Mean Processing Time (sec, 95% CI)", fontsize=11)
    ax.set_ylim(0, (max(times) if times else 10) * 1.4)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_efficiency(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = fin[fin["ablation_condition"] == c]
        t   = sub["execution_time_ms"].dropna() / 1000
        rows.append({
            "조건": _COND_LABELS.get(c, c),
            "평균 처리 시간 (초)": f"{t.mean():.1f} ± {t.std(ddof=1):.1f}" if len(t) > 1 else "—",
            "건수": len(sub),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 요약 카드
# ══════════════════════════════════════════════════════════════════════════════
def _render_summary_cards(df):
    fin = _final(df)
    cols = st.columns(5)
    names = ["Full", "No SC", "No MT", "No LC", "Baseline"]
    for col, c, name in zip(cols, _COND_ORDER, names):
        sub  = fin[fin["ablation_condition"] == c]
        avg_f = sub["ragas_f"].mean()
        hallu = sub["hallucination_detected"].mean() * 100 if len(sub) else 0
        col.metric(
            label=f"[{c}] {name}",
            value=f"F = {avg_f:.3f}" if not pd.isna(avg_f) else "—",
            delta=f"환각 {hallu:.1f}%  |  {len(sub)}건",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 메인 렌더
# ══════════════════════════════════════════════════════════════════════════════
def render_performance_viz() -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    st.title("Ablation Study — Performance Visualization")
    st.caption("논문 Ablation Study 결과 시각화 — Supabase `rag_audit_log` 기반 (조건 A~E)")

    with st.spinner("데이터 로딩 중..."):
        df = _load_data()

    if df.empty:
        st.error("데이터를 불러오지 못했습니다. Supabase 연결을 확인하세요.")
        return

    if st.button("새로고침", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    # ── 요약 카드 ──────────────────────────────────────────────────────────────
    st.markdown("### 조건별 요약")
    _render_summary_cards(df)
    st.markdown("---")

    # ── 1. RAGAS 메트릭 비교 ──────────────────────────────────────────────────
    st.markdown("### 1. RAGAS 메트릭 비교")
    st.caption("조건별 Faithfulness / Answer Relevance / Context Precision 평균 ± 95% CI")
    buf = _plot_ragas_comparison(df)
    if buf:
        st.image(buf, use_container_width=True)
    st.dataframe(_table_ragas(df), hide_index=True, use_container_width=True)
    st.markdown("---")

    # ── 2. 환각 감소 효과 ─────────────────────────────────────────────────────
    st.markdown("### 2. 환각 감소 효과")
    st.caption("조건별 환각 감지 비율 및 Baseline(E) 대비 감소율")
    buf = _plot_hallucination(df)
    if buf:
        st.image(buf, use_container_width=True)
    st.dataframe(_table_hallucination(df), hide_index=True, use_container_width=True)
    st.markdown("---")

    # ── 3. 에스컬레이션 패턴 ──────────────────────────────────────────────────
    st.markdown("### 3. 에스컬레이션 패턴 분석 (Tier 분포 — 조건 A)")
    st.caption("Full System 조건에서 Tier 0 → 1 → 2 분포")
    buf = _plot_tier_distribution(df)
    if buf:
        st.image(buf, use_container_width=True)
    else:
        st.info("조건 A 데이터가 없습니다.")
    tier_tbl = _table_tier_distribution(df)
    if not tier_tbl.empty:
        st.dataframe(tier_tbl, hide_index=True, use_container_width=True)
    st.markdown("---")

    # ── 4. 수준 분류기 성능 ───────────────────────────────────────────────────
    st.markdown("### 4. 수준 분류기 성능 (조건 A / B / C)")
    st.caption("query_level_label (P/C) vs user_level 비교 — Accuracy / Precision / Recall / F1")
    m = _compute_classifier_metrics(df)
    if m:
        buf = _plot_classifier(m)
        if buf:
            st.image(buf, use_container_width=True)
        st.dataframe(pd.DataFrame([{
            "정확도": f"{m['Accuracy']*100:.1f}%",
            "정밀도": f"{m['Precision']*100:.1f}%",
            "재현율": f"{m['Recall']*100:.1f}%",
            "F1": f"{m['F1']*100:.1f}%",
            "TP": m["TP"], "TN": m["TN"],
            "FP": m["FP"], "FN": m["FN"],
            "전체 n": m["n"],
        }]), hide_index=True, use_container_width=True)
    else:
        st.info("분류기 평가 데이터가 없습니다. (query_level_label 컬럼 필요)")
    st.markdown("---")

    # ── 4-b. FK Grade ─────────────────────────────────────────────────────────
    st.markdown("### 4-b. FK Grade — Level Classifier Indirect Validation")
    st.caption(
        "FK Grade Level (English original, before Korean translation)  |  "
        "Consumer target ≤ 9  |  Professional target ≥ 12  |  is_final=True only"
    )
    buf = _plot_fk_grade(df)
    if buf:
        st.image(buf, use_container_width=True)
    else:
        st.info("fk_grade 데이터가 없습니다. (새 질의부터 측정 시작)")
    fk_tbl = _table_fk_grade(df)
    if not fk_tbl.empty:
        st.dataframe(fk_tbl, hide_index=True, use_container_width=True)
    st.markdown("---")

    # ── 5. 자가 교정 루프 수렴 ────────────────────────────────────────────────
    st.markdown("### 5. Self-Correction Loop Convergence (Condition A)")
    st.caption("Mean Q_total and convergence rate per loop iteration (τ_Q = 0.8)")
    buf = _plot_loop_convergence(df)
    if buf:
        st.image(buf, use_container_width=True)
    loop_tbl = _table_loop(df)
    if not loop_tbl.empty:
        st.dataframe(loop_tbl, hide_index=True, use_container_width=True)
    else:
        st.info("조건 A 데이터가 없습니다.")
    st.markdown("---")

    # ── 6. 구성 요소 기여도 ───────────────────────────────────────────────────
    st.markdown("### 6. Component Contribution  Δk = Full(A) − Ablated")
    st.caption("Performance drop when each component (Self-Correction / Multi-Tier / Level-Classifier) is removed")
    delta_df = _compute_contribution(df)
    buf = _plot_contribution(delta_df)
    if buf:
        st.image(buf, use_container_width=True)
    if not delta_df.empty:
        display_df = delta_df.copy()
        for col in ["ΔF", "ΔAR", "ΔCP", "ΔQ"]:
            display_df[col] = display_df[col].apply(
                lambda v: (f"+{v:.3f}" if v >= 0 else f"{v:.3f}") if v is not None and not pd.isna(v) else "—"
            )
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.markdown("---")

    # ── 7. 계산 효율성 ────────────────────────────────────────────────────────
    st.markdown("### 7. 계산 효율성 (처리 시간)")
    st.caption("조건별 쿼리당 평균 처리 시간 (초, 95% CI)")
    buf = _plot_efficiency(df)
    if buf:
        st.image(buf, use_container_width=True)
    st.dataframe(_table_efficiency(df), hide_index=True, use_container_width=True)
