"""RAG Performance Visualization Dashboard.

Four key hypothesis visualizations for PhD dissertation:
  1. Self-Correction Effect (loop_count vs Faithfulness)
  2. Intelligent Escalation Validity (tier_id vs Answer Relevance Box Plot)
  3. Retrieval Precision vs Answer Relevance Correlation (CP vs AR Scatter)
  4. User-Level Personalization (user_level Grouped Bar)
"""
from __future__ import annotations

import warnings
import logging

import matplotlib
matplotlib.use("Agg")  # Prevent GUI backend conflict in Streamlit

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

logger = logging.getLogger(__name__)

# ── Font Setup ────────────────────────────────────────────────────────────────
def _setup_font() -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_data() -> pd.DataFrame:
    """Load rag_audit_log into a DataFrame (5-min cache)."""
    import psycopg2
    import config.settings as s

    sql = """
        SELECT
            request_id, user_level, tier_id, loop_count,
            ragas_f, ragas_ar, ragas_cp,
            is_escalated, is_fallback,
            retrieved_doc_count, execution_time_ms, created_at
        FROM public.rag_audit_log
        ORDER BY created_at
    """
    try:
        conn = psycopg2.connect(s.SUPABASE_DB_URL)
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        logger.error("Data load failed: %s", e)
        return pd.DataFrame()

    for col in ["ragas_f", "ragas_ar", "ragas_cp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["loop_count"]   = df["loop_count"].fillna(0).astype(int)
    df["tier_id"]      = df["tier_id"].fillna(0).astype(int)
    df["is_escalated"] = df["is_escalated"].fillna(False).astype(bool)
    df["is_fallback"]  = df["is_fallback"].fillna(False).astype(bool)
    return df


# ── Common Helpers ────────────────────────────────────────────────────────────
_THRESHOLD    = 0.80
_LEVEL_COLORS = {"Professional": "#2980b9", "Consumer": "#e67e22"}


def _fig(w: float = 8, h: float = 5) -> tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    return fig, ax


def _save_buf(fig: plt.Figure):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Hypothesis 1: Self-Correction Effect ─────────────────────────────────────
def _plot_self_correction(df: pd.DataFrame):
    """Line plot: loop_count vs mean Faithfulness with 95% CI."""
    subset = df[df["ragas_f"].notna()].copy()
    if subset.empty:
        return None

    grp = (
        subset.groupby("loop_count")["ragas_f"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["ci95"] = 1.96 * grp["std"] / np.sqrt(grp["count"])

    fig, ax = _fig(7, 5)

    ax.plot(grp["loop_count"], grp["mean"],
            marker="o", color="#2980b9", linewidth=2.5, markersize=9,
            zorder=3, label="Mean Faithfulness")
    ax.fill_between(
        grp["loop_count"],
        grp["mean"] - grp["ci95"],
        grp["mean"] + grp["ci95"],
        alpha=0.18, color="#2980b9", label="95% Confidence Interval"
    )
    for _, row in grp.iterrows():
        ax.annotate(
            f"{row['mean']:.3f}",
            (row["loop_count"], row["mean"]),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10, fontweight="bold", color="#2c3e50"
        )

    ax.axhline(_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"Threshold ({_THRESHOLD})")

    ax.set_title("Hypothesis 1: Faithfulness Improvement via Self-Correction Loop",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Self-Correction Loop Count", fontsize=11)
    ax.set_ylabel("Mean Faithfulness Score", fontsize=11)
    ax.set_xticks(grp["loop_count"].tolist())
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _save_buf(fig)


# ── Immediate Escalation Analysis ────────────────────────────────────────────
# Condition 1: AR < 0.3  → no relevant content in DB at all (AR alone)
# Condition 2: F < 0.3 AND CP < 0.2 → retrieval completely off-target (both)
_IMM_AR_THR = 0.3   # CRITICAL_AR_THRESHOLD
_IMM_F_THR  = 0.3   # CRITICAL_F_THRESHOLD
_IMM_CP_THR = 0.2   # CRITICAL_CP_THRESHOLD


def _plot_ar_escalation_zone(df: pd.DataFrame):
    """AR Escalation Zone: AR value distribution with escalation threshold line.

    AR is independent of F — shows how AR alone triggers immediate escalation.
    Uses strip plot + KDE to show full distribution against the 0.3 threshold.
    """
    subset = df[df["ragas_ar"].notna()].copy()
    if subset.empty:
        return None

    fig, ax = _fig(9, 5)

    ar_vals  = subset["ragas_ar"].values
    esc_vals = subset[subset["is_escalated"] & (subset["ragas_ar"] < _IMM_AR_THR)]["ragas_ar"].values
    norm_vals = subset[~(subset["is_escalated"] & (subset["ragas_ar"] < _IMM_AR_THR))]["ragas_ar"].values

    # KDE curve
    from scipy.stats import gaussian_kde
    xs = np.linspace(0, 1.05, 300)
    if len(ar_vals) > 2:
        kde = gaussian_kde(ar_vals, bw_method=0.25)
        ys  = kde(xs)
        ax.plot(xs, ys, color="#2980b9", linewidth=2.2, label="AR Distribution (KDE)", zorder=3)
        ax.fill_between(xs, ys, where=(xs < _IMM_AR_THR),
                        color="#e74c3c", alpha=0.18, label="Escalation Zone (AR < {})".format(_IMM_AR_THR))
        ax.fill_between(xs, ys, where=(xs >= _IMM_AR_THR),
                        color="#2980b9", alpha=0.10)

    # Strip of individual points on x-axis
    y_jitter = np.random.default_rng(42).uniform(-0.003, 0.003, len(norm_vals))
    ax.scatter(norm_vals, y_jitter, color="#2980b9", alpha=0.5, s=40,
               edgecolors="white", linewidths=0.4, zorder=4, label="Normal records")
    if len(esc_vals):
        y_esc = np.random.default_rng(42).uniform(-0.003, 0.003, len(esc_vals))
        ax.scatter(esc_vals, y_esc, marker="*", color="#e74c3c", s=220,
                   edgecolors="#922b21", linewidths=0.8, zorder=5,
                   label="Immediate Escalation — AR (★)")

    # Threshold lines
    ax.axvline(_IMM_AR_THR, color="#e74c3c", linestyle="--", linewidth=2.0,
               label=f"Escalation Threshold (AR = {_IMM_AR_THR})")
    ax.axvline(_THRESHOLD, color="#95a5a6", linestyle=":", linewidth=1.3, alpha=0.7,
               label=f"Success Threshold (AR = {_THRESHOLD})")

    # Mean annotation
    mean_ar = ar_vals.mean()
    ax.axvline(mean_ar, color="#27ae60", linestyle="-.", linewidth=1.5, alpha=0.8,
               label=f"Mean AR = {mean_ar:.3f}")

    ax.set_title("Immediate Escalation Zone (AR-Based): Answer Relevance Distribution",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Answer Relevance (AR)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0.0, 1.05)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_buf(fig)


def _plot_decision_zone(df: pd.DataFrame):
    """Viz 1: Decision Zone Scatter — CP vs F with Immediate Escalation Zone highlighted."""
    subset = df[df["ragas_f"].notna() & df["ragas_cp"].notna()].copy()
    if subset.empty:
        return None

    fig, ax = _fig(8, 6)

    # Immediate Escalation Zone shading
    ax.axhspan(0, _IMM_F_THR,  xmin=0, xmax=1, alpha=0.0)   # reset
    ax.fill_between([0, _IMM_CP_THR], [0, 0], [_IMM_F_THR, _IMM_F_THR],
                    color="#e74c3c", alpha=0.15, zorder=1)
    ax.text(0.01, _IMM_F_THR * 0.45,
            "Immediate\nEscalation Zone\n(CP<{:.1f} & F<{:.1f})".format(_IMM_CP_THR, _IMM_F_THR),
            fontsize=8.5, color="#c0392b", fontweight="bold", va="center")

    # Normal points
    normal = subset[~(subset["is_escalated"] & (subset["ragas_cp"] < _IMM_CP_THR) & (subset["ragas_f"] < _IMM_F_THR))]
    ax.scatter(normal["ragas_cp"], normal["ragas_f"],
               color="#2980b9", alpha=0.55, s=60, edgecolors="white",
               linewidths=0.6, label="Normal / Self-Corrected", zorder=3)

    # Immediate escalation points
    esc = subset[subset["is_escalated"] & (subset["ragas_cp"] < _IMM_CP_THR) & (subset["ragas_f"] < _IMM_F_THR)]
    if not esc.empty:
        ax.scatter(esc["ragas_cp"], esc["ragas_f"],
                   marker="*", color="#e74c3c", s=260, edgecolors="#922b21",
                   linewidths=0.8, label="Immediate Escalation (★)", zorder=5)

    # Threshold lines
    ax.axvline(_IMM_CP_THR, color="#e74c3c", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.axhline(_IMM_F_THR,  color="#e74c3c", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.axvline(_THRESHOLD,  color="#95a5a6", linestyle=":", linewidth=1.1, alpha=0.6)
    ax.axhline(_THRESHOLD,  color="#95a5a6", linestyle=":", linewidth=1.1, alpha=0.6)
    ax.text(_THRESHOLD + 0.01, 0.02, f"F={_THRESHOLD}", color="#95a5a6", fontsize=8)
    ax.text(0.01, _THRESHOLD + 0.01, f"CP={_THRESHOLD}", color="#95a5a6", fontsize=8)

    ax.set_title("Escalation Decision Zone: Context Precision vs Faithfulness",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Context Precision (CP)", fontsize=11)
    ax.set_ylabel("Faithfulness (F)", fontsize=11)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_buf(fig)


# ── Context Precision vs Answer Relevance Scatter ─────────────────────────────
def _plot_cp_ar_scatter(df: pd.DataFrame):
    """Scatter plot: Context Precision vs Answer Relevance with regression line."""
    subset = df[df["ragas_cp"].notna() & df["ragas_ar"].notna()].copy()
    if subset.empty:
        return None

    fig, ax = _fig(8, 6)

    for level, grp in subset.groupby("user_level"):
        color = _LEVEL_COLORS.get(level, "#7f8c8d")
        ax.scatter(grp["ragas_cp"], grp["ragas_ar"],
                   color=color, alpha=0.65, s=70,
                   edgecolors="white", linewidths=0.6,
                   label=level, zorder=3)

    x_all = subset["ragas_cp"].values
    y_all = subset["ragas_ar"].values
    if len(x_all) > 2:
        z  = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="#e74c3c", linewidth=2,
                linestyle="--", label=f"Regression (slope={z[0]:.3f})", zorder=4)
        corr = np.corrcoef(x_all, y_all)[0, 1]
        ax.text(0.05, 0.93, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.85))

    ax.axhline(_THRESHOLD, color="#95a5a6", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(_THRESHOLD, color="#95a5a6", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(_THRESHOLD + 0.005, 0.32, f"CP={_THRESHOLD}", color="#95a5a6", fontsize=8.5)
    ax.text(0.32, _THRESHOLD + 0.008, f"AR={_THRESHOLD}", color="#95a5a6", fontsize=8.5)

    ax.set_title("Retrieval Precision (CP) vs Answer Relevance (AR) Correlation",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Context Precision (CP)", fontsize=11)
    ax.set_ylabel("Answer Relevance (AR)", fontsize=11)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_buf(fig)


# ── Hypothesis 3: User-Level Personalization ─────────────────────────────────
def _plot_user_level_bar(df: pd.DataFrame):
    """Grouped bar chart: Professional vs Consumer across F, AR, CP."""
    subset = df[df["ragas_f"].notna()].copy()
    if subset.empty:
        return None

    metrics       = ["ragas_f", "ragas_ar", "ragas_cp"]
    metric_labels = ["Faithfulness (F)", "Answer Relevance (AR)", "Context Precision (CP)"]
    levels        = ["Professional", "Consumer"]

    stats: dict[str, dict] = {}
    for lv in levels:
        stats[lv] = {}
        grp = subset[subset["user_level"] == lv]
        for m in metrics:
            vals = grp[m].dropna().values
            mean = vals.mean() if len(vals) else 0.0
            ci   = (vals.std() / np.sqrt(len(vals)) * 1.96) if len(vals) > 1 else 0.0
            stats[lv][m] = {"mean": mean, "ci": ci}

    x     = np.arange(len(metrics))
    width = 0.32
    fig, ax = _fig(9, 5.5)

    for i, lv in enumerate(levels):
        offset = (i - 0.5) * width
        means  = [stats[lv][m]["mean"] for m in metrics]
        cis    = [stats[lv][m]["ci"]   for m in metrics]
        bars   = ax.bar(x + offset, means, width,
                        label=lv, color=_LEVEL_COLORS[lv],
                        alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(x + offset, means, yerr=cis, fmt="none",
                    color="#2c3e50", capsize=5, linewidth=1.5, zorder=4)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#2c3e50")

    ax.axhline(_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"Threshold ({_THRESHOLD})", zorder=2)

    ax.set_title("Hypothesis 3: RAGAS Metrics by User Level (Professional vs Consumer)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.set_ylabel("Mean Score (95% CI)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0.5, 1.12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _save_buf(fig)


# ── Multi-Tier Knowledge Hierarchy Analysis ──────────────────────────────────

def _plot_cumulative_success(df: pd.DataFrame):
    """Viz 1: Cumulative Success Rate by Tier — Stacked Bar Chart.

    Success = F >= 0.8 AND AR >= 0.8, judged per unique request_id.
    Tier 0 only → Tier 0+1 → Tier 0+1+2 cumulative.
    """
    sub = df[df["ragas_f"].notna() & df["ragas_ar"].notna()].copy()
    if sub.empty:
        return None

    # Per request_id: best row at each tier ceiling
    def success_at_tier(max_tier: int) -> float:
        pool = sub[sub["tier_id"] <= max_tier]
        # For each request, take the row with highest tier reached
        best = pool.sort_values("tier_id").groupby("request_id").last().reset_index()
        ok = ((best["ragas_f"] >= _THRESHOLD) & (best["ragas_ar"] >= _THRESHOLD)).sum()
        return ok / len(best) * 100 if len(best) else 0.0

    stages   = ["Tier 0\n(Vector DB)", "Tier 0+1\n(+LLM)", "Tier 0+1+2\n(+Web)"]
    rates    = [success_at_tier(0), success_at_tier(1), success_at_tier(2)]
    gains    = [rates[0], rates[1] - rates[0], rates[2] - rates[1]]
    colors   = ["#27ae60", "#f39c12", "#e74c3c"]
    labels   = ["Tier 0 Success", "Gained by Tier 1", "Gained by Tier 2"]

    fig, ax = _fig(8, 5.5)

    bottoms = [0, gains[0], gains[0] + gains[1]]
    for i, (gain, color, label, bottom) in enumerate(zip(gains, colors, labels, bottoms)):
        if gain <= 0:
            continue
        ax.bar(stages, [gain if j == i else 0 for j in range(3)],
               bottom=[bottom if j == i else 0 for j in range(3)],
               color=color, alpha=0.85, label=label,
               edgecolor="white", linewidth=1.2, width=0.45)

    # Cumulative rate labels on top of each bar
    for i, (stage, rate) in enumerate(zip(stages, rates)):
        ax.text(i, rate + 1.5, f"{rate:.1f}%",
                ha="center", fontsize=12, fontweight="bold", color="#2c3e50")

    # Gain arrows
    for i in range(1, 3):
        if rates[i] > rates[i-1]:
            ax.annotate(
                f"+{rates[i]-rates[i-1]:.1f}%",
                xy=(i, rates[i]), xytext=(i - 0.35, rates[i] - gains[i] / 2),
                fontsize=9.5, color="white", fontweight="bold", ha="center"
            )

    ax.set_title("Cumulative Answer Success Rate by Knowledge Tier",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Success Rate (F≥0.8 & AR≥0.8, %)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _save_buf(fig)




# ── Summary Metric Cards ──────────────────────────────────────────────────────
def _render_summary_cards(df: pd.DataFrame) -> None:
    # ── request 기준 집계 ──────────────────────────────────────────────────────
    # 각 request_id에서 가장 높은 tier(최종 결과) 행만 추출
    final = (
        df.sort_values(["request_id", "tier_id", "loop_count"])
          .groupby("request_id", as_index=False)
          .last()
    )
    total         = len(final)
    success_mask  = (
        (final["ragas_f"]  >= _THRESHOLD) &
        (final["ragas_ar"] >= _THRESHOLD) &
        (final["ragas_cp"] >= _THRESHOLD)
    )
    success_rate  = success_mask.mean() * 100 if total else 0.0
    avg_f         = final["ragas_f"].mean()
    avg_ar        = final["ragas_ar"].mean()
    avg_cp        = final["ragas_cp"].mean()
    # Escalation: 해당 request에서 중간 에스컬레이션이 한 번이라도 있었던 비율
    escalated_req = df.groupby("request_id")["is_escalated"].any()
    escalated_pct = escalated_req.mean() * 100 if total else 0.0
    # Fallback: 최종 행이 fallback인 요청 비율
    fallback_pct  = final["is_fallback"].mean() * 100 if total else 0.0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Requests",              f"{total:,}")
    c2.metric("Success (F∧AR∧CP≥0.8)", f"{success_rate:.1f}%")
    c3.metric("Avg F",           f"{avg_f:.3f}"  if not pd.isna(avg_f)  else "—")
    c4.metric("Avg AR",          f"{avg_ar:.3f}" if not pd.isna(avg_ar) else "—")
    c5.metric("Avg CP",          f"{avg_cp:.3f}" if not pd.isna(avg_cp) else "—")
    c6.metric("Escalation",      f"{escalated_pct:.1f}%")
    c7.metric("Fallback",        f"{fallback_pct:.1f}%")


# ── Main Render ───────────────────────────────────────────────────────────────
def render_performance_viz() -> None:
    """Render the full performance visualization dashboard."""
    _setup_font()

    st.title("RAG Performance Visualization")
    st.caption("PhD Dissertation Analysis — based on Supabase `rag_audit_log`")

    with st.spinner("Loading data..."):
        df = _load_data()

    if df.empty:
        st.error("Failed to load data. Please check Supabase connection.")
        return

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Filters")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        min_d = df["created_at"].min().date()
        max_d = df["created_at"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range", value=(min_d, max_d),
            min_value=min_d, max_value=max_d,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            import datetime, pytz
            d_from = datetime.datetime.combine(date_range[0], datetime.time.min).replace(tzinfo=pytz.utc)
            d_to   = datetime.datetime.combine(date_range[1], datetime.time.max).replace(tzinfo=pytz.utc)
            df = df[(df["created_at"] >= d_from) & (df["created_at"] <= d_to)]

    levels = st.sidebar.multiselect(
        "User Level",
        ["Professional", "Consumer"],
        default=["Professional", "Consumer"],
    )
    if levels:
        df = df[df["user_level"].isin(levels)]

    if df.empty:
        st.warning("No data found for the selected filters.")
        return

    if st.button("Refresh Data", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Summary Cards ─────────────────────────────────────────────────────────
    st.markdown("### Summary Statistics")
    _render_summary_cards(df)

    st.markdown("---")

    # ── Hypothesis 1 ──────────────────────────────────────────────────────────
    st.markdown("### Hypothesis 1: Self-Correction Effect on Faithfulness")
    st.caption("Repeated self-correction loops improve Faithfulness and suppress hallucination.")
    buf1 = _plot_self_correction(df)
    if buf1:
        st.image(buf1, use_container_width=True)

        # CI 공식 및 대입값 표시
        with st.expander("95% Confidence Interval — Formula & Substituted Values", expanded=True):
            st.latex(r"CI = \bar{x} \pm 1.96 \times \dfrac{s}{\sqrt{n}}")
            st.caption(
                "where  **x̄** = sample mean,  **s** = sample standard deviation,  "
                "**n** = sample size,  **1.96** = z-score for 95% confidence level"
            )
            st.markdown("**Substituted values by Loop Count (Faithfulness)**")

            rows = []
            for lc, grp in df.groupby("loop_count"):
                vals = grp["ragas_f"].dropna().values
                if len(vals) == 0:
                    continue
                n    = len(vals)
                mean = vals.mean()
                s    = vals.std(ddof=1) if n > 1 else 0.0
                se   = s / np.sqrt(n)
                ci   = 1.96 * se
                rows.append({
                    "Loop Count": int(lc),
                    "n": n,
                    "x̄ (Mean)": round(mean, 4),
                    "s (Std Dev)": round(s, 4),
                    "s / √n (SE)": round(se, 4),
                    "1.96 × SE": round(ci, 4),
                    "CI Lower": round(mean - ci, 4),
                    "CI Upper": round(mean + ci, 4),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Insufficient data.")

    st.markdown("---")

    st.markdown("---")

    # ── Hypothesis 2 ──────────────────────────────────────────────────────────
    st.markdown("### Hypothesis 2: Intelligent Escalation Validity")
    st.caption(
        "When the vector DB lacks relevant content, the system intelligently escalates "
        "to a higher knowledge tier instead of repeating ineffective self-correction loops."
    )

    # ── Immediate Escalation Analysis ────────────────────────────────────────────
    st.markdown("#### Immediate Escalation Analysis")
    st.caption(
        "The system triggers **immediate escalation** (skipping self-correction) under two independent conditions:  \n"
        "① **AR < {ar}** — Answer Relevance alone is critically low (no relevant content in DB)  \n"
        "② **F < {f} AND CP < {cp}** — Both Faithfulness and Context Precision are critically low (retrieval completely off-target)".format(
            ar=_IMM_AR_THR, f=_IMM_F_THR, cp=_IMM_CP_THR
        )
    )

    # ── Condition ①: AR-based escalation ──────────────────────────────────────
    st.markdown("#### Condition ① — AR-Based Immediate Escalation (AR < {:.1f})".format(_IMM_AR_THR))
    st.caption(
        "When Answer Relevance falls below **{:.1f}**, the system infers the vector DB contains "
        "no relevant content and immediately escalates without wasting loop iterations.".format(_IMM_AR_THR)
    )
    buf_ar = _plot_ar_escalation_zone(df)
    if buf_ar:
        st.image(buf_ar, use_container_width=True)
        ar_esc_n = int((df["is_escalated"] & (df["ragas_ar"] < _IMM_AR_THR)).sum())
        all_n    = int(df["ragas_ar"].notna().sum())
        st.markdown(
            f"★ AR-triggered escalation: **{ar_esc_n}** / {all_n} records &nbsp;|&nbsp; "
            f"Threshold: AR < `{_IMM_AR_THR}`",
            unsafe_allow_html=True,
        )
    else:
        st.info("Insufficient data.")

    # ── Condition ②: CP & F based escalation ──────────────────────────────────
    st.markdown("#### Condition ② — CP & F Dual-Threshold Immediate Escalation (F < {f} AND CP < {cp})".format(
        f=_IMM_F_THR, cp=_IMM_CP_THR))
    st.caption(
        "When both Faithfulness **and** Context Precision are critically low simultaneously, "
        "the retrieval is judged completely off-target and the system escalates immediately.".format()
    )
    buf_dz = _plot_decision_zone(df)
    if buf_dz:
        st.image(buf_dz, use_container_width=True)
        imm_n = int(((df["ragas_cp"] < _IMM_CP_THR) & (df["ragas_f"] < _IMM_F_THR) & df["is_escalated"]).sum())
        all_n = int((df["ragas_cp"].notna() & df["ragas_f"].notna()).sum())
        st.markdown(
            f"★ CP & F dual-triggered escalation: **{imm_n}** / {all_n} records &nbsp;|&nbsp; "
            f"Zone: CP < `{_IMM_CP_THR}` AND F < `{_IMM_F_THR}`",
            unsafe_allow_html=True,
        )
    else:
        st.info("Insufficient data.")

    st.markdown("---")

    # ── CP vs AR Scatter ──────────────────────────────────────────────────────
    st.markdown("### Retrieval Precision (CP) vs Answer Relevance (AR) Correlation")
    st.caption("Higher context precision is positively correlated with answer relevance.")
    buf3 = _plot_cp_ar_scatter(df)
    if buf3:
        st.image(buf3, use_container_width=True)
        valid = df[["ragas_cp", "ragas_ar"]].dropna()
        if len(valid) > 2:
            corr  = valid["ragas_cp"].corr(valid["ragas_ar"])
            above = ((valid["ragas_cp"] >= _THRESHOLD) & (valid["ragas_ar"] >= _THRESHOLD)).sum()
            pct   = above / len(valid) * 100
            st.markdown(
                f"**Pearson r = `{corr:.4f}`** &nbsp;|&nbsp; "
                f"Both CP≥{_THRESHOLD} & AR≥{_THRESHOLD}: **{above} records ({pct:.1f}%)**",
                unsafe_allow_html=True,
            )
    else:
        st.info("Insufficient data.")

    st.markdown("---")

    # ── Hypothesis 3 ──────────────────────────────────────────────────────────
    st.markdown("### Hypothesis 3: RAGAS Metrics by User Level")
    st.caption("The system maintains high reliability scores for both Professional and Consumer users.")
    buf4 = _plot_user_level_bar(df)
    if buf4:
        st.image(buf4, use_container_width=True)

        # CI 공식 및 대입값 표시
        with st.expander("95% Confidence Interval — Formula & Substituted Values", expanded=True):
            st.latex(r"CI = \bar{x} \pm 1.96 \times \dfrac{s}{\sqrt{n}}")
            st.caption(
                "where  **x̄** = sample mean,  **s** = sample standard deviation,  "
                "**n** = sample size,  **1.96** = z-score for 95% confidence level"
            )
            st.markdown("**Substituted values by User Level & Metric**")

            metrics_map = {
                "ragas_f":  "Faithfulness (F)",
                "ragas_ar": "Answer Relevance (AR)",
                "ragas_cp": "Context Precision (CP)",
            }
            rows = []
            for lv in ["Professional", "Consumer"]:
                grp = df[df["user_level"] == lv]
                for col, label in metrics_map.items():
                    vals = grp[col].dropna().values
                    if len(vals) == 0:
                        continue
                    n    = len(vals)
                    mean = vals.mean()
                    s    = vals.std(ddof=1) if n > 1 else 0.0
                    se   = s / np.sqrt(n)
                    ci   = 1.96 * se
                    rows.append({
                        "User Level": lv,
                        "Metric": label,
                        "n": n,
                        "x̄ (Mean)": round(mean, 4),
                        "s (Std Dev)": round(s, 4),
                        "s / √n (SE)": round(se, 4),
                        "1.96 × SE": round(ci, 4),
                        "CI Lower": round(mean - ci, 4),
                        "CI Upper": round(mean + ci, 4),
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Insufficient data.")

    st.markdown("---")

    # ── Multi-Tier Knowledge Hierarchy Analysis ───────────────────────────────
    st.markdown("### Multi-Tier Knowledge Hierarchy Analysis")
    st.caption(
        "Demonstrates that the Multi-Tier architecture (Tier 0 → Tier 1 → Tier 2) "
        "substantially expands answer coverage and quality beyond a single-source RAG system."
    )

    # Viz 1: Cumulative Success Rate
    st.markdown("#### Viz 1 — Cumulative Answer Success Rate by Tier")
    st.caption("Each tier added cumulatively increases the proportion of requests meeting quality thresholds (F≥0.8 & AR≥0.8).")
    buf_cs = _plot_cumulative_success(df)
    if buf_cs:
        st.image(buf_cs, use_container_width=True)
        def _success_rate(max_tier):
            pool = df[df["ragas_f"].notna() & df["ragas_ar"].notna() & (df["tier_id"] <= max_tier)]
            best = pool.sort_values("tier_id").groupby("request_id").last()
            ok = ((best["ragas_f"] >= _THRESHOLD) & (best["ragas_ar"] >= _THRESHOLD)).sum()
            return round(ok / len(best) * 100, 1) if len(best) else 0.0
        r0, r1, r2 = _success_rate(0), _success_rate(1), _success_rate(2)
        st.dataframe(pd.DataFrame([
            {"Stage": "Tier 0 only (Vector DB)",   "Success Rate (%)": r0, "Gain (%)": "—"},
            {"Stage": "Tier 0+1 (+LLM)",           "Success Rate (%)": r1, "Gain (%)": f"+{r1-r0:.1f}"},
            {"Stage": "Tier 0+1+2 (+Web Search)",  "Success Rate (%)": r2, "Gain (%)": f"+{r2-r1:.1f}"},
        ]), hide_index=True, use_container_width=False)
    else:
        st.info("Insufficient data.")

