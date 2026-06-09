#!/usr/bin/env python
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from graph import run_medical_self_corrective_rag
from stqs_questions import QUESTIONS, STQS_EXPECTED_TOTAL

FALLBACK_QUERY_INDICES_T0_COND_C = sorted(
    {
        2,
        3,
        8,
        12,
        14,
        15,
        17,
        18,
        19,
        20,
        21,
        24,
        26,
        27,
        32,
        38,
        39,
        44,
        45,
        48,
        52,
        53,
        55,
        56,
        58,
        59,
        68,
        71,
        73,
        76,
        77,
        80,
        84,
        90,
        91,
        97,
        102,
        103,
    }
)


def _detect_fallback(state: dict) -> bool:
    log = state.get("log") or []
    ans = state.get("answer") or ""
    if any("[Final] Fallback" in line for line in log):
        return True
    return "신뢰할 수 있는 근거를 찾지 못했습니다" in ans


def main() -> int:
    assert len(QUESTIONS) == STQS_EXPECTED_TOTAL

    fb = 0
    ok = 0
    errs = []

    print(f"T0 재작성 Fallback 인덱스 {len(FALLBACK_QUERY_INDICES_T0_COND_C)}개, 조건 C\n")

    for q_idx in FALLBACK_QUERY_INDICES_T0_COND_C:
        disease, level_label, exp_tier, question = QUESTIONS[q_idx - 1]
        t0 = time.perf_counter()
        try:
            st = run_medical_self_corrective_rag(
                question,
                forced_user_level=None,
                step_callback=None,
                llm_provider="openai",
                ablation_condition="C",
                expected_tier=exp_tier,
                query_index=q_idx,
                disease=disease,
                query_level_label=level_label,
            )
            elapsed = time.perf_counter() - t0
            is_fb = _detect_fallback(st)
            f = st.get("critic_score", 0.0)
            ar = st.get("answer_relevance_score", 0.0)
            cp = st.get("context_precision_score", 0.0)

            fb += int(is_fb)
            ok += 1

            mark = "[FALLBACK]" if is_fb else ""
            print(
                f"[{q_idx:3d}] [{level_label}T{exp_tier}] {elapsed:5.1f}s "
                f"F={f:.2f} AR={ar:.2f} CP={cp:.2f} {mark} {disease}"
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            errs.append((q_idx, str(e)))
            print(f"[{q_idx:3d}] ERROR ({elapsed:.1f}s): {e}")

    print(f"\n완료: 성공 {ok}/{len(FALLBACK_QUERY_INDICES_T0_COND_C)}, Fallback {fb}, 오류 {len(errs)}")
    if errs:
        for q_idx, msg in errs:
            print(f"  idx {q_idx}: {msg[:200]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
