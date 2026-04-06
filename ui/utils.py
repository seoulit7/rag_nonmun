"""UI 전반에서 공통으로 사용하는 점수 표시 헬퍼."""


def score_badge(score: float) -> str:
    """점수를 색상 배지 마크다운 문자열로 변환한다."""
    if score >= 0.8:
        return f"🟢 **{score:.2f}**"
    if score >= 0.6:
        return f"🟡 **{score:.2f}**"
    return f"🔴 **{score:.2f}**"


def score_label(score: float) -> str:
    """점수를 '양호'/'보통'/'미흡' 레이블로 변환한다."""
    return "양호" if score >= 0.8 else ("보통" if score >= 0.6 else "미흡")
