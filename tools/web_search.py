import json

from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

import config.settings as settings


@tool
def search_web(query: str) -> str:
    """DuckDuckGo를 통해 최신 의료 정보를 웹에서 검색합니다.

    Args:
        query: 웹 검색에 사용할 영문 의료 쿼리

    Returns:
        JSON 문자열 {"chunks": [...], "sources": [...]}
    """
    search = DuckDuckGoSearchAPIWrapper(max_results=settings.WEB_SEARCH_MAX_RESULTS)
    results = search.results(
        query + " medical information",
        num_results=settings.WEB_SEARCH_MAX_RESULTS,
    )

    chunks, sources = [], []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("link", "")
        chunk = f"{title}: {snippet}"
        chunks.append(chunk[:1500])
        sources.append(url or title)

    return json.dumps({"chunks": chunks, "sources": sources})
