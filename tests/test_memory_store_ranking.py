from datetime import datetime, timezone

from langmem.knowledge.extraction import MemoryStoreManager, SearchItem


def test_duplicate_memory_keeps_its_highest_search_score():
    now = datetime.now(timezone.utc)
    best = SearchItem(("memories",), "best", {}, now, now, score=0.9)
    later_match = SearchItem(("memories",), "best", {}, now, now, score=0.1)
    other = SearchItem(("memories",), "other", {}, now, now, score=0.5)

    results = MemoryStoreManager._sort_results([[best], [later_match, other]], 1)

    assert list(results.values()) == [best]
