# 6. Option (i) - AI Crawling Strategy
from pagerank import pagerank_closed_form


def top_k_crawl_urls(web_graph: dict, pagerank_scores: dict, k: int = 5,
                     blocked_prefixes: tuple = ('/private', '/admin')) -> list:
    """
    Returns top-k URLs to crawl based on PageRank authority,
    filtering out disallowed paths (robots.txt heuristic).
    
    Heuristic: prefer pages with PageRank > mean AND not blocked.
    Returns list of (url, score, reason) tuples.
    """
    mean_pr = np.mean(list(pagerank_scores.values()))
    candidates = []
    for url, score in pagerank_scores.items():
        # Check robots.txt heuristic
        is_blocked = any(url.startswith(prefix) for prefix in blocked_prefixes)
        if is_blocked:
            continue
        outlinks = web_graph.get(url, [])
        # Bonus for hub-like pages (many outlinks = more training diversity)
        hub_bonus = np.log1p(len(outlinks)) * 0.01
        adjusted_score = score + hub_bonus
        candidates.append((url, adjusted_score, score, len(outlinks)))
    
    # Sort by adjusted score
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:k]
    
    print(f"\n{'='*60}")
    print(f"AI Crawling Strategy - Top {k} URLs")
    print(f"{'='*60}")
    print(f"Mean PageRank: {mean_pr:.4f}")
    print(f"\n{'URL':<30} {'PR Score':>10} {'Hub Bonus':>10} {'Outlinks':>8}")
    print("-" * 60)
    for url, adj_score, raw_score, n_out in top_k:
        bonus = adj_score - raw_score
        print(f"{url:<30} {raw_score:>10.4f} {bonus:>10.4f} {n_out:>8}")
    return top_k


# Example web graph for crawling demo
sample_web_graph = {
    'arxiv.org/ai':       ['nature.com', 'openai.com', 'deepmind.com', 'github.com'],
    'openai.com':         ['arxiv.org/ai', 'github.com', 'huggingface.co'],
    'deepmind.com':       ['arxiv.org/ai', 'nature.com'],
    'nature.com':         ['arxiv.org/ai', 'pubmed.gov'],
    'pubmed.gov':         ['nature.com'],
    'github.com':         ['openai.com', 'huggingface.co', 'arxiv.org/ai'],
    'huggingface.co':     ['arxiv.org/ai', 'github.com', 'openai.com'],
    '/private/admin':     ['openai.com'],     
}

pr_crawl, *_ = pagerank_closed_form(sample_web_graph, p=0.15)
top_urls = top_k_crawl_urls(sample_web_graph, pr_crawl, k=5)

print("\nRationale: High-PageRank pages are authoritative hubs that many")
print("other trusted pages link to -> richer, higher-quality training data.")
print("Hub bonus rewards pages with many outlinks (more diverse content).")
print("Robots.txt heuristic filters /private and /admin paths.")