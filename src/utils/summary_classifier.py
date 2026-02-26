def classify_mode(user_query: str) -> str:
    q = user_query.lower()

    if any(w in q for w in ["tldr", "brief", "short", "quick", "gist"]):
        return "snapshot"

    if any(w in q for w in ["detail", "analyze", "full", "deep"]):
        return "deepdive"

    return "overview"
