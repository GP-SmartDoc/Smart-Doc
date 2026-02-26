
def compute_budget(doc_tokens: int, mode_config):
    target = int(doc_tokens * mode_config["compression"])
    return min(target, mode_config["max_tokens"])
