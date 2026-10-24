"""Token accounting from llm_usage, prompt-head sizing from llm_exchange."""

from collections.abc import Sequence


def token_totals(events: Sequence[dict]) -> dict:
    """
    Aggregate token usage from llm_usage events.
    
    Only events whose "message" == "llm_usage" AND whose data has a numeric
    "prompt_tokens" are counted.
    
    Returns exactly: {"prompt": int, "completion": int, "total": int, "records": int}
    """
    prompt_sum = 0
    completion_sum = 0
    total_sum = 0
    records = 0
    
    for event in events:
        if event.get("message") != "llm_usage":
            continue
            
        data = event.get("data", {})
        prompt_tokens = data.get("prompt_tokens")
        
        # Must have numeric prompt_tokens
        if not isinstance(prompt_tokens, (int, float)):
            continue
            
        # Clamp negative values to 0
        prompt = max(0, int(prompt_tokens))
        completion = max(0, int(data.get("completion_tokens", 0)))
        total = max(0, int(data.get("total_tokens", 0)))
        
        # Derive total if 0
        if total == 0:
            total = prompt + completion
            
        prompt_sum += prompt
        completion_sum += completion
        total_sum += total
        records += 1
        
    return {
        "prompt": prompt_sum,
        "completion": completion_sum,
        "total": total_sum,
        "records": records
    }


def role_tokens(events: Sequence[dict]) -> dict[str, int]:
    """
    Sum total_tokens by model_role from llm_usage events.
    
    Uses the same eligible population as token_totals.
    Keys sorted ascending.
    """
    role_sums: dict[str, int] = {}
    
    for event in events:
        if event.get("message") != "llm_usage":
            continue
            
        data = event.get("data", {})
        prompt_tokens = data.get("prompt_tokens")
        
        # Must have numeric prompt_tokens
        if not isinstance(prompt_tokens, (int, float)):
            continue
            
        model_role = data.get("model_role")
        if not isinstance(model_role, str):
            continue
            
        total = max(0, int(data.get("total_tokens", 0)))
        
        # Derive total if 0 (same logic as token_totals)
        if total == 0:
            prompt = max(0, int(prompt_tokens))
            completion = max(0, int(data.get("completion_tokens", 0)))
            total = prompt + completion
            
        role_sums[model_role] = role_sums.get(model_role, 0) + total
        
    # Return with keys sorted ascending
    return {k: role_sums[k] for k in sorted(role_sums.keys())}


def exchange_sizes(events: Sequence[dict]) -> dict:
    """
    Calculate prompt/response head sizes from llm_exchange events.
    
    Only events whose "message" == "llm_exchange".
    
    Returns exactly: {"max_prompt_head": int, "max_response_head": int, 
                      "mean_prompt_head": float, "count": int}
    """
    prompt_lengths = []
    response_lengths = []
    count = 0
    
    for event in events:
        if event.get("message") != "llm_exchange":
            continue
            
        data = event.get("data", {})
        count += 1
        
        prompt_head = data.get("prompt_head")
        response_head = data.get("response_head")
        
        prompt_len = len(str(prompt_head)) if prompt_head is not None else 0
        response_len = len(str(response_head)) if response_head is not None else 0
        
        prompt_lengths.append(prompt_len)
        response_lengths.append(response_len)
        
    if count == 0:
        return {
            "max_prompt_head": 0,
            "max_response_head": 0,
            "mean_prompt_head": 0.0,
            "count": 0
        }
        
    return {
        "max_prompt_head": max(prompt_lengths),
        "max_response_head": max(response_lengths),
        "mean_prompt_head": sum(prompt_lengths) / count,
        "count": count
    }
