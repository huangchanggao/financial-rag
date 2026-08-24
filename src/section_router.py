def route_ticker(question):
    q = question.lower()

    if "nvidia" in q or "nvda" in q:
        return "NVDA"

    if "amd" in q or "advanced micro devices" in q:
        return "AMD"

    if "microsoft" in q or "msft" in q:
        return "MSFT"

    return None