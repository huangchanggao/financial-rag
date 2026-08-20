def route_ticker(question):
    q = question.lower()

    if "nvidia" in q or "nvda" in q:
        return "NVDA"

    if "amd" in q or "advanced micro devices" in q:
        return "AMD"

    if "microsoft" in q or "msft" in q:
        return "MSFT"

    return None

def route_section(question):
    q = question.lower()

    # Item 7A - Market Risk
    if (
        "market risk" in q
        or "interest rate risk" in q
        or "foreign exchange risk" in q
        or "foreign exchange rate" in q
    ):
        return "Item 7A"

    # Item 1C - Cybersecurity
    if (
        "cybersecurity" in q
        or "cyber security" in q
    ):
        return "Item 1C"

    # Item 1A - Risk Factors
    if (
        "risk" in q
        or "risks" in q
        or "competition" in q
        or "competitive" in q
        or "competitors" in q
        or "regulation" in q
        or "regulations" in q
        or "government regulation" in q
        or "export control" in q
        or "data security" in q
        or "supply chain" in q
        or "third-party manufacturer" in q
        or "third party manufacturer" in q
        or "customer demand" in q
    ):
        return "Item 1A"

    # Item 7 - MD&A
    if (
        "financial condition" in q
        or "operating results" in q
        or "results of operations" in q
        or "revenue" in q
        or "business performance" in q
        or "financial performance" in q
    ):
        return "Item 7"

    # Item 9A - Controls
    if (
        "controls and procedures" in q
        or "internal control" in q
        or "internal controls" in q
    ):
        return "Item 9A"

    # Item 10 - Governance
    if (
        "directors" in q
        or "corporate governance" in q
        or "executive officers" in q
        or "audit committee" in q
        or "code of conduct" in q
    ):
        return "Item 10"

    # Item 11 - Compensation
    if (
        "executive compensation" in q
        or "director compensation" in q
        or "compensation committee" in q
    ):
        return "Item 11"

    # Item 1 - Business
    if (
        "business" in q
        or "accelerated computing" in q
        or "gpu" in q
        or "gpus" in q
        or "platform" in q
        or "platforms" in q
        or "software" in q
        or "data center" in q
    ):
        return "Item 1"

    return None