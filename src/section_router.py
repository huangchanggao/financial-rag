def route_section(question):
    q = question.lower()

    # ==================================================
    # Item 7A - Market Risk
    # ==================================================

    if (
        "market risk" in q
        or "interest rate risk" in q
        or "foreign exchange risk" in q
        or "foreign exchange rate" in q
    ):
        return (
            "Item 7A - Quantitative and Qualitative "
            "Disclosures about Market Risk"
        )

    # ==================================================
    # Item 1A - Risk Factors
    # ==================================================

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
        or "cybersecurity" in q
        or "data security" in q
        or "supply chain" in q
        or "third-party manufacturer" in q
        or "third party manufacturer" in q
        or "customer demand" in q
    ):
        return "Item 1A - Risk Factors"

    # ==================================================
    # Item 7 - Management's Discussion and Analysis
    # ==================================================

    if (
        "financial condition" in q
        or "operating results" in q
        or "results of operations" in q
        or "revenue" in q
        or "business performance" in q
        or "financial performance" in q
    ):
        return (
            "Item 7 - Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations"
        )

    # ==================================================
    # Item 9A - Controls and Procedures
    # ==================================================

    if (
        "controls and procedures" in q
        or "internal control" in q
        or "internal controls" in q
    ):
        return "Item 9A - Controls and Procedures"

    # ==================================================
    # Item 10 - Directors / Corporate Governance
    # ==================================================

    if (
        "directors" in q
        or "corporate governance" in q
        or "executive officers" in q
        or "audit committee" in q
        or "code of conduct" in q
    ):
        return (
            "Item 10 - Directors, Executive Officers "
            "and Corporate Governance"
        )

    # ==================================================
    # Item 11 - Executive Compensation
    # ==================================================

    if (
        "executive compensation" in q
        or "director compensation" in q
        or "compensation committee" in q
    ):
        return "Item 11 - Executive Compensation"

    # ==================================================
    # Item 1 - Business
    # ==================================================

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
        return "Item 1 - Business"

    # ==================================================
    # 無法判斷
    # ==================================================

    return None