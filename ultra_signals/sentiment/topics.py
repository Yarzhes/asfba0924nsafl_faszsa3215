"""Topic taxonomy and seed keywords/regex for topic-aware sentiment.

This is a minimal, editable taxonomy used by the rule-based fallback classifier.
"""

TOPIC_TAXONOMY = {
    "etf_regulation": {
        "label": "ETF / Regulation",
        "seeds": [r"\betf\b", r"\bsec\b", r"regulat", r"approval", r"denied", r"spot etf"],
        "desc": "News and opinion about ETFs, regulators and legal action",
    },
    "hack_exploit": {
        "label": "Hack / Exploit",
        "seeds": [r"\bhack\b", r"exploit", r"rug pull", r"breach", r"compromise", r"stolen"],
        "desc": "Security incidents, exploits and thefts",
    },
    "adoption": {
        "label": "Adoption / Macro",
        "seeds": [r"adopt", r"institutional", r"merchant", r"integration", r"adoption"],
        "desc": "Adoption by institutions, payments, macro events",
    },
    "dev_roadmap": {
        "label": "Development / Roadmap",
        "seeds": [r"upgrade", r"hard fork", r"mainnet", r"testnet", r"roadmap", r"merge", r"consensus"],
        "desc": "Protocol development and roadmap news",
    },
    "exchange_outage": {
        "label": "Exchange / Outage",
        "seeds": [r"outage", r"downtime", r"withdrawal", r"exchange down", r"maintenance"],
        "desc": "Exchange incidents and availability issues",
    },
    "meme_retail": {
        "label": "Meme / Retail Euphoria",
        "seeds": [r"to the moon|moon(ed)?|moonshot|hodl|rocket|tothemoon", r"rekt", r"pump", r"diamond hands", r"bagholder"],
        "desc": "Retail-driven meme and euphoric chatter",
    },
    "smart_money": {
        "label": "Smart Money / Whales",
        "seeds": [r"whale", r"onchain whale", r"large buy", r"squid|smart money", r"balance shift", r"scooping"],
        "desc": "Whale / institutional flow and commentary",
    },
}

DEFAULT_TOPIC = "uncertainty"
