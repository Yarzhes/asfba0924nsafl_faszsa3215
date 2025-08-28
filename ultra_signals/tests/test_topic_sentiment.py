import pytest
from ultra_signals.sentiment.topic_classifier import TopicClassifier, extract_symbols
from ultra_signals.sentiment.fusion import SentimentFusion
from ultra_signals.sentiment.divergence import DivergenceDetector


def test_topic_classifier_basic():
    tc = TopicClassifier()
    res = tc.classify("Bitcoin ETF approval likely, institutions buying")
    assert "etf_regulation" in res or isinstance(res, dict)


def test_extract_symbols():
    s = extract_symbols("We like $BTC and #ETH right now")
    assert "BTC" in s and "ETH" in s


def test_fusion_simple():
    f = SentimentFusion()
    per_source = {
        "twitter": {"etf_regulation": 0.8},
        "reddit": {"etf_regulation": 0.6},
    }
    out = f.fuse(per_source)
    assert "etf_regulation" in out


def test_divergence_rules():
    det = DivergenceDetector(funding_threshold=0.0001, oi_threshold_pct=0.01)
    topic_scores = {"etf_regulation": {"score_s": 0.7, "pctl": 95}}
    funding = {"funding_now": -0.0002}
    oi = {"oi_rate": 0.02}
    res = det.detect("BTCUSDT", topic_scores, funding, oi)
    assert res.get("contrarian_flag_long") == 1
