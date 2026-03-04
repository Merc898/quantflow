"""Signal fusion, aggregation, and recommendation engine."""

from quantflow.signals.aggregator import AggregationResult, EnsembleAggregator
from quantflow.signals.calibrator import DynamicWeightCalibrator
from quantflow.signals.explainer import ExplanationReport, RecommendationExplainer
from quantflow.signals.normalizer import SignalNormalizer
from quantflow.signals.recommendation import FinalRecommendation, RecommendationEngine
from quantflow.signals.regime_detector import RegimeDetector, RegimeState

__all__ = [
    "AggregationResult",
    "DynamicWeightCalibrator",
    "EnsembleAggregator",
    "ExplanationReport",
    "FinalRecommendation",
    "RecommendationEngine",
    "RecommendationExplainer",
    "RegimeDetector",
    "RegimeState",
    "SignalNormalizer",
]