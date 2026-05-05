"""Herramientas disponibles para el agente LangGraph."""

from src.agent.tools.mock_augment import augment_data_linear_relation
from src.agent.tools.mock_drift import detect_drift_kolmogorov_smirnov
from src.agent.tools.mock_synthetic import generate_synthetic_series

AGENT_TOOLS = [
    detect_drift_kolmogorov_smirnov,
    generate_synthetic_series,
    augment_data_linear_relation,
]

__all__ = [
    "AGENT_TOOLS",
    "detect_drift_kolmogorov_smirnov",
    "generate_synthetic_series",
    "augment_data_linear_relation",
]
