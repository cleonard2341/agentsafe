from agentsafe.detectors.action_statement import ActionStatementDetector
from agentsafe.detectors.base import BaseDetector
from agentsafe.detectors.behavioral_baseline import BehavioralBaselineDetector
from agentsafe.detectors.prompt_injection import PromptInjectionDetector
from agentsafe.detectors.scope_creep import ScopeCreepDetector

__all__ = [
    "BaseDetector",
    "ActionStatementDetector",
    "BehavioralBaselineDetector",
    "PromptInjectionDetector",
    "ScopeCreepDetector",
]


def default_detectors() -> list[BaseDetector]:
    # BehavioralBaselineDetector requires a repo reference — it must be
    # added manually via agentsafe.wrap(detectors=[...]) or it gets
    # injected automatically in wrap() below.
    return [
        PromptInjectionDetector(),
        ScopeCreepDetector(),
        ActionStatementDetector(),
    ]
