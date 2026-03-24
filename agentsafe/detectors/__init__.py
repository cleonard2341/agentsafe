from agentsafe.detectors.action_statement import ActionStatementDetector
from agentsafe.detectors.base import BaseDetector
from agentsafe.detectors.behavioral_baseline import BehavioralBaselineDetector
from agentsafe.detectors.loop_detection import LoopDetector
from agentsafe.detectors.prompt_injection import PromptInjectionDetector
from agentsafe.detectors.scope_creep import ScopeCreepDetector
from agentsafe.detectors.self_continuity import SelfContinuityDetector
from agentsafe.detectors.sycophancy import SycophancyDetector

__all__ = [
    "BaseDetector",
    "ActionStatementDetector",
    "BehavioralBaselineDetector",
    "LoopDetector",
    "PromptInjectionDetector",
    "ScopeCreepDetector",
    "SelfContinuityDetector",
    "SycophancyDetector",
]


def default_detectors() -> list[BaseDetector]:
    # BehavioralBaselineDetector requires a repo reference — injected in wrap()
    return [
        PromptInjectionDetector(),
        ScopeCreepDetector(),
        ActionStatementDetector(),
        SycophancyDetector(),
        SelfContinuityDetector(),
        LoopDetector(),
    ]
