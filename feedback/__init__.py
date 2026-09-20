"""Feedback loop sistemi - modelin hatalarından öğrenme mekanizması.

Bu modül şu bileşenleri içerir:
1. ErrorTracker: Tahminleri ve sonuçları kaydedip analiz eder
2. PatternDetector: Sistemli hata kalıplarını tespit eder
3. SelfCorrectionEngine: Model parametrelerini otomatik günceller
4. FeatureGenerator: Hata kalıplarından yeni özellikler türetir
5. FeedbackOrchestrator: Tüm bileşenleri koordine eder
"""

from feedback.error_tracker import ErrorTracker
from feedback.pattern_detector import PatternDetector
from feedback.self_correction import SelfCorrectionEngine
from feedback.feature_generator import ErrorFeatureGenerator
from feedback.orchestrator import FeedbackOrchestrator

__all__ = [
    "ErrorTracker",
    "PatternDetector",
    "SelfCorrectionEngine",
    "ErrorFeatureGenerator",
    "FeedbackOrchestrator",
]
