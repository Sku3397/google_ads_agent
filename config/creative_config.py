from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CreativeConfig:
    """Configuration settings for creative optimization."""

    # Minimum thresholds for statistical significance
    MIN_SAMPLE_SIZE: int = 1000
    MIN_CONFIDENCE_LEVEL: float = 0.90
    DEFAULT_CONFIDENCE_LEVEL: float = 0.95

    # Performance thresholds
    MIN_CTR_THRESHOLD: float = 0.01  # 1% CTR
    MIN_CONV_RATE_THRESHOLD: float = 0.02  # 2% conversion rate
    MAX_CPA_THRESHOLD: float = 50.0  # $50 CPA

    # Text analysis settings
    MIN_HEADLINE_LENGTH: int = 4
    MAX_HEADLINE_LENGTH: int = 30
    MIN_DESCRIPTION_LENGTH: int = 10
    MAX_DESCRIPTION_LENGTH: int = 90

    # Testing parameters
    MIN_TEST_DURATION_DAYS: int = 14
    MAX_TEST_DURATION_DAYS: int = 90
    MIN_VARIANTS_PER_TEST: int = 2
    MAX_VARIANTS_PER_TEST: int = 4

    # Creative element weights for scoring
    ELEMENT_WEIGHTS: Dict[str, float] = {"ctr": 0.3, "conversion_rate": 0.4, "cpa": 0.3}

    # Recommendation thresholds
    SIMILARITY_THRESHOLD: float = 0.8  # Text similarity threshold
    FATIGUE_THRESHOLD: float = 0.2  # CTR drop threshold for fatigue detection

    # API query settings
    DEFAULT_LOOKBACK_DAYS: int = 30
    MAX_LOOKBACK_DAYS: int = 90
    RESULTS_PER_PAGE: int = 1000

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate threshold ranges
            assert 0 < cls.MIN_CONFIDENCE_LEVEL < cls.DEFAULT_CONFIDENCE_LEVEL <= 1.0
            assert cls.MIN_SAMPLE_SIZE > 0
            assert 0 < cls.MIN_CTR_THRESHOLD < 1
            assert 0 < cls.MIN_CONV_RATE_THRESHOLD < 1
            assert cls.MAX_CPA_THRESHOLD > 0

            # Validate text length settings
            assert 0 < cls.MIN_HEADLINE_LENGTH < cls.MAX_HEADLINE_LENGTH
            assert 0 < cls.MIN_DESCRIPTION_LENGTH < cls.MAX_DESCRIPTION_LENGTH

            # Validate test parameters
            assert 0 < cls.MIN_TEST_DURATION_DAYS < cls.MAX_TEST_DURATION_DAYS
            assert 1 < cls.MIN_VARIANTS_PER_TEST <= cls.MAX_VARIANTS_PER_TEST

            # Validate weights sum to 1
            assert abs(sum(cls.ELEMENT_WEIGHTS.values()) - 1.0) < 0.001

            # Validate thresholds
            assert 0 <= cls.SIMILARITY_THRESHOLD <= 1
            assert 0 <= cls.FATIGUE_THRESHOLD <= 1

            # Validate API settings
            assert 0 < cls.DEFAULT_LOOKBACK_DAYS <= cls.MAX_LOOKBACK_DAYS
            assert cls.RESULTS_PER_PAGE > 0

            return True

        except AssertionError:
            return False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "min_sample_size": cls.MIN_SAMPLE_SIZE,
            "confidence_level": cls.DEFAULT_CONFIDENCE_LEVEL,
            "min_ctr": cls.MIN_CTR_THRESHOLD,
            "min_conv_rate": cls.MIN_CONV_RATE_THRESHOLD,
            "max_cpa": cls.MAX_CPA_THRESHOLD,
            "headline_length": {"min": cls.MIN_HEADLINE_LENGTH, "max": cls.MAX_HEADLINE_LENGTH},
            "description_length": {
                "min": cls.MIN_DESCRIPTION_LENGTH,
                "max": cls.MAX_DESCRIPTION_LENGTH,
            },
            "test_duration": {
                "min_days": cls.MIN_TEST_DURATION_DAYS,
                "max_days": cls.MAX_TEST_DURATION_DAYS,
            },
            "variants": {"min": cls.MIN_VARIANTS_PER_TEST, "max": cls.MAX_VARIANTS_PER_TEST},
            "element_weights": cls.ELEMENT_WEIGHTS,
            "thresholds": {
                "similarity": cls.SIMILARITY_THRESHOLD,
                "fatigue": cls.FATIGUE_THRESHOLD,
            },
            "api_settings": {
                "lookback_days": cls.DEFAULT_LOOKBACK_DAYS,
                "max_lookback": cls.MAX_LOOKBACK_DAYS,
                "results_per_page": cls.RESULTS_PER_PAGE,
            },
        }
