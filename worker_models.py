"""
Worker-specific models module
This module provides model initialization for RQ workers only
"""

import logging

# Setup logger for worker models
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variables for worker models
_table_masking_worker = None
_crypto_manager = None
_initialized = False


def initialize_worker_models():
    """
    Initialize models for worker use.
    This should be called once per worker process.
    """
    global _table_masking_worker, _crypto_manager, _initialized

    if _initialized:
        return

    logger.info("Initializing worker models...")

    try:
        # Import required modules
        import yaml
        import sys
        import traceback

        # Import the classes from service (we only need the class definitions)
        from service import CryptoFileManager, TextMasking

        # Load configuration
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize crypto manager
        _crypto_manager = CryptoFileManager()

        # Initialize table masking worker
        _table_masking_worker = TextMasking(
            model_path=config["paths"]["model"],
            rules_file_path=config["paths"]["rules_file"],
            medicine_path=config["paths"]["medicine_file"],
            name_path=config["paths"]["name_file"],
            extra_rules_file_path=config["paths"]["extra_rules_file"],
            crypto_manager=_crypto_manager,
            alg=config["alg"],
        )

        _initialized = True
        logger.info("Worker models initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize worker models: {e}")
        traceback.print_exc()
        raise


def get_table_masking_worker():
    """Get the table masking worker instance"""
    if not _initialized:
        initialize_worker_models()
    return _table_masking_worker


def get_crypto_manager():
    """Get the crypto manager instance"""
    if not _initialized:
        initialize_worker_models()
    return _crypto_manager


def is_initialized():
    """Check if models have been initialized"""
    return _initialized