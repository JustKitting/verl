# Copyright 2025 Nous Research
# Licensed under the Apache License, Version 2.0
"""HTTP utilities."""

import logging
import time

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def wait_for_service(url: str, timeout: int = 60, interval: float = 1.0, name: str = None) -> bool:
    """Poll until service returns 200."""
    name = name or url
    logger.info(f"Waiting for {name}...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                logger.info(f"{name} is ready")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)

    logger.error(f"{name} not available after {timeout}s")
    return False


def retry_on_failure(attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    """Decorator for retrying on HTTP failures with exponential backoff."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(requests.RequestException),
    )
