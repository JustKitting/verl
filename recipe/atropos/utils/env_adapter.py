#!/usr/bin/env python3
"""
Universal VeRL-Atropos Environment Adapter.

Wraps any upstream Atropos environment for use with VeRL by reading
server URLs from ATROPOS_SERVER_URLS environment variable.
"""

import json
import os
from typing import List, Tuple, Type, TypeVar

from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.envs.server_handling.server_baseline import APIServerConfig


T = TypeVar("T", bound=BaseEnv)


def get_verl_server_configs(model_name: str = None) -> List[APIServerConfig]:
    """Build APIServerConfig list from ATROPOS_SERVER_URLS env var."""
    server_urls_json = os.environ.get("ATROPOS_SERVER_URLS")
    if not server_urls_json:
        raise ValueError(
            "ATROPOS_SERVER_URLS environment variable is required but not set.\n"
            "Set ATROPOS_SERVER_URLS='[\"http://host:port/v1\"]' or run via VeRL trainer."
        )

    server_urls = json.loads(server_urls_json)
    model_name = model_name or os.environ.get("ATROPOS_MODEL_NAME", "unknown")

    servers = []
    for url in server_urls:
        base_url = url if url.endswith("/v1") else url.rstrip("/") + "/v1"
        servers.append(
            APIServerConfig(
                base_url=base_url,
                model_name=model_name,
                server_type="sglang",
                api_key="x",
                timeout=1200,
                num_max_requests_at_once=512,
                num_requests_for_eval=256,
            )
        )

    print(f"[verl_adapter] Configured {len(servers)} server(s), model: {model_name}")
    return servers


def create_verl_adapter(
    env_class: Type[T],
    default_tokenizer: str = "Qwen/Qwen3-0.6B",
) -> Type[T]:
    """
    Wrap an Atropos environment class for VeRL integration.

    Overrides config_init() to inject server URLs from ATROPOS_SERVER_URLS env var.
    """

    class VeRLAdapter(env_class):
        @classmethod
        def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
            try:
                env_config, _ = super(VeRLAdapter, cls).config_init()
            except Exception:
                env_config = cls.env_config_cls() if hasattr(cls, 'env_config_cls') else BaseEnvConfig()

            model_name = os.environ.get("ATROPOS_MODEL_NAME")
            if model_name:
                env_config.tokenizer_name = model_name
            elif env_config.tokenizer_name == BaseEnvConfig().tokenizer_name:
                env_config.tokenizer_name = default_tokenizer

            server_configs = get_verl_server_configs(env_config.tokenizer_name)
            return env_config, server_configs

    VeRLAdapter.__name__ = f"VeRL{env_class.__name__}"
    VeRLAdapter.__qualname__ = f"VeRL{env_class.__name__}"
    return VeRLAdapter
