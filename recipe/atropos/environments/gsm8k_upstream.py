#!/usr/bin/env python3
"""
Wrapper for upstream Atropos GSM8K environment.

Usage:
    python -m recipe.atropos.environments.gsm8k_upstream serve --env.group_size 8
"""

import aiohttp
from atroposlib.envs.server_handling import sglang_server


async def _patched_tokens_and_logprobs(self, **kwargs):
    """Patched version that adds logprob_start_len to skip input token logprobs."""
    assert kwargs.get("model") is not None, "Model is required!"
    assert kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None, "Prompt or input_ids required!"

    if "input_ids" in kwargs:
        prompt_tokens = kwargs.pop("input_ids")
        kwargs.pop("prompt", None)
    else:
        prompt_tokens = self.tokenizer.encode(kwargs.pop("prompt"))

    if len(prompt_tokens) >= 2 and prompt_tokens[0] == self.tokenizer.bos_token_id == prompt_tokens[1]:
        prompt_tokens = prompt_tokens[1:]

    if "max_tokens" in kwargs:
        kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
    kwargs.pop("model", None)

    request_data = {
        "input_ids": prompt_tokens,
        "sampling_params": kwargs,
        "return_logprob": True,
        "logprob_start_len": max(0, len(prompt_tokens) - 1),
        "return_text_in_logprobs": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.config.base_url.replace('/v1', '')}/generate",
            json=request_data,
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as response:
            response.raise_for_status()
            results = await response.json()

    if not isinstance(results, list):
        results = [results]

    output_tokens_list = []
    output_logprobs_list = []
    finish_reasons_list = []

    for result in results:
        meta_info = result.get("meta_info", {})
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        logprobs = [item[0] for item in output_token_logprobs]
        output_ids = [item[1] for item in output_token_logprobs]
        output_tokens_list.append(output_ids)
        output_logprobs_list.append(logprobs)
        finish_reasons_list.append(meta_info.get("finish_reason", {}))

    return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons_list


sglang_server.SGLangServer._tokens_and_logprobs_completion_wrapper = _patched_tokens_and_logprobs

try:
    from environments.gsm8k_server import GSM8kEnv as _UpstreamGSM8kEnv
except ImportError:
    try:
        from atropos.environments.gsm8k_server import GSM8kEnv as _UpstreamGSM8kEnv
    except ImportError:
        raise ImportError(
            "Could not import upstream GSM8kEnv. Please either:\n"
            "1. Clone https://github.com/NousResearch/atropos and add to PYTHONPATH\n"
            "2. Or use the local copy: recipe.atropos.environments.gsm8k"
        )

from recipe.atropos.utils import create_verl_adapter

GSM8kEnv = create_verl_adapter(_UpstreamGSM8kEnv, default_tokenizer="Qwen/Qwen3-0.6B")

__all__ = ["GSM8kEnv"]

if __name__ == "__main__":
    GSM8kEnv.cli()
