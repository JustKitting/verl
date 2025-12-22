#!/usr/bin/env python3
# Copyright 2025 Nous Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GSM8K environment for Atropos training.

This is a close port of the official Atropos GSM8K environment:
https://github.com/NousResearch/atropos/blob/main/environments/gsm8k_server.py
"""

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
import logging

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse as _parse, verify as _verify
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

# Wrap math_verify functions to handle asyncio event loop issues
# math_verify internally uses asyncio.get_running_loop() which fails in certain contexts
def _ensure_event_loop():
    """Ensure there's an event loop available for math_verify."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create one for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def parse(*args, **kwargs):
    """Wrapper around math_verify.parse that ensures event loop exists."""
    _ensure_event_loop()
    try:
        return _parse(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Error parsing: {e}")
        return []  # Return empty list on parse failure

def verify(*args, **kwargs):
    """Wrapper around math_verify.verify that ensures event loop exists."""
    _ensure_event_loop()
    try:
        return _verify(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Error verifying: {e}")
        return False  # Return False on verify failure

# Patch SGLangServer to add logprob_start_len (fixes SGLang assertion error)
from atroposlib.envs.server_handling import sglang_server
import aiohttp

_original_tokens_and_logprobs = sglang_server.SGLangServer._tokens_and_logprobs_completion_wrapper

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
        "logprob_start_len": max(0, len(prompt_tokens) - 1),  # Skip most input token logprobs
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

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

ANSWER_EXTRACTION_CONFIG = LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        equations=True,
        boxed="all",
        units=True,
    ),
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""


class GSM8kRow(TypedDict):
    question: str
    answer: str


class GSM8kEnv(BaseEnv):
    """
    Atropos environment for GSM8k math problems.
    Generates rollouts, scores them, and provides training batches.
    """

    name = "gsm8k"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-0.6B",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="gsm8k",
        )

        # Check for ATROPOS_SERVER_URLS environment variable (JSON list or comma-separated)
        # Model name comes from ATROPOS_MODEL_NAME or defaults
        server_urls_env = os.environ.get("ATROPOS_SERVER_URLS", "")
        model_name = os.environ.get("ATROPOS_MODEL_NAME", "default")

        if server_urls_env:
            try:
                # Try JSON first
                urls = json.loads(server_urls_env)
            except json.JSONDecodeError:
                # Fall back to comma-separated
                urls = [u.strip() for u in server_urls_env.split(",") if u.strip()]

            if urls:
                logger.info(f"Using {len(urls)} server(s) from ATROPOS_SERVER_URLS: {urls}")
                server_configs = [
                    APIServerConfig(
                        model_name=model_name,
                        base_url=url if url.endswith("/v1") else url.rstrip("/") + "/v1",
                        api_key="dummy",
                        num_requests_for_eval=256,
                        server_type="sglang",
                    )
                    for url in urls
                ]
                return env_config, server_configs

        # No server URLs provided - fail fast with clear error
        raise ValueError(
            "ATROPOS_SERVER_URLS environment variable is required but not set.\n"
            "This environment expects VeRL trainer to pass SGLang server URLs.\n"
            "If running standalone, set: ATROPOS_SERVER_URLS='[\"http://localhost:9001/v1\"]'"
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"]
                    .split("#")[-1]
                    .strip()
                    .replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question: str, answer: str) -> dict:
        """Rollout and score evaluation with detailed sample data collection."""

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.6,
            )

            response_content = completion.choices[0].message.content

        # Parse and verify with full error handling (math_verify can have asyncio issues)
        try:
            gold_parsed = parse(
                "\\boxed{" + answer + "}",
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            answer_parsed = parse(
                response_content.split("</think>")[-1],
                extraction_config=[ANSWER_EXTRACTION_CONFIG],
                extraction_mode="first_match",
            )
            score = 1 if verify(answer_parsed, gold_parsed) else 0
        except Exception as e:
            logger.warning(f"math_verify error in eval, marking as wrong: {e}")
            score = 0
            gold_parsed = None
            answer_parsed = None

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_content},
            ],
            "question": question,
            "gold_answer": answer,
            "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
            "response_after_think": (
                response_content.split("</think>")[-1]
                if "</think>" in response_content
                else response_content
            ),
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["question"], item["gold_answer"])
            )
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores)

        end_time = time.time()

        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(
        self, item: GSM8kRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["question"]}
        gold_answer = (
            "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"
        )

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[{"role": "system", "content": system_prompt}, user_message],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()
        to_backlog = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "gold_answer": gold_answer,
                    "finish_reason": chat_completion.finish_reason,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()
        try:
            gold_parsed = parse(
                rollout_group_data[0]["gold_answer"],
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
        except Exception as e:
            logger.warning(f"math_verify error parsing gold answer, skipping batch: {e}")
            return None
        if len(gold_parsed) != 0:
            random.shuffle(rollout_group_data)
            filtered_count = 0
            for item in rollout_group_data:
                try:
                    answer_parsed = parse(
                        item["messages"][-1]["content"].split("</think>")[-1],
                        extraction_config=[ANSWER_EXTRACTION_CONFIG],
                        extraction_mode="first_match",
                    )
                    reward = verify(answer_parsed, gold_parsed)
                except Exception as e:
                    logger.warning(f"math_verify error, marking as wrong: {e}")
                    reward = False

                tokens = item["tokens"]
                masks = item["masks"]
                logprobs = item["logprobs"]

                # Remove obviously bad examples (very short responses)
                response_len = len([1 for i in masks if i != -100])
                if response_len < 10:
                    filtered_count += 1
                    continue
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["inference_logprobs"].append(logprobs)
                scores["scores"].append(1.0 if reward else -1.0)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            if filtered_count > 0:
                logger.warning(
                    f"Filtered {filtered_count}/{len(rollout_group_data)} examples with <10 response tokens"
                )

            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))

            # Length penalty when all responses are correct
            if all([score == 1 for score in scores["scores"]]):
                token_lengths = [len(token) for token in scores["tokens"]]
                if max(token_lengths) == 0:
                    return None

                max_allowed_length = self.config.max_token_length
                length_threshold = max_allowed_length * 0.5

                scores["scores"] = []
                for length in token_lengths:
                    if length <= length_threshold:
                        scores["scores"].append(1.0)
                    else:
                        percentage_of_range = (length - length_threshold) / (
                            max_allowed_length - length_threshold
                        )
                        percentage_of_range = min(percentage_of_range, 1.0)
                        scores["scores"].append(1.0 - percentage_of_range)

            if all([scores["scores"][0] == score for score in scores["scores"]]):
                return None
            return scores
        else:
            return None

    async def get_next_item(self) -> GSM8kRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GSM8kEnv.cli()
