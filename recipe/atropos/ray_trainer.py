# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Atropos-integrated PPO/GRPO Trainer for VeRL.

This trainer pulls pre-generated and pre-scored rollout data from the Atropos
Trajectory API instead of generating rollouts internally. This enables
integration with Atropos's multi-environment RL framework.

Key differences from standard RayPPOTrainer:
1. Data comes from Atropos API (GET /batch) instead of internal dataloader
2. Rollout generation is skipped (environments do this)
3. Reward computation is skipped (environments score the data)
4. Token-level advantages from Atropos are used if provided
5. Weight sync is controlled by queue staleness (not every step)
"""

import logging
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss

from .utils.debug import debug_batch_data, save_batch_tensors

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from .data_source import AtroposDataSource

logger = logging.getLogger(__name__)


def compute_advantage_with_atropos_override(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    """
    Compute advantage, but use Atropos-provided advantages if available.

    If data.batch contains "atropos_advantages", ADD those to the computed
    GRPO advantages. This allows environments to provide formatting bonuses
    on top of the correctness signal.
    """
    # First compute GRPO advantages normally (captures correctness)
    data = compute_advantage(
        data=data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    # Add Atropos formatting bonuses on top of GRPO advantages
    if "atropos_advantages" in data.batch.keys():
        data.batch["advantages"] = data.batch["advantages"] + data.batch["atropos_advantages"]

    return data


class RayAtroposTrainer(RayPPOTrainer):
    """
    PPO/GRPO Trainer that integrates with Atropos Trajectory API.

    This trainer:
    1. Registers with the Atropos Trajectory API
    2. Pulls pre-scored batches from the API (instead of generating internally)
    3. Trains using GRPO (or other advantage estimators)
    4. Updates inference weights periodically

    The inference servers (SGLang) are still managed by VeRL and their
    endpoints should be provided to Atropos environments.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        atropos_config: dict = None,
        **kwargs,
    ):
        """
        Args:
            config: VeRL configuration
            tokenizer: Tokenizer for the model
            role_worker_mapping: Mapping of roles to workers
            resource_pool_manager: Ray resource pool manager
            atropos_config: Atropos-specific configuration:
                - api_url: URL of Atropos Trajectory API
                - batch_timeout: Timeout for fetching batches
                - register_kwargs: Arguments for API registration
                - sync: Weight sync control settings
        """
        # Store before super().__init__ (needed in _create_dataloader)
        self._atropos_config = atropos_config or {}

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs,
        )

        env_cfg = config.get("env", {})

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer must have pad_token_id set. "
                "Set tokenizer.pad_token_id or use a tokenizer with padding configured."
            )

        self.atropos_source = AtroposDataSource(
            api_url=self._atropos_config["api_url"],
            pad_token_id=tokenizer.pad_token_id,
            max_seq_len=env_cfg.max_token_length,
            truncation_policy=config.data.get("truncation", "error"),
        )
        self.atropos_batch_timeout = self._atropos_config["batch_timeout"]
        self.atropos_register_kwargs = self._atropos_config.get("register_kwargs", {})
        self._inference_urls = []
        self._env_process = None
        self._env_log_file = None

        sync_config = self._atropos_config.get("sync", {})
        self.sync_queue_threshold = sync_config.get("queue_threshold", 1)
        self.sync_max_steps = sync_config.get("max_steps_between_sync", 4)
        self.sync_min_steps = sync_config.get("min_steps_between_sync", 1)
        self.sync_log_drift = sync_config.get("log_drift", True)

        self._steps_since_sync = 0
        self._last_sync_step = 0
        self._total_syncs = 0

        debug_config = self._atropos_config.get("debug", {})
        self._debug_enabled = debug_config.get("enabled", False)
        self._debug_output_dir = debug_config.get("output_dir", "./logs")
        self._debug_save_tensors_at_steps = set(debug_config.get("save_tensors_at_steps", []))

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Override to skip file-based dataset loading (data comes from Trajectory API)."""
        self.train_dataset = None
        self.val_dataset = val_dataset
        self.train_dataloader = None
        self.val_dataloader = None

        if self.config.trainer.total_training_steps is None:
            raise ValueError(
                "trainer.total_training_steps must be set for Atropos integration. "
                "Unlike standard training, Atropos doesn't have a finite dataset to derive step count from."
            )
        self.total_training_steps = self.config.trainer.total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

    def _save_checkpoint(self):
        """Override to handle None train_dataloader."""
        class DummyDataloader:
            def state_dict(self):
                return {"atropos": True, "step": 0}

        original_dataloader = self.train_dataloader
        self.train_dataloader = DummyDataloader()
        try:
            super()._save_checkpoint()
        finally:
            self.train_dataloader = original_dataloader

    def _discover_inference_urls(self) -> list[str]:
        """
        Discover all inference URLs from VeRL's async rollout manager.

        Returns list of URLs where SGLang servers are running.

        Raises RuntimeError if URLs cannot be discovered.
        """
        if not getattr(self, 'async_rollout_mode', False):
            raise RuntimeError(
                "Atropos integration requires async rollout mode. "
                "Set actor_rollout_ref.rollout.mode='async' in your config."
            )

        if hasattr(self, 'async_rollout_manager') and self.async_rollout_manager is not None:
            server_addrs = getattr(self.async_rollout_manager, 'server_addresses', None)
            if server_addrs and len(server_addrs) > 0:
                urls = [f"http://{addr}/v1" for addr in server_addrs]
                logger.info(f"Discovered {len(urls)} inference URL(s): {urls}")
                return urls

        raise RuntimeError(
            "Could not discover inference URLs from async_rollout_manager. "
            "Ensure SGLang started correctly."
        )

    def _register_with_atropos(self):
        """Register this trainer with the Atropos Trajectory API."""
        env_cfg = self.config.get("env", {})

        register_kwargs = {
            "wandb_group": self.config.trainer.experiment_name,
            "wandb_project": self.config.trainer.project_name,
            **self.atropos_register_kwargs,
        }

        self.atropos_source.register(
            batch_size=env_cfg.batch_size,
            max_token_len=env_cfg.max_token_length,
            num_steps=env_cfg.total_steps,
            starting_step=self.global_steps,
            **register_kwargs,
        )

    def _start_environment(self):
        """Start the Atropos environment as a subprocess."""
        import json
        import os
        import subprocess
        import sys

        # Read from env.* config section (matches native Atropos format)
        env_cfg = self.config.get("env", {})

        env_module = self._atropos_config.get("environment_module")
        if not env_module:
            raise ValueError(
                "atropos.environment_module is required but not set in config.\n"
                "Set: atropos.environment_module: 'recipe.atropos.environments.gsm8k_upstream'"
            )

        logger.info(f"Starting environment: module={env_module}, batch_size={env_cfg.batch_size}, group_size={env_cfg.group_size}")

        proc_env = os.environ.copy()
        proc_env["ATROPOS_SERVER_URLS"] = json.dumps(self._inference_urls)
        proc_env["ATROPOS_MODEL_NAME"] = env_cfg.tokenizer_name

        # Pass env.* config as CLI args (native Atropos format)
        cmd = [
            sys.executable, "-m", env_module, "serve",
            "--env.rollout_server_url", str(env_cfg.rollout_server_url),
            "--env.tokenizer_name", str(env_cfg.tokenizer_name),
            "--env.group_size", str(env_cfg.group_size),
            "--env.batch_size", str(env_cfg.batch_size),
            "--env.total_steps", str(env_cfg.total_steps),
            "--env.steps_per_eval", str(env_cfg.steps_per_eval),
            "--env.max_token_length", str(env_cfg.max_token_length),
            "--env.use_wandb", str(env_cfg.use_wandb),
            "--env.wandb_name", str(env_cfg.wandb_name),
        ]

        # Redirect subprocess output to log file (not pipe - pipes can block)
        log_dir = self._debug_output_dir or "./logs"
        env_log_path = os.path.join(log_dir, "gsm8k_env.log")
        os.makedirs(log_dir, exist_ok=True)

        self._env_log_file = open(env_log_path, "w")
        self._env_process = subprocess.Popen(
            cmd,
            env=proc_env,
            stdout=self._env_log_file,
            stderr=subprocess.STDOUT,
        )
        logger.info(f"GSM8k environment started as subprocess (PID: {self._env_process.pid}, log: {env_log_path})")

    def _cleanup_environment(self):
        """Clean up the environment subprocess and log file."""
        if self._env_process is not None:
            if self._env_process.poll() is None:
                logger.info(f"Terminating environment subprocess (PID: {self._env_process.pid})...")
                self._env_process.terminate()
                try:
                    self._env_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Environment subprocess did not terminate, killing...")
                    self._env_process.kill()
            self._env_process = None

        if self._env_log_file is not None:
            self._env_log_file.close()
            self._env_log_file = None

    def _should_sync_weights(self) -> tuple[bool, dict]:
        """
        Determine if we should sync weights to inference based on queue staleness.

        Returns:
            (should_sync, metrics_dict) - whether to sync and staleness metrics
        """
        metrics = {}
        status = self.atropos_source.get_status()
        queue_size = status.get("queue_size", 0)
        current_step = status.get("current_step", 0)

        metrics["staleness/queue_size"] = queue_size
        metrics["staleness/api_step"] = current_step
        metrics["staleness/steps_since_sync"] = self._steps_since_sync
        metrics["staleness/total_syncs"] = self._total_syncs

        if self._steps_since_sync < self.sync_min_steps:
            return False, metrics

        # Queue low = data is fresh, good time to sync
        if queue_size <= self.sync_queue_threshold:
            metrics["staleness/sync_reason"] = "queue_low"
            return True, metrics

        # Fallback: sync if too long since last sync
        if self._steps_since_sync >= self.sync_max_steps:
            metrics["staleness/sync_reason"] = "max_steps"
            return True, metrics

        return False, metrics

    def _compute_drift_metrics(self, batch: DataProto, computed_log_probs: torch.Tensor) -> dict:
        """
        Compute drift between rollout policy and current policy.

        Args:
            batch: Batch with rollout_log_probs from inference
            computed_log_probs: Log probs from current trained policy

        Returns:
            dict with drift metrics
        """
        metrics = {}

        if "rollout_log_probs" not in batch.batch.keys():
            return metrics

        rollout_lp = batch.batch["rollout_log_probs"]
        current_lp = computed_log_probs
        response_mask = batch.batch["response_mask"]

        lp_diff = (current_lp - rollout_lp).abs()

        if response_mask.sum() > 0:
            drift = (lp_diff * response_mask).sum() / response_mask.sum()
            metrics["drift/avg_logprob_diff"] = drift.item()

            log_ratio = current_lp - rollout_lp
            ratio = torch.exp(log_ratio.clamp(-10, 10))
            avg_ratio = (ratio * response_mask).sum() / response_mask.sum()
            metrics["drift/avg_importance_ratio"] = avg_ratio.item()

            max_diff = (lp_diff * response_mask).max()
            metrics["drift/max_logprob_diff"] = max_diff.item()

        return metrics

    def _update_sync_state(self):
        """Update sync tracking state after a sync."""
        self._steps_since_sync = 0
        self._last_sync_step = self.global_steps
        self._total_syncs += 1

    def _load_checkpoint(self):
        """
        Override parent _load_checkpoint to handle missing dataloader.

        The Atropos trainer doesn't use a traditional dataloader - data comes
        from the Atropos Trajectory API. So we skip dataloader state restoration.
        """
        import os

        from verl.utils.checkpoint.checkpoint_handler import find_latest_ckpt_path

        if self.config.trainer.resume_mode == "disable":
            return 0

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Resuming from step {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        # Atropos uses external API for data, no dataloader state to restore
        print("Skipping dataloader state restoration")

    def fit(self):
        """
        The training loop for Atropos-integrated training.

        Unlike standard PPO training, this loop:
        1. Pulls batches from Atropos API (not internal dataloader)
        2. Skips rollout generation (done by Atropos environments)
        3. Skips reward computation (done by Atropos environments)
        4. Uses Atropos advantages if provided
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._inference_urls = self._discover_inference_urls()
        self._register_with_atropos()
        self._start_environment()
        self._force_initial_sync = True

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Atropos Training",
        )

        self.global_steps += 1
        timing_raw = defaultdict(float)

        try:
            while self.global_steps <= self.total_training_steps:
                metrics = {}

                with marked_timer("fetch_batch", timing_raw, color="cyan"):
                    batch = self.atropos_source.get_batch(timeout=self.atropos_batch_timeout)
                    if batch is None:
                        time.sleep(0.5)
                        continue

                if self._debug_enabled:
                    debug_batch_data(batch, self.global_steps, "after_fetch", self._debug_output_dir)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if "token_level_scores" not in batch.batch.keys():
                        raise ValueError("Atropos batch missing 'token_level_scores'")
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if "rollout_log_probs" not in batch.batch.keys():
                            raise ValueError(
                                "Atropos batch missing 'rollout_log_probs'. "
                                "Ensure your environment captures inference_logprobs during generation."
                            )
                        computed_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = computed_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, color="green"):
                        batch = compute_advantage_with_atropos_override(
                            data=batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self._debug_enabled:
                        debug_batch_data(batch, self.global_steps, "after_advantage", self._debug_output_dir)

                    if self.sync_log_drift:
                        drift_metrics = self._compute_drift_metrics(
                            batch, computed_log_prob.batch["old_log_probs"]
                        )
                        metrics.update(drift_metrics)

                    self._steps_since_sync += 1
                    should_sync, staleness_metrics = self._should_sync_weights()
                    metrics.update(staleness_metrics)

                    if self._force_initial_sync:
                        should_sync = True
                        self._force_initial_sync = False
                        logger.info("Forcing initial weight sync...")

                    batch.meta_info["do_sync"] = should_sync

                    if self._debug_enabled:
                        debug_batch_data(batch, self.global_steps, "before_update", self._debug_output_dir)
                        if self.global_steps in self._debug_save_tensors_at_steps:
                            save_batch_tensors(batch, self.global_steps, self._debug_output_dir)

                    with marked_timer("update_actor", timing_raw, color="magenta"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if should_sync:
                        self._update_sync_state()

                data_metrics = compute_data_metrics(batch=batch, use_critic=False)
                metrics.update(data_metrics)

                timing_metrics = compute_timing_metrics(batch, timing_raw)
                metrics.update(timing_metrics)

                tracker.log(data=metrics, step=self.global_steps)

                if self.global_steps % self.config.trainer.save_freq == 0 or is_last_step:
                    self._save_checkpoint()

                if self.val_reward_fn is not None and self.global_steps % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        tracker.log(data=val_metrics, step=self.global_steps)

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": metrics.get("actor/loss", 0)})
                self.global_steps += 1
                timing_raw.clear()

            logger.info(f"Training complete. Total syncs performed: {self._total_syncs}")
        finally:
            self._cleanup_environment()
            progress_bar.close()
