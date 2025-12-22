# Copyright 2025 Anthropic
# Custom FSDP workers for Atropos with weight sync support

import asyncio
import concurrent.futures
import logging
import time

from verl import DataProto
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__name__)


class AtroposActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """
    Extended AsyncActorRolloutRefWorker with weight sync support for Atropos.

    In async rollout mode, the SGLang inference servers need weight updates
    periodically. Sync is controlled by meta_info["do_sync"] in the data.
    """

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Update actor weights and optionally sync to inference.

        If data.meta_info["do_sync"] is True, syncs weights to vLLM after training.
        Sync timing metrics are added to the output meta_info.
        """
        output = super().update_actor(data)

        # Check if trainer wants us to sync
        do_sync = data.meta_info.get("do_sync", False)

        if do_sync and self._is_actor and hasattr(self, 'rollout') and self.rollout is not None:
            sync_metrics = self._sync_weights_to_inference()
            # Add sync metrics to output
            if "metrics" not in output.meta_info:
                output.meta_info["metrics"] = {}
            output.meta_info["metrics"].update(sync_metrics)

        return output

    def _sync_weights_to_inference(self) -> dict:
        """
        Sync trained weights to vLLM inference engine.

        Returns:
            dict with sync timing metrics
        """
        metrics = {}

        logger.info("Syncing weights to vLLM rollout engine...")
        sync_start = time.time()
        log_gpu_memory_usage("Before weight sync", logger=logger)

        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - run sync in a thread pool
            logger.debug("Running sync in thread pool (async context detected)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_sync_coroutines)
                rollout_time, trainer_time = future.result(timeout=300)
        except RuntimeError:
            # No running loop - safe to use run_until_complete
            logger.debug("Running sync directly (no async context)")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                rollout_start = time.time()
                loop.run_until_complete(self.rollout_mode())
                rollout_time = time.time() - rollout_start

                trainer_start = time.time()
                loop.run_until_complete(self.trainer_mode())
                trainer_time = time.time() - trainer_start
            finally:
                loop.close()

        metrics["sync/rollout_mode_time"] = rollout_time
        metrics["sync/trainer_mode_time"] = trainer_time
        log_gpu_memory_usage("After sync", logger=logger)

        get_torch_device().empty_cache()

        total_sync_time = time.time() - sync_start
        metrics["sync/total_time"] = total_sync_time
        logger.info(f"Weight sync complete in {total_sync_time:.2f}s")

        return metrics

    def _run_sync_coroutines(self) -> tuple[float, float]:
        """Run sync coroutines in a new event loop (for thread pool execution)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rollout_start = time.time()
            loop.run_until_complete(self.rollout_mode())
            rollout_time = time.time() - rollout_start

            trainer_start = time.time()
            loop.run_until_complete(self.trainer_mode())
            trainer_time = time.time() - trainer_start

            return rollout_time, trainer_time
        finally:
            loop.close()
