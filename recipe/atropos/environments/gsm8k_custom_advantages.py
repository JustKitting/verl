# Copyright 2025 Nous Research
# GSM8K environment with custom per-token advantages for answer formatting
"""
GSM8K environment that adds custom advantages to answer formatting tokens.

This demonstrates how to use per-token advantages to teach the model
specific formatting patterns faster - in this case, the boxed answer format.
"""

import logging
import re
from typing import List, Optional, Union

from .gsm8k import GSM8kEnv, ScoredDataGroup

logger = logging.getLogger(__name__)


class GSM8KCustomAdvantagesEnvironment(GSM8kEnv):
    """
    GSM8K environment with custom per-token advantages.

    Adds positive advantages to tokens that are part of the answer box format,
    encouraging the model to learn the correct output style faster.
    """

    # Tokens/patterns to boost (will be matched against decoded text)
    # These are common answer box patterns in GSM8K
    BOOST_PATTERNS = [
        r"\\boxed\{",      # LaTeX boxed
        r"\\boxed",        # boxed command
        r"\}",             # closing brace (part of boxed)
        r"####",           # GSM8K answer delimiter
        r"The answer is",  # Common answer prefix
        r"Answer:",        # Answer label
    ]

    # Advantage boost for matching tokens
    ADVANTAGE_BOOST = 0.1

    # Base advantage for all response tokens (can be 0)
    BASE_ADVANTAGE = 0.0

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score rollouts and add custom per-token advantages."""

        # Get base scores from parent
        scores = await super().score(rollout_group_data)

        if scores is None:
            return None

        # Add per-token advantages
        scores["advantages"] = []

        for i, (tokens, masks) in enumerate(zip(scores["tokens"], scores["masks"])):
            advantages = self._compute_token_advantages(tokens, masks)
            scores["advantages"].append(advantages)

        # Log some stats
        if scores["advantages"]:
            boosted_counts = [sum(1 for a in adv if a > self.BASE_ADVANTAGE)
                           for adv in scores["advantages"]]
            avg_boosted = sum(boosted_counts) / len(boosted_counts)
            logger.debug(f"Average tokens boosted per example: {avg_boosted:.1f}")

        return scores

    def _compute_token_advantages(
        self,
        tokens: List[int],
        masks: List[int]
    ) -> List[float]:
        """
        Compute per-token advantages.

        Boosts tokens that match answer formatting patterns.

        Args:
            tokens: Token IDs for the full sequence
            masks: -100 for prompt tokens, token_id for response tokens

        Returns:
            List of advantage values, same length as tokens
        """
        advantages = []

        # Decode tokens to find patterns
        try:
            decoded = self.tokenizer.decode(tokens)
        except Exception:
            # If decode fails, return base advantages
            return [self.BASE_ADVANTAGE if m != -100 else 0.0 for m in masks]

        # Find positions of boost patterns in decoded text
        boost_positions = set()
        for pattern in self.BOOST_PATTERNS:
            for match in re.finditer(pattern, decoded, re.IGNORECASE):
                # Mark character positions
                for pos in range(match.start(), match.end()):
                    boost_positions.add(pos)

        # Map character positions back to token positions
        # This is approximate but works reasonably well
        char_pos = 0
        for idx, (token_id, mask) in enumerate(zip(tokens, masks)):
            if mask == -100:
                # Prompt token - no advantage
                advantages.append(0.0)
            else:
                # Response token - check if any chars in this token should be boosted
                try:
                    token_text = self.tokenizer.decode([token_id])
                    token_len = len(token_text)
                except Exception:
                    token_len = 1

                # Check if any character in this token's range is boosted
                should_boost = any(
                    pos in boost_positions
                    for pos in range(char_pos, char_pos + token_len)
                )

                if should_boost:
                    advantages.append(self.BASE_ADVANTAGE + self.ADVANTAGE_BOOST)
                else:
                    advantages.append(self.BASE_ADVANTAGE)

                char_pos += token_len

        return advantages


# For CLI compatibility
Environment = GSM8KCustomAdvantagesEnvironment

if __name__ == "__main__":
    GSM8KCustomAdvantagesEnvironment.cli()
