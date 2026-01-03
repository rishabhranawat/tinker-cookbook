"""
Evaluate a reasoning model on GSM8K test set.

This script loads a saved checkpoint and evaluates it on the GSM8K test set,
reporting accuracy and format compliance metrics.

Usage:
    # Evaluate the latest checkpoint from a training run
    python eval_reasoning.py --log_path ~/tinker-logs/reasoning-rl

    # Evaluate a specific checkpoint by name
    python eval_reasoning.py --checkpoint_name 000100

    # Evaluate the base model (before training)
    python eval_reasoning.py --model_name meta-llama/Llama-3.2-1B

    # Evaluate with custom parameters
    python eval_reasoning.py --log_path ~/tinker-logs/reasoning-rl --num_problems 100 --temperature 0.7
"""
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from typing import cast

import chz
from datasets import Dataset, load_dataset
from tqdm import tqdm

import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.reasoning_rl.reasoning_env import ReasoningEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_format(sample_str: str) -> bool:
    """Check if response has required XML format."""
    reasoning_pattern = r"<start_reasoning>(.*?)</start_reasoning>"
    reasoning_match = re.search(reasoning_pattern, sample_str, re.DOTALL)
    if not reasoning_match:
        return False

    answer_pattern = r"<start_answer>(.*?)</start_answer>"
    answer_match = re.search(answer_pattern, sample_str, re.DOTALL)
    if not answer_match:
        return False

    answer_content = answer_match.group(1)
    try:
        _ = extract_boxed(answer_content)
        return True
    except ValueError:
        return False


def extract_answer_from_xml(sample_str: str) -> str | None:
    """Extract answer from XML format, return None if invalid."""
    try:
        answer_pattern = r"<start_answer>(.*?)</start_answer>"
        answer_match = re.search(answer_pattern, sample_str, re.DOTALL)
        if not answer_match:
            return None
        return extract_boxed(answer_match.group(1))
    except ValueError:
        return None


def grade_response(response: str, ground_truth: str) -> dict[str, bool | str | None]:
    """
    Grade a model response.

    Returns:
        Dictionary with:
        - format_valid: Whether XML format is correct
        - extracted_answer: The extracted answer (or None if invalid)
        - correct: Whether the answer is correct
    """
    format_valid = check_format(response)
    extracted_answer = extract_answer_from_xml(response) if format_valid else None

    correct = False
    if extracted_answer is not None:
        try:
            correct = grade_answer(extracted_answer, ground_truth)
        except Exception as e:
            logger.debug(f"Grading error: {e}")
            correct = False

    return {
        "format_valid": format_valid,
        "extracted_answer": extracted_answer,
        "correct": correct,
    }


@dataclass
class EvalResult:
    """Results from evaluating on GSM8K."""
    total: int
    correct: int
    format_valid: int
    format_invalid: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def format_accuracy(self) -> float:
        return self.format_valid / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return f"""
Evaluation Results:
==================
Total problems: {self.total}
Correct answers: {self.correct} ({self.accuracy:.1%})
Format valid: {self.format_valid} ({self.format_accuracy:.1%})
Format invalid: {self.format_invalid}
"""


async def evaluate_model(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    num_problems: int = 100,
    max_tokens: int = 512,
    show_examples: int = 5,
) -> EvalResult:
    """
    Evaluate model on GSM8K test set.

    Args:
        sampling_client: Client to use for sampling
        renderer: Renderer for tokenization
        num_problems: Number of test problems to evaluate (default 100, max 1319)
        max_tokens: Maximum tokens to generate
        show_examples: Number of example responses to print

    Returns:
        EvalResult with accuracy metrics
    """
    # Load GSM8K test set
    logger.info("Loading GSM8K test set...")
    ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="test"))

    # Limit to num_problems
    if num_problems < len(ds):
        ds = ds.select(range(num_problems))

    logger.info(f"Evaluating on {len(ds)} problems from GSM8K test set")

    # Create completer (uses temperature=1.0)
    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
    )

    # Prepare few-shot prefix
    fewshot_prefix = ReasoningEnv.standard_fewshot_prefix()

    # Evaluate each problem
    results = []
    examples_shown = 0

    for idx, row in enumerate(tqdm(ds, desc="Evaluating")):
        problem = row["question"]
        answer_text = row["answer"]

        # Extract ground truth answer
        lines = answer_text.splitlines()
        ground_truth = None
        for line in reversed(lines):
            s = line.strip()
            if s.startswith("####"):
                ground_truth = s[4:].strip()
                if ground_truth.startswith(":"):
                    ground_truth = ground_truth[1:].strip()
                ground_truth = ground_truth.replace(",", "").strip()
                break

        if ground_truth is None:
            logger.warning(f"Problem {idx}: Could not extract ground truth")
            continue

        # Build prompt with few-shot example
        messages = fewshot_prefix + [
            {"role": "user", "content": problem + ReasoningEnv.question_suffix()}
        ]

        # Get model response
        response = await completer(messages)
        response_text = response["content"]

        # Grade response
        grade = grade_response(response_text, ground_truth)
        results.append(grade)

        # Show examples
        if examples_shown < show_examples:
            logger.info(f"\n{'='*80}")
            logger.info(f"Example {idx + 1}")
            logger.info(f"{'='*80}")
            logger.info(f"Problem: {problem}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"\nModel response:\n{response_text}")
            logger.info(f"\nGrade: {grade}")
            logger.info(f"{'='*80}\n")
            examples_shown += 1

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    format_valid = sum(1 for r in results if r["format_valid"])
    format_invalid = total - format_valid

    return EvalResult(
        total=total,
        correct=correct,
        format_valid=format_valid,
        format_invalid=format_invalid,
    )


@chz.chz
class EvalConfig:
    """Configuration for evaluation."""

    # Model specification (choose one):
    log_path: str | None = None  # Path to training run logs
    checkpoint_name: str | None = None  # Specific checkpoint name (e.g., "000100")
    model_name: str | None = None  # Base model to evaluate (e.g., "meta-llama/Llama-3.2-1B")

    # Evaluation parameters
    num_problems: int = 100  # Number of test problems to evaluate
    max_tokens: int = 512  # Maximum tokens to generate
    show_examples: int = 5  # Number of example responses to print

    # Advanced
    base_url: str | None = None  # Tinker API base URL


async def main(config: EvalConfig):
    """Run evaluation."""

    # Determine which model to load
    service_client = tinker.ServiceClient(base_url=config.base_url)

    if config.log_path:
        # Load from checkpoint
        import os
        log_path = os.path.expanduser(config.log_path)

        if config.checkpoint_name:
            # Load specific checkpoint
            checkpoint_path = os.path.join(
                log_path, "checkpoints.jsonl"
            )
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"No checkpoints found at {checkpoint_path}")

            checkpoints = checkpoint_utils.load_checkpoints_file(log_path)
            matching = [c for c in checkpoints if c.get("name") == config.checkpoint_name]
            if not matching:
                raise ValueError(
                    f"Checkpoint '{config.checkpoint_name}' not found. "
                    f"Available: {[c.get('name') for c in checkpoints]}"
                )
            checkpoint = matching[0]
        else:
            # Load latest checkpoint
            checkpoint = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
            if checkpoint is None:
                raise ValueError(f"No valid checkpoints found in {log_path}")

        sampler_path = checkpoint["sampler_path"]
        logger.info(f"Loading checkpoint from: {sampler_path}")
        sampling_client = service_client.create_sampling_client(sampler_path)

        # Get model name from checkpoint metadata (if available)
        # For now, default to Llama-3.2-1B
        model_name = config.model_name or "meta-llama/Llama-3.2-1B"

    elif config.model_name:
        # Load base model
        logger.info(f"Loading base model: {config.model_name}")
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)
        model_name = config.model_name

    else:
        raise ValueError(
            "Must specify either --log_path (to load checkpoint) or "
            "--model_name (to load base model)"
        )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Run evaluation
    logger.info("Starting evaluation...")
    result = await evaluate_model(
        sampling_client=sampling_client,
        renderer=renderer,
        num_problems=config.num_problems,
        max_tokens=config.max_tokens,
        show_examples=config.show_examples,
    )

    # Print results
    print(result)


if __name__ == "__main__":
    # Parse command-line arguments
    blueprint = chz.Blueprint(EvalConfig)
    blueprint.make_from_argv(sys.argv[1:])
    config = blueprint.make()

    # Run evaluation
    asyncio.run(main(config))
