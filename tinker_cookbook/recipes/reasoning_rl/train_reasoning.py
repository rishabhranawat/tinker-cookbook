"""
Train Llama-3.2-1B to use step-by-step reasoning with XML formatting.

This recipe teaches a non-reasoning model to:
1. Break down problems step-by-step
2. Format output with <start_reasoning>...</start_reasoning> and <start_answer>...</start_answer> tags
3. Provide correct answers after showing reasoning

Uses PPO loss with group_size=8 for on-policy RL training.
"""
import asyncio
import sys

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.reasoning_rl.reasoning_env import ReasoningGsm8kDatasetBuilder
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """
    Build the training configuration for reasoning RL.

    Key settings:
    - Model: meta-llama/Llama-3.2-1B (small non-reasoning model)
    - Loss: PPO (more stable than importance sampling for this task)
    - Group size: 8 (generates 8 solution attempts per problem)
    - Batch size: 64 (64 different problems per batch)
    - Learning rate: 5e-5 (slightly higher for small model + LoRA)
    """
    model_name = "meta-llama/Llama-3.2-1B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Create dataset builder with group_size=8
    dataset_builder = ReasoningGsm8kDatasetBuilder(
        batch_size=64,          # Number of different problems per batch
        group_size=8,           # Generate 8 attempts per problem (for advantage estimation)
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        convo_prefix="standard",  # Use the few-shot example
        seed=42,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "~/tinker-logs/reasoning-rl",
            "dataset_builder": dataset_builder,

            # Optimization settings
            "learning_rate": 5e-5,
            "loss_fn": "ppo",  # Use PPO instead of importance sampling

            # Generation settings
            "max_tokens": 512,  # Allow longer responses for reasoning
            "temperature": 1.0,  # Default sampling temperature

            # LoRA settings
            "lora_rank": 32,  # Standard LoRA rank

            # Evaluation and checkpointing
            "eval_every": 10,   # Evaluate every 10 batches
            "save_every": 20,   # Save checkpoint every 20 batches

            # Logging
            "num_groups_to_log": 4,  # Log 4 trajectory groups per iteration
        }
    )


def main(config: train.Config):
    """Run the reasoning RL training."""
    # Check if log directory exists - ask before overwriting
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Run the async training loop
    asyncio.run(train.main(config))


if __name__ == "__main__":
    # Build config blueprint (allows CLI overrides)
    blueprint = build_config_blueprint()

    # Parse command-line arguments to override config
    # Example: python train_reasoning.py --learning_rate 1e-4 --batch_size 32
    blueprint.make_from_argv(sys.argv[1:])

    # Run training
    main(blueprint.make())
