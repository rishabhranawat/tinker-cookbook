# Reasoning RL: Teaching Non-Reasoning Models to Think Step-by-Step

This recipe teaches **Llama-3.2-1B** (a small, non-reasoning model) to use explicit step-by-step reasoning with structured XML formatting.

## Goal

Transform a standard language model into one that:
1. **Breaks down problems** step-by-step
2. **Shows its work** in a structured format
3. **Provides correct answers** after reasoning

## Output Format

The model learns to produce responses in this format:

```
<start_reasoning>
Let me work through this step-by-step:
1. First, I need to find the total...
2. Then I calculate...
3. Finally, I subtract...
</start_reasoning>
<start_answer>
\boxed{42}
</start_answer>
```

## Architecture

### Components

1. **ReasoningEnv** (`reasoning_env.py`)
   - Custom RL environment that enforces the XML format
   - Checks for `<start_reasoning>` and `<start_answer>` tags
   - Extracts and grades the final answer from `\boxed{}`
   - Provides binary reward: +1 for correct answer in proper format, 0 otherwise

2. **ReasoningGsm8kDatasetBuilder** (`reasoning_env.py`)
   - Loads GSM8K math word problems
   - Creates groups of 8 solution attempts per problem
   - Includes a few-shot example showing the expected format

3. **Training Script** (`train_reasoning.py`)
   - Configures PPO-based RL training
   - Uses group_size=8 for advantage estimation
   - Trains with LoRA adapters (rank 32)

### Key Design Decisions

**Why PPO?**
- More stable than importance sampling for structured output tasks
- Clips policy updates to prevent catastrophic forgetting of the format

**Why group_size=8?**
- Generates 8 different solution attempts per problem
- Advantages are computed by ranking these 8 solutions
- Balances exploration (trying different reasoning paths) with computational cost

**Why format_coef=0.0?**
- No partial reward for just using the format correctly
- Only correct answers get reward (+1)
- Forces the model to learn meaningful reasoning, not just template filling

## Usage

### Basic Training

```bash
python tinker_cookbook/recipes/reasoning_rl/train_reasoning.py
```

This will:
- Train on GSM8K math problems
- Save checkpoints to `~/tinker-logs/reasoning-rl`
- Evaluate every 10 batches
- Log trajectory samples for inspection

### Configuration Options

Override any config parameter via command line:

```bash
# Adjust learning rate
python train_reasoning.py --learning_rate 1e-4

# Change batch size and group size
python train_reasoning.py --batch_size 32 --group_size 16

# Use importance sampling instead of PPO
python train_reasoning.py --loss_fn importance_sampling

# Train for fewer batches (quick test)
python train_reasoning.py --batch_size 4 --eval_every 2
```

### Monitor Training

Logs are saved to `~/tinker-logs/reasoning-rl/`:
- `metrics.jsonl` - Training metrics (reward, KL divergence, etc.)
- `train_iteration_XXXXXX.html` - Trajectory visualizations
- `checkpoints/` - Model weights for resuming or deployment

## How It Works

### Training Loop (One Iteration)

1. **Sample Phase**
   - Get 64 math problems (batch_size=64)
   - For each problem, generate 8 solution attempts (group_size=8)
   - Total: 64 × 8 = 512 rollouts per iteration

2. **Reward Computation**
   - Check each solution's format (has required XML tags?)
   - Extract answer from `<start_answer>` section
   - Grade answer against ground truth
   - Reward: +1 if correct, 0 otherwise

3. **Advantage Estimation**
   - Within each group of 8 solutions for the same problem:
     - Rank solutions by reward
     - Compute advantages (zero-centered within group)
     - Solutions better than the group average get positive advantage

4. **Training Step**
   - Use PPO loss to update the model
   - Increase probability of high-advantage solutions
   - Decrease probability of low-advantage solutions
   - Clip updates to maintain stability

5. **Checkpointing**
   - Create new sampling client from updated weights
   - Save checkpoint every 20 iterations

### Reward Function Details

The reward function in `ReasoningEnv.step()`:

```python
def step(self, action: Action) -> StepResult:
    # Parse the model's response
    message, parse_success = self.renderer.parse_response(action)
    content = renderers.get_text_content(message)

    # Check format (XML tags + \boxed{})
    correct_format = self.check_format(content)

    # Check answer correctness
    correct_answer = self.check_answer(content)

    # Binary reward: only correct answers get +1
    total_reward = float(correct_answer)

    return StepResult(reward=total_reward, episode_done=True, ...)
```

### Format Checking Logic

The `check_format()` method validates:

```python
def check_format(self, sample_str: str) -> bool:
    # Must have <start_reasoning>...</start_reasoning>
    reasoning_match = re.search(r"<start_reasoning>(.*?)</start_reasoning>", sample_str, re.DOTALL)
    if not reasoning_match:
        return False

    # Must have <start_answer>...</start_answer>
    answer_match = re.search(r"<start_answer>(.*?)</start_answer>", sample_str, re.DOTALL)
    if not answer_match:
        return False

    # Must have \boxed{} inside answer section
    answer_content = answer_match.group(1)
    try:
        extract_boxed(answer_content)
        return True
    except ValueError:
        return False
```

## Expected Results

After training, the model should:

1. **Always use the XML format** (learned from the reward structure)
2. **Show reasoning steps** in the `<start_reasoning>` section
3. **Improve answer accuracy** on GSM8K (baseline ~0% → target ~30-50%)
4. **Generalize the reasoning pattern** to new math problems

### Sample Output (After Training)

**Problem:**
> Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

**Model Response:**
```
<start_reasoning>
Let me solve this step-by-step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans, and each can has 3 balls
3. So he gets 2 × 3 = 6 new tennis balls
4. Total tennis balls = 5 (original) + 6 (new) = 11 tennis balls
</start_reasoning>
<start_answer>
\boxed{11}
</start_answer>
```

## Troubleshooting

### Model doesn't learn the format

- **Check few-shot example**: Make sure `convo_prefix="standard"` is set
- **Increase format_coef**: Try `format_coef=0.1` for partial reward on format
- **Lower learning rate**: Try `learning_rate=1e-5` for more gradual learning

### Model learns format but gives wrong answers

- **This is expected early in training** - format is easier to learn than reasoning
- **Increase training time**: Run more batches
- **Try different group_size**: Larger groups (16) provide stronger signal

### Training is too slow

- **Reduce batch_size**: Try `batch_size=32` instead of 64
- **Reduce max_tokens**: Try `max_tokens=256` instead of 512
- **Reduce group_size**: Try `group_size=4` (but may hurt sample efficiency)

### Out of memory errors

- **Reduce batch_size**: The main lever for memory usage
- **Reduce max_tokens**: Shorter sequences use less memory
- **Use num_substeps**: Add `num_substeps=2` to split batch into smaller chunks

## Next Steps

### Extend to Other Tasks

Replace `ReasoningGsm8kDatasetBuilder` with your own dataset:

```python
class CustomReasoningDataset(RLDataset):
    def __init__(self, ...):
        # Load your custom dataset
        self.ds = load_dataset("your/dataset")
        ...

    def _make_env_group_builder(self, x, group_size):
        return ProblemGroupBuilder(
            env_thunk=partial(
                ReasoningEnv,
                problem=x["question"],
                answer=x["answer"],
                self.renderer,
                convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )
```

### Improve Reward Function

Add reasoning quality scoring:

```python
def step(self, action: Action) -> StepResult:
    # ... existing code ...

    # Bonus for showing multiple reasoning steps
    reasoning_content = extract_reasoning(content)
    num_steps = len(re.findall(r'^\d+\.', reasoning_content, re.MULTILINE))
    step_bonus = min(num_steps * 0.1, 0.5)  # Cap at +0.5

    total_reward = correct_answer + step_bonus
    return StepResult(reward=total_reward, ...)
```

### Multi-Turn Reasoning

Enable the model to refine its reasoning across multiple turns (see `docs/rl/sequence-extension.mdx`).

## Implementation Details

### File Structure

```
tinker_cookbook/recipes/reasoning_rl/
├── __init__.py                 # Module initialization
├── reasoning_env.py            # Core environment and dataset
├── train_reasoning.py          # Main training script
└── README.md                   # This file
```

### Dependencies

All dependencies are already in `tinker-cookbook`:
- `tinker` - API client for Tinker service
- `datasets` - HuggingFace datasets (GSM8K)
- `chz` - Configuration system
- Existing math grading utilities

### Key Type Signatures

```python
class ReasoningEnv(ProblemEnv):
    async def initial_observation(self) -> tuple[Observation, StopCondition]
    async def step(self, action: Action) -> StepResult
    def check_format(self, sample_str: str) -> bool
    def check_answer(self, sample_str: str) -> bool

class ReasoningGsm8kDatasetBuilder(RLDatasetBuilder):
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]
```

## References

- **GSM8K**: [Grade School Math 8K dataset](https://github.com/openai/grade-school-math)
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **Tinker docs**: See `docs/rl/rl-basic.mdx` for RL fundamentals
