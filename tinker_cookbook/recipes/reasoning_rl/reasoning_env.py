"""
Environment for teaching models to use step-by-step reasoning with XML formatting.

The model must produce output in this format:
<start_reasoning>
[step-by-step reasoning here]
</start_reasoning>
<start_answer>
[final answer here]
</start_answer>
"""
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class ReasoningEnv(ProblemEnv):
    """
    Environment that requires the model to produce reasoning before answering.

    Expected format:
    <start_reasoning>
    Let me think step by step...
    </start_reasoning>
    <start_answer>
    \boxed{42}
    </start_answer>
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        timeout: float = 1.0,
    ):
        # format_coef=0.0 means no partial reward for format - only correct answers get reward
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.problem = problem
        self.answer = answer
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return (
            " Think step-by-step and format your response as:\n"
            "<start_reasoning>\n"
            "[your reasoning here]\n"
            "</start_reasoning>\n"
            "<start_answer>\n"
            "\\boxed{[your answer]}\n"
            "</start_answer>"
        )

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """
        Check if the response has the required XML structure:
        - Has <start_reasoning>...</start_reasoning>
        - Has <start_answer>...</start_answer>
        - Has a \boxed{} answer inside <start_answer>
        """
        # Check for reasoning tags
        reasoning_pattern = r"<start_reasoning>(.*?)</start_reasoning>"
        reasoning_match = re.search(reasoning_pattern, sample_str, re.DOTALL)
        if not reasoning_match:
            return False

        # Check for answer tags
        answer_pattern = r"<start_answer>(.*?)</start_answer>"
        answer_match = re.search(answer_pattern, sample_str, re.DOTALL)
        if not answer_match:
            return False

        # Check if there's a \boxed{} in the answer section
        answer_content = answer_match.group(1)
        try:
            _ = extract_boxed(answer_content)
            return True
        except ValueError:
            return False

    def extract_answer_from_xml(self, sample_str: str) -> str:
        """Extract the answer from the <start_answer> section."""
        answer_pattern = r"<start_answer>(.*?)</start_answer>"
        answer_match = re.search(answer_pattern, sample_str, re.DOTALL)
        if not answer_match:
            raise ValueError("No <start_answer> section found")
        return answer_match.group(1)

    def check_answer(self, sample_str: str) -> bool:
        """Grade the answer extracted from the XML format."""
        try:
            # Extract from <start_answer> section
            answer_content = self.extract_answer_from_xml(sample_str)
            # Extract from \boxed{}
            extracted_answer = extract_boxed(answer_content)
        except ValueError:
            return False

        # Grade using the math grading function
        result = run_with_timeout_signal(
            grade_answer,
            args=(extracted_answer, self.answer),
            timeout_seconds=int(self.timeout)
        )
        if result is None:
            logger.warning(f"Timeout grading {extracted_answer} against {self.answer}")
            return False
        return result

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        """Provide a few-shot example showing the expected reasoning format."""
        return [
            {
                "role": "user",
                "content": (
                    "Sarah has 3 boxes of pencils. Each box contains 12 pencils. "
                    "She gives away 8 pencils. How many pencils does she have left?"
                    + ReasoningEnv.question_suffix()
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<start_reasoning>\n"
                    "Let me work through this step-by-step:\n"
                    "1. First, I need to find the total number of pencils Sarah starts with\n"
                    "2. She has 3 boxes, each with 12 pencils: 3 × 12 = 36 pencils\n"
                    "3. Then she gives away 8 pencils\n"
                    "4. So the remaining pencils are: 36 - 8 = 28 pencils\n"
                    "</start_reasoning>\n"
                    "<start_answer>\n"
                    "\\boxed{28}\n"
                    "</start_answer>"
                ),
            },
        ]


class ReasoningGsm8kDataset(RLDataset):
    """GSM8K dataset adapted for reasoning format training."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        import math
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        """Create an environment group builder for a single problem."""
        try:
            problem = x["question"]
            # Extract the final answer from GSM8K format (#### ANSWER)
            answer_text = x["answer"]
            lines = answer_text.splitlines()
            for line in reversed(lines):
                s = line.strip()
                if s.startswith("####"):
                    answer = s[4:].strip()
                    if answer.startswith(":"):
                        answer = answer[1:].strip()
                    answer = answer.replace(",", "").strip()
                    break
            else:
                logger.warning(f"No answer found for {answer_text}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None

        return ProblemGroupBuilder(
            env_thunk=partial(
                ReasoningEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="reasoning_gsm8k",
        )


@chz.chz
class ReasoningGsm8kDatasetBuilder(RLDatasetBuilder):
    """Builder for reasoning GSM8K dataset."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[ReasoningGsm8kDataset, ReasoningGsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = ReasoningEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        datasets = [
            ReasoningGsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])
