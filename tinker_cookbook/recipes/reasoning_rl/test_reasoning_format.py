"""
Smoke test for reasoning format validation.

Tests the XML format checking without requiring Tinker API access.
"""
import re


def extract_boxed(text: str) -> str:
    """Simple version of boxed extraction for testing."""
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("No \\boxed{} found")
    return match.group(1)


def check_format(sample_str: str) -> bool:
    """Test the format checking logic."""
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


def extract_answer_from_xml(sample_str: str) -> str:
    """Extract the answer from the <start_answer> section."""
    answer_pattern = r"<start_answer>(.*?)</start_answer>"
    answer_match = re.search(answer_pattern, sample_str, re.DOTALL)
    if not answer_match:
        raise ValueError("No <start_answer> section found")
    return extract_boxed(answer_match.group(1))


def test_valid_format():
    """Test a correctly formatted response."""
    response = """<start_reasoning>
Let me work through this step-by-step:
1. First, I need to find the total number of pencils
2. Sarah has 3 boxes with 12 pencils each: 3 × 12 = 36
3. She gives away 8 pencils
4. Remaining: 36 - 8 = 28
</start_reasoning>
<start_answer>
\\boxed{28}
</start_answer>"""

    assert check_format(response), "Valid format should pass"
    answer = extract_answer_from_xml(response)
    assert answer == "28", f"Expected '28', got '{answer}'"
    print("✓ Valid format test passed")


def test_missing_reasoning():
    """Test response missing reasoning section."""
    response = """<start_answer>
\\boxed{42}
</start_answer>"""

    assert not check_format(response), "Should fail without reasoning section"
    print("✓ Missing reasoning test passed")


def test_missing_answer():
    """Test response missing answer section."""
    response = """<start_reasoning>
Some reasoning here
</start_reasoning>"""

    assert not check_format(response), "Should fail without answer section"
    print("✓ Missing answer test passed")


def test_missing_boxed():
    """Test response with answer section but no \\boxed{}."""
    response = """<start_reasoning>
Some reasoning
</start_reasoning>
<start_answer>
Just plain text answer
</start_answer>"""

    assert not check_format(response), "Should fail without \\boxed{}"
    print("✓ Missing boxed test passed")


def test_complex_answer():
    """Test with complex mathematical answer."""
    response = """<start_reasoning>
Let's solve this equation:
1. Start with 2x + 3 = 7
2. Subtract 3: 2x = 4
3. Divide by 2: x = 2
</start_reasoning>
<start_answer>
The solution is \\boxed{2}
</start_answer>"""

    assert check_format(response), "Complex format should pass"
    answer = extract_answer_from_xml(response)
    assert answer == "2", f"Expected '2', got '{answer}'"
    print("✓ Complex answer test passed")


def test_multiline_reasoning():
    """Test with multi-line reasoning."""
    response = """<start_reasoning>
This problem requires several steps:

Step 1: Calculate the initial amount
- John has $50
- He spends $15 on food
- Remaining: $50 - $15 = $35

Step 2: Calculate the final amount
- He earns $20
- Total: $35 + $20 = $55
</start_reasoning>
<start_answer>
\\boxed{55}
</start_answer>"""

    assert check_format(response), "Multiline reasoning should pass"
    answer = extract_answer_from_xml(response)
    assert answer == "55", f"Expected '55', got '{answer}'"
    print("✓ Multiline reasoning test passed")


def run_all_tests():
    """Run all format validation tests."""
    print("Running reasoning format smoke tests...\n")

    test_valid_format()
    test_missing_reasoning()
    test_missing_answer()
    test_missing_boxed()
    test_complex_answer()
    test_multiline_reasoning()

    print("\n✓ All tests passed!")
    print("\nThe format validation logic is working correctly.")
    print("You can now run the full training pipeline with:")
    print("  python tinker_cookbook/recipes/reasoning_rl/train_reasoning.py")


if __name__ == "__main__":
    run_all_tests()
