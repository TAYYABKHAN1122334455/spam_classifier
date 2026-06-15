# =============================================================================
# test_cases.py
# Author  : Jamal Raja (Roll No. 032)
# Purpose : Validate the trained Naive Bayes spam classifier against a
#           curated set of 10 labelled test emails covering both spam and
#           legitimate (ham) scenarios. Prints a structured report with
#           pass/fail status, predicted label, and confidence per test case,
#           followed by a summary table.
# Dependency: predict.py and trained model files must exist before running.
# =============================================================================

from predict import predict_spam

# ---------------------------------------------------------------------------
# Test Suite
# Each entry is a tuple of (email_text, expected_label).
# expected_label must be either "SPAM" or "NOT SPAM".
# ---------------------------------------------------------------------------
TEST_CASES: list[tuple[str, str]] = [
    (
        "Congratulations! You won a FREE iPhone. Click now to claim your prize!",
        "SPAM",
    ),
    (
        "Hey, are we still meeting tomorrow at 5 PM? Let me know.",
        "NOT SPAM",
    ),
    (
        "URGENT: Your bank account has been suspended. Verify your details immediately!",
        "SPAM",
    ),
    (
        "Can you please send me the lecture notes from today's class?",
        "NOT SPAM",
    ),
    (
        "Win cash prizes worth $1000! Limited time offer. Call now and claim your reward!",
        "SPAM",
    ),
    (
        "Mom said dinner is ready. Come home when you can.",
        "NOT SPAM",
    ),
    (
        "FREE entry! You have been selected for a $500 gift card. Respond now!",
        "SPAM",
    ),
    (
        "The assignment submission deadline is next Monday. Please do not forget.",
        "NOT SPAM",
    ),
    (
        "Your account password expires today. Click the link below to update it immediately.",
        "SPAM",
    ),
    (
        "Are you coming to the birthday dinner tonight? We are meeting at 8.",
        "NOT SPAM",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_tests(test_cases: list[tuple[str, str]]) -> None:
    """
    Execute all test cases against predict_spam() and print a formatted report.

    For each test case the function prints:
        - Pass or Fail status
        - Truncated email preview (first 60 characters)
        - Expected vs predicted label
        - Confidence percentage of the prediction

    A summary table is printed after all individual results.

    Parameters
    ----------
    test_cases : list[tuple[str, str]]
        List of (email_text, expected_label) tuples to evaluate.
    """
    total  = len(test_cases)
    passed = 0
    failed = 0

    separator = "=" * 70

    print(separator)
    print("         EMAIL SPAM CLASSIFIER - TEST REPORT")
    print(separator)

    for index, (email, expected) in enumerate(test_cases, start=1):
        result, confidence = predict_spam(email)

        # Determine pass / fail
        is_correct  = result == expected
        status_icon = "PASS" if is_correct else "FAIL"

        if is_correct:
            passed += 1
        else:
            failed += 1

        # Truncate email preview to keep output compact
        preview = email[:60] + ("..." if len(email) > 60 else "")

        print(f"\nTest {index:02d}  [{status_icon}]")
        print(f"  Email      : {preview}")
        print(f"  Expected   : {expected}")
        print(f"  Predicted  : {result}  (Confidence: {confidence}%)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print(separator)
    print(f"  Total Tests  : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print(f"  Test Score   : {(passed / total) * 100:.0f}%")
    print(separator)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_tests(TEST_CASES)
