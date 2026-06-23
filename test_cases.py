from predict import predict_spam

TEST_CASES = [
    ("Congratulations! You won a FREE iPhone. Click now to claim your prize!", "SPAM"),
    ("Hey, are we still meeting tomorrow at 5 PM? Let me know.", "NOT SPAM"),
    ("URGENT: Your bank account has been suspended. Verify your details immediately!", "SPAM"),
    ("Can you please send me the lecture notes from today's class?", "NOT SPAM"),
    ("Win cash prizes worth $1000! Limited time offer.", "SPAM"),
    ("Mom said dinner is ready. Come home when you can.", "NOT SPAM"),
    ("FREE entry! You have been selected for a $500 gift card.", "SPAM"),
    ("The assignment submission deadline is next Monday.", "NOT SPAM"),
    ("Your account password expires today. Click the link below.", "SPAM"),
    ("Are you coming to the birthday dinner tonight?", "NOT SPAM"),
]


def run_tests():
    passed = 0
    for i, (email, expected) in enumerate(TEST_CASES, 1):
        result, confidence = predict_spam(email)
        status = "PASS" if result == expected else "FAIL"
        print(f"Test {i:02d} [{status}] → Expected: {expected} | Got: {result} ({confidence}%)")
        if result == expected:
            passed += 1

    print(f"\n{passed}/{len(TEST_CASES)} tests passed.")


if __name__ == "__main__":
    run_tests()