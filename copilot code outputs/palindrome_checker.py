def is_palindrome(s):
    """
    Check if a string is a palindrome.
    Ignores case and non-alphanumeric characters.
    
    Args:
        s (str): The input string to check
        
    Returns:
        bool: True if palindrome, False otherwise
    """
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    
    # Check if it reads the same forwards and backwards
    return cleaned == cleaned[::-1]


# Example test cases
if __name__ == "__main__":
    test_cases = [
        "A man, a plan, a canal: Panama",  # True
        "race a car",                       # False
        "Was it a car or a cat I saw?",    # True
        "Madam, I'm Adam",                  # True
        "hello",                            # False
        "12321",                            # True
        "A1b2B1a",                          # True
        "python",                           # False
        " ",                                # True (empty after cleaning)
        "0P",                               # False
    ]
    
    print("Palindrome Checker - Test Results:\n")
    for test in test_cases:
        result = is_palindrome(test)
        print(f"'{test}' -> {result}")
