# src/template_extractor.py
import re

def normalize_msg(message: str) -> str:
    """
    Normalizes a log message by removing variable parts like numbers, IPs, etc.
    This helps the model focus on the log's template or pattern.

    Args:
        message (str): The raw log message.

    Returns:
        str: The normalized log message.
    """
    if not isinstance(message, str):
        return ""

    # 1. Replace IP addresses with a generic placeholder
    message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', message)

    # 2. Replace numbers (integers, floats, hex) with a placeholder
    message = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', message) # Hex codes
    message = re.sub(r'\b\d+\b', '<NUM>', message)       # Integers

    # 3. Replace file paths (both Unix and Windows style)
    message = re.sub(r'(?:\/|\\)[^\s]+', '<PATH>', message)

    # 4. Replace other common symbols and clean up spacing
    message = re.sub(r'[^\w\s]', ' ', message) # Remove punctuation
    message = re.sub(r'\s+', ' ', message).strip() # Collapse multiple spaces

    return message.lower() # Return in lowercase for consistency

# This block allows you to test the function directly
if __name__ == '__main__':
    test_messages = [
        "User 'alice' logged in from 192.168.1.10",
        "Transaction failed with error code 50021",
        "File not found at /var/log/system.log",
        "Memory usage exceeded threshold: 8192 MB",
        "CRITICAL,user-database,Exception: Unable to connect to the database master node."
    ]

    print("--- Testing Log Normalization ---")
    for msg in test_messages:
        normalized = normalize_msg(msg)
        print(f"Original:  {msg}")
        print(f"Normalized: {normalized}\n")