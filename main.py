#!/usr/bin/env python3
"""
Entry point script to run the verification comparison module.

Usage:
    uv run main.py comparison
    uv run main.py help
"""

import sys

from src.verification.comparison import main as comparison_main


def print_help():
    help_message = """
Available commands:

  comp
    Run the comparison module to evaluate model performance and fairness.

  help
    Display this help message.
"""
    print(help_message)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run main.py [comp|help]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "comp":
        comparison_main()
    elif command == "help":
        print_help()
    else:
        print(f"Unknown command: {command}")
        print("Use 'uv run main.py help' to look for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
