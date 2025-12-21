#!/usr/bin/env python3
"""
Entry point script to run the verification comparison module.

Usage:
    uv run main.py comparison
    uv run main.py train
    uv run main.py check
    uv run main.py help
"""

import sys

from src.rl.test_env import test_environment
from src.rl.train import train as rl_train
from src.verification.comparison import main as comparison_main


def print_help():
    help_message = """
Available commands:

  comp
    Run the comparison module to evaluate model performance and fairness.

  train
    Train the reinforcement learning model.

  check
    Test the reinforcement learning environment.

  help
    Display this help message.
"""
    print(help_message)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run main.py [comp|train|check|help]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "comp":
        comparison_main()
    elif command == "train":
        rl_train()
    elif command == "check":
        test_environment()
    elif command == "help":
        print_help()
    else:
        print(f"Unknown command: {command}")
        print("Use 'uv run main.py help' to look for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
