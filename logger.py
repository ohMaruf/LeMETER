import sys

from enum import Enum


class AnsiEscapeCodes(Enum):
    BOLD_BLUE = "\033[1;94m"
    BOLD_YELLOW = "\033[1;93m"
    BOLD_RED = "\033[1;91m"
    RESET = "\033[0m"

    def __str__(self):
        return self.value


def info(message: str) -> None:
    print(f"{AnsiEscapeCodes.BOLD_BLUE}info:{AnsiEscapeCodes.RESET} {message}", file=sys.stderr)  # noqa: E501


def warn(message: str) -> None:
    print(f"{AnsiEscapeCodes.BOLD_YELLOW}warn:{AnsiEscapeCodes.RESET} {message}", file=sys.stderr)  # noqa: E501


def log(message: str) -> None:
    print(f"{AnsiEscapeCodes.BOLD_RED}error:{AnsiEscapeCodes.RESET} {message}", file=sys.stderr)  # noqa: E501
