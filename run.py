"""Render a scene defined in a Python file beside this script."""
from pathlib import Path

from agentbrush.cli import parser, render


if __name__ == '__main__':
    render(parser().parse_args(), Path(__file__).resolve().parent)
