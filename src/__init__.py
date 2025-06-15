# You can leave this empty or define package-level imports
import sys
from colorama import Fore, Style
import pandas as pd
import numpy as np
from loguru import logger


try:
    # Remove default handler
    logger.remove()

    # Add custom handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add todo level if it doesn't exist
    if "todo" not in logger._core.levels:
        logger.level("todo", no=15, color="<bg white><black>", icon="🚨")

    # override default logger info level
    # forestgreen
    logger.level("data", no=10, color="<black>", icon="🔍")
    logger.add(
        sys.stderr,
        format=("{level.icon} | " + "{message}"),
        colorize=True,
        filter=lambda record: record["level"].name == "data",
    )
except Exception as e:
    pass
