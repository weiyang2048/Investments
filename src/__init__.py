# You can leave this empty or define package-level imports
import sys
from colorama import Fore, Style
import pandas as pd
import numpy as np
from loguru import logger


logger.remove()

logger.level("todo", no=15, color="<bg white><black>", icon="🚨")
logger.add(
    sys.stderr,
    format=(
        "<black>{time:MM-DD HH:mm}</black> | "
        + "<black>{name}:{function} [{line}]</black> | "
        + "<lvl>{level}</lvl> {level.icon} | "
        + "<lvl>{message}</lvl>"
    ),
    colorize=True,
    filter=lambda record: record["level"].name == "todo",
)

# override default logger info level
# forestgreen
logger.level("data", no=10, color="<black>", icon="🔍")
logger.add(
    sys.stderr,
    format=("{level.icon} | " + "{message}"),
    colorize=True,
    filter=lambda record: record["level"].name == "data",
)
