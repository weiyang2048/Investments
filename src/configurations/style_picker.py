from omegaconf import DictConfig, OmegaConf
from pyhere import here
import random

style_conf = OmegaConf.load(here("conf/style_conf/color_line_styles.yaml"))
# OmegaConf.resolve(conf)


def get_random_style(type: str = "color"):
    if type == "color":
        return random.choice(style_conf["colors"])
    elif type == "line_style":
        return random.choice(style_conf["line_styles"])


if __name__ == "__main__":
    pass
