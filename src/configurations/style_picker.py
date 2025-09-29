from omegaconf import DictConfig, OmegaConf
from pyhere import here
import random
from src.configurations import register_resolvers

style_conf = OmegaConf.load(here("conf/style_conf/color_line_styles.yaml"))
# OmegaConf.resolve(conf)

main_conf = OmegaConf.load(here("conf/main.yaml"))
register_resolvers()
# OmegaConf.resolve(main_conf)
print(main_conf)


def get_random_style(type: str = "color"):
    if type == "color":
        return random.choice(style_conf["colors"])
    elif type == "line_style":
        return random.choice(style_conf["line_styles"])


if __name__ == "__main__":
    pass
