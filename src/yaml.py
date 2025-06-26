from omegaconf import OmegaConf


def register_resolvers():
    if not OmegaConf.has_resolver("union"):
        OmegaConf.register_new_resolver("union", lambda *args: sum(args, []))
