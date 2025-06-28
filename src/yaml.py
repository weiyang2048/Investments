from omegaconf import OmegaConf


def union(args):
    unique_list = []
    for arg in args:
        for x in arg:
            if x not in unique_list:
                unique_list.append(x)
    return unique_list


def drop(args):
    return [x for x in args[0] if x not in args[1]]


def register_resolvers():
    if not OmegaConf.has_resolver("union"):
        OmegaConf.register_new_resolver("union", lambda *args: union(args))
    if not OmegaConf.has_resolver("drop"):
        OmegaConf.register_new_resolver("drop", lambda *args: drop(args))
