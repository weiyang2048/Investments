from omegaconf import OmegaConf
import yaml
from pyhere import here


def union(args):
    unique_list = []
    for arg in args:
        for x in arg:
            if x not in unique_list:
                unique_list.append(x)
    return unique_list


def drop(args):
    return [x for x in args[0] if x not in args[1]]


def include(args):

    results, collection_type = None, None
    for file in args:
        filename = here("conf/", file.replace(".", "/") + ".yaml")
        with open(filename, "r") as f:
            temp = yaml.load(f, yaml.FullLoader)
            if collection_type is None:
                collection_type = type(temp)
                results = collection_type()
            if collection_type == dict:
                results.update(temp)
            else:
                results.extend(temp)

    return results


def register_resolvers():
    if not OmegaConf.has_resolver("union"):
        OmegaConf.register_new_resolver("union", lambda *args: union(args))
    if not OmegaConf.has_resolver("drop"):
        OmegaConf.register_new_resolver("drop", lambda *args: drop(args))
    if not OmegaConf.has_resolver("include"):
        OmegaConf.register_new_resolver("include", lambda *args: include(args))
