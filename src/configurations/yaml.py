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


def get_keys(args):
    for file in args:
        filename = here("conf/", file.replace(".", "/") + ".yaml")
        with open(filename, "r") as f:
            temp = yaml.load(f, yaml.FullLoader)
            return list(temp.keys())


def register_resolvers():
    if not OmegaConf.has_resolver("union"):
        print("registering union resolver")
        OmegaConf.register_new_resolver("union", lambda *args: union(args))
    else:
        print("union resolver already registered")
    if not OmegaConf.has_resolver("include"):
        OmegaConf.register_new_resolver("include", lambda *args: include(args))
    else:
        print("include resolver already registered")
    if not OmegaConf.has_resolver("get_keys"):
        OmegaConf.register_new_resolver("get_keys", lambda *args: get_keys(args))
    else:
        print("get_keys resolver already registered")
