import codecs
import json
import os
import yaml


def parse_file(path):
    """
    Parse configuration values form the given file.

    Args:
        path (str): Path of the file.  It should be a JSON file or
            a YAML file, with corresponding file extension.
    """
    _, ext = os.path.splitext(path)
    config_dict = None
    if ext == '.json':
        with codecs.open(path, 'rb', 'utf-8') as f:
            config_dict = dict(json.load(f))
    elif ext in ('.yml', '.yaml'):
        with codecs.open(path, 'rb', 'utf-8') as f:
            config_dict = dict(yaml.load(f))
    else:
        raise ValueError('Config file of this type is not supported: {}'.
                         format(path))
    return config_dict


class Singleton(object):
    """
    Base class for singleton classes.

    >>> class Parent(Singleton):
    ...     pass

    >>> class Child(Parent):
    ...     pass

    >>> Parent() is Parent()
    True
    >>> Child() is Child()
    True
    >>> Parent() is not Child()
    True
    """

    __instances_dict = {}

    def __new__(cls, *args, **kwargs):
        if cls not in Singleton.__instances_dict:
            Singleton.__instances_dict[cls] = \
                object.__new__(cls, *args, **kwargs)
        return Singleton.__instances_dict[cls]

