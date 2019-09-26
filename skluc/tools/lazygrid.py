#!/usr/bin/env python3

"""
The lazygrid program allows to build command lines array from a configuration file.

Command line arrays are text files containing one set of command line arguments at each line.

This is usefull for grid search: you can launch the same script many times with different arguments stored in the command line array.


How does this work?
-------------------

You can find an example of configuration file just beside this script: `lazyfile_example.yml`. There is also an example output of such configuration file in `arrayparam_example.txt`.
Just like Makefiles, the lazyfile is structured by rules that rely upon each other.

The rule `all` is the master rule of the file, this is the one that will be executed and its absence will result in an error. The rule `all` calls
other rules. In contrary to other rules, the rule `all` will not concatenate the output of its subrule but simply launch them, one after the other.

A rule can have keyword parameters or simple parameters:

{"--this-is-a-kw-parameter": ["kw parameter can take one value", "or one other value"]}

["--this-is-a-simple-boolean-parameter", "--this-is-an-other-simple-boolean-parameter"]


Usage:
    lazygrid -l lazyfile

Options:
    -l --lazyfile lazyfile          The input configuration yml file.
"""

import yaml
from collections import OrderedDict
from pprint import pprint
import copy
import numpy as np
import math
import os
import time
import random

# todo specify external modules (ex numpy/maths) in the yaml file
from docopt import docopt

def build_cmd(dict_arg):
    cmd_lines = []
    try:
        todo_cmd_lines = dict_arg["all"].keys()
    except KeyError:
        raise KeyError("There should be a section 'all'")

    if list(dict_arg.keys())[0] != "all":
        raise ValueError("The first section of the configuration file should be 'all'")
    todo_cmd_lines_cases = list(dict_arg.keys())[1:]
    cmd_line_cases = {}
    for case in todo_cmd_lines_cases:
        try:
            case_section = dict_arg[case]
        except KeyError:
            raise KeyError("Section {} referenced in all but does not exist".format(case))
        cmd_line_case = [""]
        for key, value in case_section.items():
            value = eval(str(value))
            tmp_cmd_line_case = []
            if type(value) == list:
                for cmd in cmd_line_case:
                    for elm in value:
                        tmp_cmd_line_case.append(" ".join([cmd, elm]).strip())
            elif type(value) == OrderedDict:
                for cmd in cmd_line_case:
                    for key_arg, value_arg in value.items():
                        lst_value_arg = eval(str(value_arg))
                        for value_arg in lst_value_arg:
                            tmp_cmd_line_case.append(" ".join([cmd, str(key_arg) + " " + str(value_arg)]).strip())
            elif value is None:
                try:
                    to_add_cmd_line = cmd_line_cases[key]
                except KeyError:
                    raise KeyError("{} is referenced in {} but doesnt exist. Make sure it is defined BEFORE the section {}".format(key, case, case))

                for cmd in cmd_line_case:
                    for cmd_line_to_add in to_add_cmd_line:
                        tmp_cmd_line_case.append(" ".join([cmd, cmd_line_to_add]))

            else:
                raise Exception
            cmd_line_case = copy.deepcopy(tmp_cmd_line_case)
        cmd_line_cases[case] = cmd_line_case
    for todo_cmd_line in todo_cmd_lines:
        cmd_lines.extend(cmd_line_cases[todo_cmd_line])
    return cmd_lines


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def main():
    arguments = docopt(__doc__)
    with open(os.path.abspath(arguments["--lazyfile"])) as f:
        dataMap = ordered_load(f)
    final_cmd_lines = build_cmd(dataMap)
    for line in final_cmd_lines:
        print(line)

if __name__ == "__main__":
    main()





