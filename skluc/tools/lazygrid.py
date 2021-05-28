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

"""
import click_pathlib
import yaml
import click
from collections import OrderedDict, defaultdict

# these are not used in the code but are necessary for evaluation of parameters
import numpy as np
import math
import random
from pathlib import Path
# todo specify external modules (ex numpy/maths) in the yaml file


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


class LazygridParser:
    def __init__(self, path):
        self.input_path = path
        self.lazyfilename = "/".join(str(path).split("/")[-3:]).split(".")[0]
        with open(path) as f:
            self.dataMap = ordered_load(f)
        self.final_cmd_lines = []
        self.dct_argument_combinations_by_rule = defaultdict(lambda: [""])

        self.build_cmd()

    @staticmethod
    def add_previous_rule_to_current_cmd(current_cmd, previous_rule_cmd_lines):
        tmp_cmd_line_case = []
        for cmd_line_to_add in previous_rule_cmd_lines:
            tmp_cmd_line_case.append(" ".join([current_cmd, cmd_line_to_add]))
        return tmp_cmd_line_case

    def _parse_value_type_list(self, value, cmd):
        tmp_cmd_line_case = []
        for elm in value:
            stripped_elm = elm.strip(":")
            if self.dct_argument_combinations_by_rule[stripped_elm] != [""]: # this is equivalent to say 'if the rule exists' but handle defaultdict
                # if this is a reference to a previous rule, then substitute here the content of the rule
                previous_rule_name = stripped_elm
                to_add_cmd_lines = self.dct_argument_combinations_by_rule[previous_rule_name]
                tmp_cmd_line_case.extend(self.add_previous_rule_to_current_cmd(cmd, to_add_cmd_lines))
            else:
                tmp_cmd_line_case.append(" ".join([cmd, elm]).strip())
        return tmp_cmd_line_case

    def _parse_value_type_ordereddict(self, value, cmd):
        tmp_cmd_line_case = []
        idx_value_arg = 0
        len_value_args = -1
        # each item value in the dict must be an iterable with constant length
        # create command lines that takes pairs of items but not the combination of them
        while True:
            cmd_line = ""

            for key_arg, raw_value_arg in value.items():
                # there is a lot of computation being repeated between iteration here but it shouldn't
                # cost so much
                formated_value_arg = (f"" + str(raw_value_arg)).format(LAZYFILE=self.lazyfilename)
                iterable_value_arg = eval(str(formated_value_arg))

                # check if the number of coefficients is consistent
                if len_value_args == -1:
                    len_value_args = len(iterable_value_arg)
                else:
                    try:
                        assert len_value_args == len(iterable_value_arg)
                    except AssertionError:
                        raise ValueError(
                            f"In dict with multiple entries, all entries must have the same number of elements."
                            f"len({iterable_value_arg}) == {len(iterable_value_arg)} != {len_value_args}")

                curr_value_iter = iterable_value_arg[idx_value_arg]
                cmd_line += str(key_arg) + " " + str(curr_value_iter) + " "

            tmp_cmd_line_case.append(" ".join([cmd, cmd_line]).strip())
            idx_value_arg += 1

            if idx_value_arg >= len_value_args:
                break

        return tmp_cmd_line_case

    def build_arguments_combinations_of_rule(self, rulename, rule_content):
        # the initial argument combination is just the empty argument combination
        # this will contain all the "sub command lines" of the rule
        for key, value in rule_content.items():
            # each new item in the rule will be appended to all
            # the previously constructed argument combinations of the rule

            # value = eval(str(value))
            tmp_cmd_line_case = []
            for cmd in self.dct_argument_combinations_by_rule[rulename]:
                # positional arguments are stored in a list, or it is a list of previous rules
                if type(value) == list:
                    tmp_cmd_line_case.extend(self._parse_value_type_list(value, cmd))
                # keyword arguments are stored in a dict, ordered because yaml keep the ordering
                elif type(value) == OrderedDict:
                    tmp_cmd_line_case.extend(self._parse_value_type_ordereddict(value, cmd))

                # in case there is only
                elif type(value) == str:
                    raise NotImplementedError("Sould make evaluation here")

                # in case it is a reference to an other (previous) rule
                elif value is None:
                    try:
                        to_add_cmd_lines = self.dct_argument_combinations_by_rule[key]
                    except KeyError:
                        raise KeyError(
                            "{} is referenced in {} but doesnt exist. Make sure it is defined BEFORE the section {}".format(key,
                                                                                                                            rulename,
                                                                                                                            rulename))

                    tmp_cmd_line_case.extend(self.add_previous_rule_to_current_cmd(cmd, to_add_cmd_lines))

                else:
                    raise Exception

            # new argument combinations have been created from the previous ones, and now they replace them
            self.dct_argument_combinations_by_rule[rulename] = tmp_cmd_line_case

    def build_cmd(self):
        try:
            todo_cmd_lines = self.dataMap["all"].keys()
        except KeyError:
            raise KeyError("There should be a rule 'all'")

        assert list(self.dataMap.keys())[0] == "all"

        all_rule_names = list(self.dataMap.keys())[1:]
        for rulename in all_rule_names:
            # for each rule, build the list of argument combinations which it describes
            rule_content = self.dataMap[rulename]
            self.build_arguments_combinations_of_rule(rulename, rule_content)

        for todo_cmd_line in todo_cmd_lines:
            self.final_cmd_lines.extend(self.dct_argument_combinations_by_rule[todo_cmd_line])

    def print(self):
        for line in self.final_cmd_lines:
            print(line)

    def write(self):
        with open(self.input_path.parent / (self.input_path.stem + ".grid"), 'w') as of:
            for line in self.final_cmd_lines:
                of.write(line + "\n")

    def count(self):
        print(len(self.final_cmd_lines))


@click.command()
@click.argument("lazyfile", type=click_pathlib.Path(exists=True, dir_okay=True, resolve_path=True))
@click.option("--count", "-c", is_flag=True, help="Count the number of parameter lines.")
@click.option("--write", "-w", is_flag=True, help="Write output file next to input lazyfile with .grid extension.")
def main(lazyfile, count, write):
    abspath_lazyfile = lazyfile.absolute()
    lazyfile_parser = LazygridParser(abspath_lazyfile)
    if not write:
        lazyfile_parser.print()
    else:
        lazyfile_parser.write()
    if count:
        lazyfile_parser.count()


if __name__ == "__main__":
    main()





