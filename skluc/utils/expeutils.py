import pathlib
import random
from collections import defaultdict
from typing import Iterable
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from skluc.utils import SingletonMeta
import matplotlib.pyplot as plt


def get_processed_result_filepath(curr_file, project_dir, ext="csv") -> pathlib.Path:
    """
    # todo make this function more versatile because it really relies on my organization right now

    Return output file path corresponding to current postprocessing script.

    Current file should have a path like:

    /.../.../yyyy/mm/script.py

    The outputfile will be like:

    `project_dir`/results/processed/yyyy/mm/script.`ext`


    Parameters
    ----------
    curr_file:
        The current file path with the
    project_dir:
        The root of the project.
    ext:
        The extension to append to the filename in the final path.

    Returns
    -------
        The path to the new result file.
    """
    # curr_file be like: "/home/toto/project/code/postprocessing/yyyy/mm/script.py"
    # from the example above, take only the string "yyyy/mm/script"
    curr_file_path = "/".join(str(curr_file).split("/")[-3:]).split(".")[0]
    csv_file_path = curr_file_path + f".{ext}"
    output_file = project_dir / "results/processed/" / csv_file_path
    return output_file

# todo make df utils submodule for scikit luc
def get_df_results(str_path_results, project_dir, dropna=True):
    """
    Return the df corresponding to csv file project_dir/str_path_results.

    Notes:
     - NA lines are removed.
     - str_path_results is added in column field "path_results_from_root"

    :param str_path_results:
    :param project_dir:
    :return:
    """
    abspath_results = project_dir / str_path_results
    df = pd.read_csv(abspath_results)
    if dropna:
        df.dropna(inplace=True)
    df["path_results_from_root"] = str_path_results
    return df


def build_df_from_dir(path_results_dir, col_to_delete=()):
    files = [x for x in path_results_dir.glob('**/*') if x.is_file()]
    lst_df = []
    lst_other_files = []
    real_count = 0
    for csv_filename in files:
        if csv_filename.suffix == ".csv":
            df = pd.read_csv(csv_filename, index_col=False)
            lst_df.append(df)
            real_count += 1
        else:
            lst_other_files.append(csv_filename)

    total_df = pd.concat(lst_df)

    for c in col_to_delete:
        total_df = total_df.drop([c], axis=1)

    return total_df


def get_line_of_interest(df: pd.DataFrame, keys_of_interest: Iterable, dct_values: dict,
                         dct_mapping_key_of_interest_to_dict: dict = None) -> pd.DataFrame:
    """
    Get the single line in `df` that perfectly matches the values in `dct_values` for the `keys_of_interest`.

    Parameters
    ----------
    df:
        The dataframe where to look for the perfect match.
    keys_of_interest:
        The columns of the dataframe that are interesting for the match.
    dct_values:
        The reference values of the columns to match.
    dct_mapping_key_of_interest_to_dict:
        Mapping between the keys of interest and the `dct_values`.
        It is useful if the column in the dataframe has a different name than the key in the `dct_values` attribute.

    Returns
    -------
        The dataframe containing one line of the input df matching the dct_values.
    """
    if dct_mapping_key_of_interest_to_dict is None:
        dct_mapping_key_of_interest_to_dict = dict()

    queries = []
    logger.debug(keys_of_interest)
    set_element_to_remove = set()
    # this loop buils a list of "queries"
    # (strings to evaluate in order to get a boolean index array. Example: 'df["column"] == value')
    for k in keys_of_interest:
        try:
            logger.debug("{}, {}, {}".format(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)],
                                             type(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)]), k))
            key_type = df.dtypes[k].name
            if key_type == "object" or dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)] is None or np.isnan(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)]):
                df[k] = df[k].astype(str)
                str_k = "'{}'".format(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)])
            else:
                str_k = "{}".format(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)])
        except KeyError:
            logger.warning("key {} not present in df or dict".format(k))
            set_element_to_remove.add(k)
            continue
        # if self[k] is None:
        #     str_k = "'None'"
        # elif k in ["--nb-units-dense-layer", "--param-reg-softmax-entropy", "--nb-factor", "--sparsity-factor"]:
        #     str_k = "'{}'".format(self[k])
        # else:
        #     str_k = self[k]

        query = "df_of_interest['{}']=={}".format(k, str_k)
        queries.append(query)

    # some keys of interest might be absent from df.
    keys_of_interest = list(set(keys_of_interest).difference(set_element_to_remove))

    # s_query = " & ".join(queries)
    df_of_interest = df
    for query in queries:
        s_eval = "df_of_interest[{}]".format(query)
        logger.debug(s_eval)
        df_of_interest_tmp = eval(s_eval)
        # try:
        assert not len(df_of_interest_tmp) < 1, "No corresponding line in df. Query {} discarded all".format(query)
        # except:
        #     pass
        df_of_interest = df_of_interest_tmp

    line_of_interest_no_duplicate = df_of_interest.drop_duplicates(keys_of_interest, inplace=False)
    if not df_of_interest.equals(line_of_interest_no_duplicate):
        columns_not_interesting_with_different_values = set()
        for column in set(df_of_interest.columns).difference(keys_of_interest):
            if not all(df_of_interest[column].fillna(0) == df_of_interest[column].fillna(0).iloc[0]):
                columns_not_interesting_with_different_values.add(column)
        logger.warning(f"There was some duplicated lines having same values for keys of interest. Columns with different values are: {columns_not_interesting_with_different_values}")

    line_of_interest = line_of_interest_no_duplicate

    logger.debug(line_of_interest)

    assert not len(line_of_interest) > 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory. There are multiple"
    assert not len(line_of_interest) < 1, "No corresponding pretrained model found in directory"

    return line_of_interest


class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(**dct_params, **kwargs)
        self.__init_identifier()
        self.init_output_file()

    def __init_identifier(self):
        if "identifier" not in self:
            # job_id = os.environ.get('OAR_JOB_ID')  # in case it is running with oarsub job scheduler
            # if job_id is None:
            job_id = str(int(time.time()))
            job_id = int(job_id) + random.randint(0, 10 ** len(job_id))
            # else:
            #     job_id = int(job_id)

            self["identifier"] = str(job_id)

    def init_output_file(self):
        self["output_file_resprinter"] = Path(self["identifier"] + "_results.csv")


class ResultPrinter(metaclass=SingletonMeta):
    """
    Class that handles 1-level dictionnaries and is able to print/write their values in a csv like format.
    """
    def __init__(self, parameters_dict=None, header=True, output_file=None, columns: list=None, separator=","):
        """
        :param args: the dictionnaries objects you want to print.
        :param header: tells if you want to print the header
        :param output_file: path to the outputfile. If None, no outputfile is written on ResultPrinter.print()
        :param columns: the list of columns (strings) that are allowed in the result directory.
        """
        self.__dict = dict()
        self.__header = header
        self.__output_file = output_file
        self.__columns = columns
        self.separator = separator

        if self.__columns is not None:
            self.__dict.update(dict((col, None) for col in self.__columns))

        if parameters_dict is not None:
            self.__dict.update(parameters_dict)
            self.__columns.extend(list(parameters_dict))

    def check_keys_in_columns(self, dct_key_values):
        if self.__columns is None:
            return
        else:
            set_columns = set(self.__columns)
            set_keys_dict = set(dct_key_values.keys())
            diff = set_keys_dict.difference(set_columns)
            if len(diff) == 0:
                return
            else:
                raise KeyError(f"Keys {diff} are not in the pre-specified column names.")

    def add(self, **kwargs):
        """
        Add dictionnary after initialisation.
        :param d: the dictionnary object you want to add.
        :return:
        """
        self.check_keys_in_columns(kwargs)
        self.__dict.update(kwargs)

    def _get_ordered_items(self):
        all_keys, all_values = zip(*self.__dict.items())
        arr_keys, arr_values = np.array(all_keys), np.array(all_values, dtype=object)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def print(self):
        """
        Call this function whener you want to print/write to file the content of the dictionnaires.
        :return:
        """
        headers, values = self._get_ordered_items()
        headers = [str(h) for h in headers]
        s_headers = self.separator.join(headers)
        values = [str(v) if not type(v) == str else "\""+v+"\"" for v in values]
        s_values = self.separator.join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")


class IntermediateResultStorage(metaclass=SingletonMeta):
    def __init__(self):
        self.dct_objective_values = defaultdict(list)

    def add(self, elm, list_name):
        self.dct_objective_values[list_name].append(elm)

    def clear(self):
        self.dct_objective_values = defaultdict(list)

    def __getitem__(self, item):
        return self.dct_objective_values[item]

    def get_all_names(self):
        return sorted(list(self.dct_objective_values.keys()))

    def store_all_items(self, path_output_file):
        np.savez(path_output_file, **self.dct_objective_values)

    def load_items(self, path_input_file):
        z_loaded = np.load(path_input_file)
        self.dct_objective_values.update(
            **dict(z_loaded)
        )


class ObjectiveValuesStorage(IntermediateResultStorage):
    def get_objective_values(self, list_name):
        return self[list_name]

    def get_all_curve_names(self):
        return self.get_all_names()

    def store_objective_values(self, path_output_file):
        self.store_all_items(path_output_file)

    def load_objective_values(self, path_input_file):
        self.load_items(path_input_file)

    def show(self):
        fig, tpl_axs = plt.subplots(nrows=1, ncols=len(self.dct_objective_values))

        for idx_ax, (name_trace, lst_obj_values) in enumerate(self.dct_objective_values.items()):
            iter_ids = np.arange(len(lst_obj_values))
            objective_values = np.array(lst_obj_values)
            try:
                tpl_axs[idx_ax].plot(iter_ids, objective_values)
                tpl_axs[idx_ax].set_title(name_trace)
            except TypeError:
                assert len(self.dct_objective_values) == 1
                tpl_axs.plot(iter_ids, objective_values)
                tpl_axs.set_title(name_trace)

        plt.show()


if __name__ == "__main__":
    assert ObjectiveValuesStorage() is ObjectiveValuesStorage()
    assert not (IntermediateResultStorage() is ObjectiveValuesStorage())
    assert IntermediateResultStorage() is IntermediateResultStorage()
