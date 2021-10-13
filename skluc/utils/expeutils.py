import random

import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from skluc.utils import SingletonMeta


def get_ext_output_file(curr_file, project_dir, ext="csv"):
    """
    Return output file name corresponding to current postprocessing script.

    :param ext:
    :return:
    """
    curr_file_path = "/".join(str(curr_file).split("/")[-3:]).split(".")[0]
    csv_file_path = curr_file_path + f".{ext}"
    output_file = project_dir / "results/processed/" / csv_file_path
    return output_file


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
            df = pd.read_csv(csv_filename)
            lst_df.append(df)
            real_count += 1
        else:
            lst_other_files.append(csv_filename)

    total_df = pd.concat(lst_df)

    for c in col_to_delete:
        total_df = total_df.drop([c], axis=1)

    return total_df


def get_line_of_interest(df, keys_of_interest, dct_values, dct_mapping_key_of_interest_to_dict=None):
    """
    Get the single line in `df` that perfectly matches the values in `dct_values` for the `keys_of_interest`.

    :param df: The dataframe where to look for the perfect match.
    :param keys_of_interest: The columns of the dataframe that are interesting for the match.
    :param dct_values: The reference values of the columns to match.
    :param dct_mapping_key_of_interest_to_dict: Mapping between the keys of interest and the dict values.
    :return:
    """
    if dct_mapping_key_of_interest_to_dict is None:
        dct_mapping_key_of_interest_to_dict = dict()

    queries = []
    logger.debug(keys_of_interest)
    set_element_to_remove = set()
    for k in keys_of_interest:
        logger.debug("{}, {}, {}".format(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)], type(dct_values[dct_mapping_key_of_interest_to_dict.get(k, k)]), k))
        try:
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

    line_of_interest = df_of_interest
    line_of_interest.drop_duplicates(keys_of_interest, inplace=True)
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
    def __init__(self, *args, header=True, output_file=None, columns=None):
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

        if self.__columns is not None:
            self.__dict.update(dict((col, None) for col in self.__columns))

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
        s_headers = ",".join(headers)
        values = [str(v) for v in values]
        s_values = ",".join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")

