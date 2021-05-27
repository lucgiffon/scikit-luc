import numpy as np
import os

from loguru import logger


def save_memmap_data(project_dir, output_rel_dirpath, dataname, data_size, nb_features, Xy_gen):
    output_path_obs = project_dir / output_rel_dirpath / (dataname + ".dat")
    output_path_labels = project_dir / output_rel_dirpath / (dataname + ".lab")
    fp_obs = np.memmap(output_path_obs, dtype='float32', mode='w+', shape=(data_size, nb_features))
    fp_labels = np.memmap(output_path_labels, mode='w+', shape=(data_size,))

    logger.info("{} Data will be created in file: {}; labels stored in file: {}".format(dataname, output_path_obs,
                                                                                        output_path_labels))
    logger.info("About to create {}: Total {} examples.".format(dataname, data_size))

    curr_idx = 0
    for i, (batch_X, batch_y) in enumerate(Xy_gen):
        curr_batch_size = batch_X.shape[0]
        fp_obs[curr_idx:curr_idx + curr_batch_size] = batch_X
        if batch_y is not None:
            fp_labels[curr_idx:curr_idx + curr_batch_size] = batch_y
        curr_idx += curr_batch_size

    if batch_y is None:
        os.remove(str(output_path_labels))


def generator_data(data_load_func, size_batch=10000):
    X, y = data_load_func()
    data_size = X.shape[0]
    total_nb_chunks = int(data_size // size_batch)
    remaining = int(data_size % size_batch)
    for i in range(total_nb_chunks):
        logger.info("Chunk {}/{}".format(i + 1, total_nb_chunks))
        if y is None:
            yield X[i*size_batch: (i+1)*size_batch], None
        else:
            yield X[i * size_batch: (i + 1) * size_batch], y[i * size_batch: (i + 1) * size_batch]
    if remaining > 0:
        if y is None:
            yield X[(i+1)*size_batch: ], None
        else:
            yield X[(i + 1) * size_batch:], y[(i + 1) * size_batch:]
