"""
General utility functions that doesn't fit in other categories. Usually programming tools.
"""
import logging
import daiquiri
import warnings
from weakref import WeakValueDictionary
import psutil
import time as t
import os

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger()


def deprecated(msg=""):
    """
    Decorator which can be used to mark functions as deprecated and write a message.

    :param msg: The message you need to write.
    :return: The inner decorator function
    """
    def inner(func):
        """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.
        """
        def new_func(*args, **kwargs):
            s = "Call to deprecated function {}".format(func.__name__)
            if str(msg).strip != "":
                s += ": {}.".format(msg)
            else:
                s += "."
            warnings.warn(s)
            return func(*args, **kwargs)

        new_func.__name__ = func.__name__
        new_func.__doc__ = func.__doc__
        new_func.__dict__.update(func.__dict__)

        return new_func
    return inner

class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # This variable declaration is required to force a
            # strong reference on the instance.
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

def singleton(cls):
    """
    Simple singleton implementation.

    Usage:

    @singleton
    class A:
      pass

    a = A()
    b = A()
    # a == b
    """
    instance = None

    def class_instanciation_or_not(*args, **kwargs):
        nonlocal instance
        if not instance:
            instance = cls(*args, **kwargs)
        return instance
    return class_instanciation_or_not

def memory_usage():
    """
    Copy pasted from https://airbrake.io/blog/python-exception-handling/memoryerror

    Prints current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    PROCESS = psutil.Process(os.getpid())
    GIGA = 10 ** 9
    # MEGA_STR = ' ' * MEGA

    mem = psutil.virtual_memory()
    total, available, used, free = mem.total / GIGA, mem.available / GIGA, mem.used / GIGA, mem.free / GIGA
    proc = PROCESS.memory_info()[1] / GIGA
    return 'process = %s total = %s available = %s used = %s free = %s' \
          % (proc, total, available, used, free)

def time_fct(fct, *args, n_iter=100, **kwargs):
    """
    Return the average time spent by the function.

    :param fct: the actual function to time
    :param args: the positional arguments of the function
    :param n_iter: number of runs of the function
    :param kwargs: the keyword arguments of the function
    :return: the average time of execution of the function
    """
    time_sum = 0
    for _ in range(n_iter):
        start = t.time()
        fct(*args, **kwargs)
        stop = t.time()
        time_sum += stop - start
    return time_sum / n_iter

def log_memory_usage(context=None):
    """Logs current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    if context is not None:
        str_memory_usage = context + ":\t"
    else:
        str_memory_usage = ""

    mem = memory_usage()
    str_memory_usage += mem
    logger.debug(str_memory_usage)
