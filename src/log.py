import logging
import os
import logging.config

__module_path = os.path.dirname(os.path.realpath(__file__))
logging.config.fileConfig(__module_path + "/../log.ini")


def get_child_logger(name: str):
    """Returns a child logger from the root logger.

    The configuration of the logger is written to 'log.ini'.
    """
    return logging.getLogger(f"{name}")
