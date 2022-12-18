import logging
import sys

def get_logger(lib_name):
    #log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log = logging.getLogger(lib_name)
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    return log