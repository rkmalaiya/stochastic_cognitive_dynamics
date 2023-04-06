import logging
import sys

def get_logger(lib_name = None):

    # can get details from https://docs.python.org/3/library/logging.html#logrecord-objects
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log = logging.getLogger(lib_name)
    hdlr = logging.StreamHandler(stream = sys.stdout)
    formatter = logging.Formatter(log_format)
    hdlr.setFormatter(formatter)
    #hdlr.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(hdlr)
    return log

if __name__ == "__main__":
    log = get_logger("testing")
    log.debug("Debug Working")
    log.info("Info Working")
    log.warning('Warning message')
    log.error("Error Working")
    log.critical('Critical message')