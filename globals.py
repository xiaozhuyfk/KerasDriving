from ConfigParser import SafeConfigParser
from itertools import tee, izip
import logging

logger = logging.getLogger(__name__)

config = None

def read_configuration(configfile):
    """Read configuration and set variables.

    :return:
    """
    global config
    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser
    return parser
