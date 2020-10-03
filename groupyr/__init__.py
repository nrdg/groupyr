from . import datasets  # noqa
from . import utils  # noqa
from .sgl import *  # noqa
from .logistic import *  # noqa
from ._version import version as __version__  # noqa

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
