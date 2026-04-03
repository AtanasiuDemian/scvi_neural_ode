"""scvi_neural_ode"""

import logging
from rich.console import Console
from rich.logging import RichHandler
from importlib.metadata import version

from ._constants import _CONSTANTS
from . import data, models

package_name = "scvi_neural_ode"
__version__ = version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("scvi_neural_ode: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["_CONSTANTS", "data", "models"]
