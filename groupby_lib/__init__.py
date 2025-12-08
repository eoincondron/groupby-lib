from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("groupby-lib")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from groupby_lib.emas import ema, ema_grouped
from groupby_lib.groupby import GroupBy, crosstab, install_groupby_fast
from groupby_lib.util import bools_to_categorical, nb_dot, pretty_cut
from groupby_lib.groupby.monkey_patch import install_groupby_fast
