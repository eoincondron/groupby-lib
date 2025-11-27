from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("groupby-lib")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from groupby_lib.groupby import GroupBy, crosstab, install_groupby_fast
from groupby_lib.util import pretty_cut, bools_to_categorical, nb_dot
