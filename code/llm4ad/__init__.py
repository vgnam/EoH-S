import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The '(repr|frozen)' attribute .* has no effect .*",
    module=r"pydantic\._internal\._generate_schema",
)

from . import base
from . import method
from . import task
from .tools import profiler
from .tools import llm

__version__ = '1.0.0'
