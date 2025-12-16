# NOTE: we use an absolute import here to get the type checking to work properly.
# Additionally we re-export the imported symbols to make ruff happy.
from nanobind_example.nanobind_example_ext import (
    add as add,
    __doc__ as __doc__,
)
