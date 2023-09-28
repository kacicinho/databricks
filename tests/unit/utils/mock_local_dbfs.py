import os

from typing import NamedTuple
from unittest.mock import MagicMock

class MockFs:

    mount = MagicMock()
    unmount = MagicMock()
    mkdirs = MagicMock()
    mv = MagicMock()
    ls = MagicMock()


class MockDbutils(NamedTuple):

    fs: MockFs = MockFs()
