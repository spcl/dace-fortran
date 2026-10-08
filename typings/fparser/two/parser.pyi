from collections.abc import Callable
from typing import Any

class ParserFactory:
    def create(self, std: str = ...) -> Callable[..., Any]: ...
