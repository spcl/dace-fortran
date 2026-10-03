from typing import Any, Callable

class ParserFactory:
    def create(self, std: str = ...) -> Callable[..., Any]: ...
