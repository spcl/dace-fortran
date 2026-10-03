"""Typing stub: fparser generates the Fortran2003 production classes dynamically, so every name is a ``Base`` subclass."""

from fparser.two.utils import Base

def __getattr__(name: str) -> type[Base]: ...
