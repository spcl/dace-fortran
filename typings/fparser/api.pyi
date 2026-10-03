from fparser.common.readfortran import FortranStringReader

def get_reader(source: str, isfree: bool | None = ..., isstrict: bool | None = ...,
               ignore_comments: bool = ..., include_dirs: list[str] | None = ...) -> FortranStringReader: ...
