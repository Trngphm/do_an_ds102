from typing import Any, Dict, Iterable, Iterator, Tuple
from tabulate import tabulate

class Registry(Iterable[Tuple[str, Any]]):
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        if obj is None:
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        else:
            name = obj.__name__
            self._do_register(name, obj)
            return obj  # <-- bạn thiếu dòng này

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return f"Registry of {self._name}:\n" + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    __str__ = __repr__
