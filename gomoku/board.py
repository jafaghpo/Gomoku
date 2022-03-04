import numpy as np
from pandas import DataFrame
from collections import namedtuple
from typing import Any

Position = namedtuple("Position", ("y", "x"))


class Board(np.ndarray):
    """
    Subclass of numpy.ndarray that represents a state of the game.
    """

    def __new__(cls, shape, dtype=int, order="C"):
        obj = np.zeros(shape, dtype=dtype, order=order).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # add new attributes created in new method here, like so:
        # self.new_attribute = getattr(obj, 'new_attribute', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        This implementation of __array_ufunc__ makes sure that all custom attributes
        are maintained when a ufunc operation is performed on our class.
        """

        args = ((i.view(np.ndarray) if isinstance(i, Board) else i) for i in inputs)
        outputs = kwargs.pop("out", None)
        if outputs:
            kwargs["out"] = tuple(
                (o.view(np.ndarray) if isinstance(o, Board) else o) for o in outputs
            )
        else:
            outputs = (None,) * ufunc.nout
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            return
        if ufunc.nout == 1:
            results = (results,)
        results = tuple(
            (self._copy_attrs_to(result) if output is None else output)
            for result, output in zip(results, outputs)
        )
        return results[0] if len(results) == 1 else results

    def _copy_attrs_to(self, target):
        """
        Copies all attributes of self to the target object.
        target must be a (subclass of) ndarray
        """
        target = target.view(Board)
        try:
            target.__dict__.update(self.__dict__)
        except AttributeError:
            pass
        return target

    def __str__(self) -> str:
        player_repr = {0: ".", 1: "X", 2: "O"}
        df = DataFrame([map(lambda index: player_repr[index], row) for row in self])
        return f"{df}"
