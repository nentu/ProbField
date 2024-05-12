from functools import lru_cache
from pprint import pprint

import numpy as np
from scipy.ndimage import shift
from matplotlib import pyplot

from config import probable_movements


def show_mtrx(mtrx):
    pprint(mtrx)

    pyplot.matshow(mtrx)
    pyplot.show()


class FieldCreator:
    def __init__(self,
                 probable_movements,
                 size: int = None,
                 pose_start=None
                 ):
        self.probable_movements = probable_movements

        self.size = size
        self.x_start, self.y_start = pose_start

    @lru_cache()
    def get_field(self, t):
        if t == 0:
            mtrx = np.full([self.size, self.size], 0, dtype=np.float64)
            mtrx[self.x_start, self.y_start] = 1
            return mtrx
        prev_field = self.get_field(t - 1)

        movements = list()

        for prob, shift_dir in probable_movements:
            movements.append(
                shift(prev_field, shift_dir) * prob
            )

        res = movements[0]
        for i in movements[1:]:
            res += i
        return res


if __name__ == "__main__":
    n = 29
    field = FieldCreator(probable_movements, n, [n // 2, n // 2])

    show_mtrx(
        field.get_field(5)
    )
