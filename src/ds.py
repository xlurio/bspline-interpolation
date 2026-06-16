from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, TypeVar


T = TypeVar("T")


class Matrix(Generic[T]):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        initial: Sequence[T] = None,
        type: Callable[[Any], T] = int,
    ) -> None:
        if initial is None:
            initial = [type() for _ in range(n_rows * n_cols)]

        self.__mat = initial
        self.__n_rows = n_rows
        self.__n_cols = n_cols

    def __get_idx_by_pos(self, row: int, col: int) -> int:
        assert row >= 0
        assert row < self.__n_rows
        assert col >= 0
        assert col < self.__n_cols

        actual_idx = row * self.__n_cols + col

        assert actual_idx < self.__n_rows * self.__n_cols

        return actual_idx

    def get_by_pos(self, row: int, col: int) -> T:
        return self.__mat[self.__get_idx_by_pos(row, col)]

    def get_vector_by_row(self, row: int) -> Sequence[T]:
        assert row >= 0
        assert row < self.__n_rows

        new_vec = []
        start_idx = row * self.__n_cols
        end_idx = (row + 1) * self.__n_cols

        for mat_idx in range(start_idx, end_idx):
            new_vec.append(self.__mat[mat_idx])

        return new_vec

    def set_by_pos(self, row: int, col: int, value: T) -> None:
        self.__mat[self.__get_idx_by_pos(row, col)] = value

    @property
    def n_rows(self) -> int:
        return self.__n_rows

    @property
    def n_cols(self) -> int:
        return self.__n_cols


type ImageColorMode = Literal["L", "RGB", "RGBA"]


class ImageMatrix(Matrix[float]):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        color_mode: ImageColorMode,
        initial: Sequence[T] = None,
        type: Callable[[Any], T] = int,
    ) -> None:
        super().__init__(n_rows, n_cols, initial, type)
        self.__color_mode = color_mode

    @property
    def color_mode(self) -> ImageColorMode:
        return self.__color_mode


class MatrixTransposer(Generic[T]):
    def __init__(self, matrix: Matrix[T]) -> None:
        self.__source = matrix
        self.__transposed = Matrix(matrix.n_cols, matrix.n_rows)

    def transpose(self) -> Matrix[T]:
        if self.__source.n_rows == 0 or self.__source.n_cols == 0:
            return self.__transposed

        self.__transpose_recursive(
            start_row=0,
            end_row=self.__source.n_rows,
            start_col=0,
            end_col=self.__source.n_cols,
        )

        return self.__transposed

    def __transpose_recursive(
        self,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
    ) -> None:
        row_len = end_row - start_row
        col_len = end_col - start_col

        assert row_len > 0
        assert col_len > 0

        if row_len == 1 and col_len == 1:
            self.__transposed.set_by_pos(
                start_col,
                start_row,
                self.__source.get_by_pos(start_row, start_col),
            )
            return

        row_ranges = (
            ((start_row, end_row),)
            if row_len == 1
            else (
                (start_row, start_row + (row_len // 2)),
                (start_row + (row_len // 2), end_row),
            )
        )
        col_ranges = (
            ((start_col, end_col),)
            if col_len == 1
            else (
                (start_col, start_col + (col_len // 2)),
                (start_col + (col_len // 2), end_col),
            )
        )

        for row_range in row_ranges:
            for col_range in col_ranges:
                self.__transpose_recursive(*row_range, *col_range)


def transpose_matrix(matrix: Matrix[T]) -> Matrix[T]:
    return MatrixTransposer(matrix).transpose()
