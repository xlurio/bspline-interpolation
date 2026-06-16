import abc
from collections.abc import Callable, Sequence
import math

from ds import Matrix


type CausalPreFilter = Matrix[float]

Z_1 = -0.2679491924311228


class PreFilterVectorBuilder(abc.ABC):
    _coeffs: list[float]
    _curr_idx: int
    _dimensionality: int

    @property
    @abc.abstractmethod
    def next_idx(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @property
    def was_built(self) -> bool:
        return self.__was_built

    @abc.abstractmethod
    def _calc_basis_coefficient(self, intensity_vector: Sequence[float]) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _advance_idx(self) -> None:
        raise NotImplementedError

    def set_basis_coefficient(self, intensity_vector: Sequence[float]) -> None:
        assert len(intensity_vector) == self._dimensionality
        assert not self.is_ready

        self._coeffs.append(self._calc_basis_coefficient(intensity_vector))
        self._advance_idx()

    def set_next_coef(self, next_intensity: float) -> None:
        assert not self.is_ready

        self._coeffs.append(self._calc_next_coef(next_intensity))
        self._advance_idx()

    @abc.abstractmethod
    def _calc_next_coef(self, next_intensity: float) -> float:
        raise NotImplementedError

    def build(self) -> tuple[float, ...]:
        assert self.is_ready

        result = tuple(self._coeffs)
        self.__was_built = True

        return result


class CausalPreFilterVectorBuilder(PreFilterVectorBuilder):
    PRECISION = 1e-2

    def __init__(self, dimensionality: int) -> None:
        self._coeffs = []
        self._curr_idx = 0
        self._dimensionality = dimensionality
        self._PreFilterVectorBuilder__was_built = False

    @property
    def next_idx(self) -> int:
        return self._curr_idx

    @property
    def is_ready(self) -> bool:
        return self._curr_idx >= self._dimensionality

    def _calc_basis_coefficient(self, intensity_vector: Sequence[float]) -> float:
        basis_coef = 0
        max_k = int(math.log10(self.__class__.PRECISION) / math.log10(abs(Z_1)))

        for k in range(max_k + 1):
            basis_coef += intensity_vector[k % self._dimensionality] * (Z_1**k)

        return basis_coef

    def _advance_idx(self):
        self._curr_idx += 1

    def _calc_next_coef(self, next_intensity: float) -> float:
        return next_intensity + Z_1 * self._coeffs[self._curr_idx - 1]


class AntiCausalPreFilterVectorBuilder(PreFilterVectorBuilder):
    def __init__(self, dimensionality: int, causal_filter_vec: Sequence[float]) -> None:
        assert len(causal_filter_vec) == dimensionality

        self._coeffs = [0.0] * dimensionality
        self._curr_idx = dimensionality - 1
        self._dimensionality = dimensionality
        self.__causal_filter_vec = causal_filter_vec
        self._PreFilterVectorBuilder__was_built = False

    @property
    def next_idx(self) -> int:
        return self._curr_idx

    @property
    def is_ready(self) -> bool:
        return self._curr_idx < 0

    def set_basis_coefficient(self, intensity_vector: Sequence[float]) -> None:
        assert len(intensity_vector) == self._dimensionality
        assert self._curr_idx == self._dimensionality - 1

        self._coeffs[self._curr_idx] = self._calc_basis_coefficient(intensity_vector)
        self._advance_idx()

    def set_next_coef(self, next_intensity: float) -> None:
        del next_intensity
        assert not self.is_ready

        self._coeffs[self._curr_idx] = self._calc_next_coef(0.0)
        self._advance_idx()

    def _calc_basis_coefficient(self, intensity_vector: Sequence[float]) -> float:
        del intensity_vector

        first_factor = Z_1 / (1 - Z_1**2)
        snd_factor = (
            self.__causal_filter_vec[self._curr_idx]
            + Z_1 * self.__causal_filter_vec[self._curr_idx - 1]
        )

        return first_factor * snd_factor

    def _advance_idx(self):
        self._curr_idx -= 1

    def _calc_next_coef(self, next_intensity: float) -> float:
        del next_intensity
        return Z_1 * (
            self._coeffs[self._curr_idx + 1] - self.__causal_filter_vec[self._curr_idx]
        )


class PreFilterFactory:
    def __init__(
        self,
        builder_factory: Callable[[int, int], PreFilterVectorBuilder],
        length: int,
        dimensionality: int,
    ) -> None:
        self.__length = length
        self.__dimensionality = dimensionality
        self.__builder_factory = builder_factory
        self.__coef_mat: CausalPreFilter = Matrix(
            self.__length, self.__dimensionality, type=float
        )

    def make(self, intensity_mat: Matrix[float]) -> CausalPreFilter:
        assert intensity_mat.n_rows == self.__length
        assert intensity_mat.n_cols == self.__dimensionality

        for row_idx in range(self.__length):
            intensity_vector = intensity_mat.get_vector_by_row(row_idx)
            builder = self.__builder_factory(self.__dimensionality, row_idx)

            builder.set_basis_coefficient(intensity_vector)

            while not builder.is_ready:
                builder.set_next_coef(intensity_vector[builder.next_idx])

            prefiltered_vector = builder.build()

            for col_idx in range(self.__dimensionality):
                self.__coef_mat.set_by_pos(
                    row_idx,
                    col_idx,
                    prefiltered_vector[col_idx],
                )

        return self.__coef_mat
