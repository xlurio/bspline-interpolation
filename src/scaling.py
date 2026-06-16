import math

from ds import ImageMatrix, Matrix
from prefiltering import (
    AntiCausalPreFilterVectorBuilder,
    CausalPreFilterVectorBuilder,
    PreFilterFactory,
)


class BSplineScalingBuilder:
    __coefs: Matrix[float]
    __scaled_image: ImageMatrix

    def __init__(self) -> None:
        self.__coefs = Matrix(1, 1)
        self.__scaled_image = ImageMatrix(1, 1, color_mode="L", type=float)
        self.__were_coefs_calculated = False
        self.__was_empty_scaled_image_created = False
        self.__were_intensities_calculated = False

    def set_prefilter(self, image: ImageMatrix) -> None:
        row_prefilter = PreFilterFactory(
            lambda dimensionality, row_idx: CausalPreFilterVectorBuilder(dimensionality),
            image.n_rows,
            image.n_cols,
        ).make(image)

        self.__coefs = PreFilterFactory(
            lambda dimensionality, row_idx: AntiCausalPreFilterVectorBuilder(
                dimensionality, row_prefilter.get_vector_by_row(row_idx)
            ),
            image.n_rows,
            image.n_cols,
        ).make(row_prefilter)

        self.__were_coefs_calculated = True

    def set_empty_scaled_image(self, image: ImageMatrix, scale: float) -> None:

        self.__scaled_image = ImageMatrix(
            n_rows=max(1, round(image.n_rows * scale)),
            n_cols=max(1, round(image.n_cols * scale)),
            color_mode=image.color_mode,
            type=float,
        )
        self.__was_empty_scaled_image_created = True

    def set_scaled_image_intensities(self, image: ImageMatrix) -> None:
        assert self.__were_coefs_calculated and self.__was_empty_scaled_image_created

        self.__recursive_set_scale_image_intensities(
            image,
            start_dest_row=0,
            end_dest_row=self.__scaled_image.n_rows - 1,
            start_dest_col=0,
            end_dest_col=self.__scaled_image.n_cols - 1,
            start_neighborhood_row=0,
            end_neighborhood_row=image.n_rows - 1,
            start_neighborhood_col=0,
            end_neighborhood_col=image.n_cols - 1,
        )
        self.__were_intensities_calculated = True

    def __get_spline_kernel(self, x: int) -> float:
        if 0 <= x < 1:
            return (2 / 3) - (abs(x) ** 2) + ((abs(x) ** 3) / 2)

        elif 1 <= x < 2:
            return (2 - (abs(x) ** 2)) / 6

        return 0

    def __recursive_set_scale_image_intensities(
        self,
        image: ImageMatrix,
        start_dest_row: int,
        end_dest_row: int,
        start_dest_col: int,
        end_dest_col: int,
        start_neighborhood_row: int,
        end_neighborhood_row: int,
        start_neighborhood_col: int,
        end_neighborhood_col: int,
    ) -> None:
        is_base_case = (
            start_dest_row == end_dest_row
            and start_dest_col == end_dest_col
            and start_neighborhood_row == end_neighborhood_row
            and start_neighborhood_col == end_neighborhood_col
        )

        if is_base_case:
            source_row = math.floor(
                start_dest_row * image.n_rows / self.__scaled_image.n_rows
            )
            source_col = math.floor(
                start_dest_col * image.n_cols / self.__scaled_image.n_cols
            )

            scaled_coord = (start_dest_row, start_dest_col)

            self.__scaled_image.set_by_pos(
                *scaled_coord,
                self.__scaled_image.get_by_pos(*scaled_coord)
                + self.__coefs.get_by_pos(
                    start_neighborhood_row, start_neighborhood_col
                )
                * self.__get_spline_kernel(source_row - start_neighborhood_row)
                * self.__get_spline_kernel(source_col - start_neighborhood_col),
            )

            return

        # Divide
        split_dest_row = start_dest_row != end_dest_row
        split_dest_col = start_dest_col != end_dest_col
        split_neighborhood_row = start_neighborhood_row != end_neighborhood_row
        split_neighborhood_col = start_neighborhood_col != end_neighborhood_col

        if split_dest_row:
            mid_dest_row = ((end_dest_row - start_dest_row) // 2) + start_dest_row

        if split_dest_col:
            mid_dest_col = ((end_dest_col - start_dest_col) // 2) + start_dest_col

        if split_neighborhood_row:
            mid_neighborhood_row = (
                (end_neighborhood_row - start_neighborhood_row) // 2
            ) + start_neighborhood_row

        if split_neighborhood_col:
            mid_neighborhood_col = (
                (end_neighborhood_col - start_neighborhood_col) // 2
            ) + start_neighborhood_col

        # Conquer
        for dest_row_range in (
            {
                (start_dest_row, mid_dest_row),
                (mid_dest_row + 1, end_dest_row),
            }
            if split_dest_row
            else {
                (start_dest_row, end_dest_row),
            }
        ):
            for dest_col_range in (
                {
                    (start_dest_col, mid_dest_col),
                    (mid_dest_col + 1, end_dest_col),
                }
                if split_dest_col
                else {
                    (start_dest_col, end_dest_col),
                }
            ):
                for neighborhood_row_range in (
                    {
                        (start_neighborhood_row, mid_neighborhood_row),
                        (mid_neighborhood_row + 1, end_neighborhood_row),
                    }
                    if split_neighborhood_row
                    else {
                        (start_neighborhood_row, end_neighborhood_row),
                    }
                ):
                    for neighborhood_col_range in (
                        {
                            (start_neighborhood_col, mid_neighborhood_col),
                            (mid_neighborhood_col + 1, end_neighborhood_col),
                        }
                        if split_neighborhood_col
                        else {
                            (start_neighborhood_col, end_neighborhood_col),
                        }
                    ):
                        self.__recursive_set_scale_image_intensities(
                            image,
                            *dest_row_range,
                            *dest_col_range,
                            *neighborhood_row_range,
                            *neighborhood_col_range,
                        )

    def get(self) -> ImageMatrix:
        assert (
            self.__were_coefs_calculated
            and self.__was_empty_scaled_image_created
            and self.__were_intensities_calculated
        )

        return self.__scaled_image
