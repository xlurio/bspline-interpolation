import math

from bsplin.ds import ImageMatrix, Matrix, transpose_matrix
from bsplin.prefiltering import (
    AntiCausalPreFilterVectorBuilder,
    CausalPreFilterVectorBuilder,
    PreFilterFactory,
)
from bsplin import dba


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
            lambda dimensionality, row_idx: CausalPreFilterVectorBuilder(
                dimensionality
            ),
            image.n_rows,
            image.n_cols,
        ).make(image)

        row_filtered = PreFilterFactory(
            lambda dimensionality, row_idx: AntiCausalPreFilterVectorBuilder(
                dimensionality, row_prefilter.get_vector_by_row(row_idx)
            ),
            image.n_rows,
            image.n_cols,
        ).make(row_prefilter)

        col_input = transpose_matrix(row_filtered)

        col_prefilter = PreFilterFactory(
            lambda dimensionality, row_idx: CausalPreFilterVectorBuilder(
                dimensionality
            ),
            col_input.n_rows,
            col_input.n_cols,
        ).make(col_input)

        col_filtered = PreFilterFactory(
            lambda dimensionality, row_idx: AntiCausalPreFilterVectorBuilder(
                dimensionality, col_prefilter.get_vector_by_row(row_idx)
            ),
            col_input.n_rows,
            col_input.n_cols,
        ).make(col_prefilter)

        self.__coefs = transpose_matrix(col_filtered)

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

        dba.scaling_function_already_seen_params.clear()

        self.__recursive_set_scale_image_intensities(
            image,
            start_dest_row=0,
            end_dest_row=self.__scaled_image.n_rows - 1,
            start_dest_col=0,
            end_dest_col=self.__scaled_image.n_cols - 1,
            start_neighborhood_row_delta=-2,
            end_neighborhood_row_delta=1,
            start_neighborhood_col_delta=-2,
            end_neighborhood_col_delta=1,
        )
        self.__were_intensities_calculated = True

    def __get_spline_kernel(self, x: float) -> float:
        abs_x = abs(x)

        if abs_x < 1:
            return (4 - (6 * (abs_x**2)) + (3 * (abs_x**3))) / 6

        if abs_x < 2:
            return ((2 - abs_x) ** 3) / 6

        return 0

    def __recursive_set_scale_image_intensities(
        self,
        image: ImageMatrix,
        start_dest_row: int,
        end_dest_row: int,
        start_dest_col: int,
        end_dest_col: int,
        start_neighborhood_row_delta: int,
        end_neighborhood_row_delta: int,
        start_neighborhood_col_delta: int,
        end_neighborhood_col_delta: int,
    ) -> None:
        is_base_case = (
            start_dest_row == end_dest_row
            and start_dest_col == end_dest_col
            and start_neighborhood_row_delta == end_neighborhood_row_delta
            and start_neighborhood_col_delta == end_neighborhood_col_delta
        )

        assert (
            start_dest_row <= end_dest_row
            and start_dest_col <= end_dest_col
            and start_neighborhood_row_delta <= end_neighborhood_row_delta
            and start_neighborhood_col_delta <= end_neighborhood_col_delta
        )

        curr_params = (
            start_dest_row,
            end_dest_row,
            start_dest_col,
            end_dest_col,
            start_neighborhood_row_delta,
            end_neighborhood_row_delta,
            start_neighborhood_col_delta,
            end_neighborhood_col_delta,
        )
        assert curr_params not in dba.scaling_function_already_seen_params
        dba.scaling_function_already_seen_params.add(curr_params)

        if is_base_case:
            source_row_float = (
                start_dest_row * image.n_rows / self.__scaled_image.n_rows
            )
            source_col_float = (
                start_dest_col * image.n_cols / self.__scaled_image.n_cols
            )
            source_row_base = math.floor(source_row_float)
            source_col_base = math.floor(source_col_float)

            var_k = source_row_base + start_neighborhood_row_delta
            var_l = source_col_base + start_neighborhood_col_delta
            clamped_k = max(0, min(self.__coefs.n_rows - 1, var_k))
            clamped_l = max(0, min(self.__coefs.n_cols - 1, var_l))

            scaled_coord = (start_dest_row, start_dest_col)

            self.__scaled_image.set_by_pos(
                *scaled_coord,
                self.__scaled_image.get_by_pos(*scaled_coord)
                + self.__coefs.get_by_pos(
                    clamped_k,
                    clamped_l,
                )
                * self.__get_spline_kernel(
                    source_row_float - var_k
                )
                * self.__get_spline_kernel(
                    source_col_float - var_l
                ),
            )

            return

        # Divide
        split_dest_row = start_dest_row != end_dest_row
        split_dest_col = start_dest_col != end_dest_col
        split_neighborhood_row_delta = (
            start_neighborhood_row_delta != end_neighborhood_row_delta
        )
        split_neighborhood_col_delta = (
            start_neighborhood_col_delta != end_neighborhood_col_delta
        )

        if split_dest_row:
            mid_dest_row = ((end_dest_row - start_dest_row) // 2) + start_dest_row

        if split_dest_col:
            mid_dest_col = ((end_dest_col - start_dest_col) // 2) + start_dest_col

        if split_neighborhood_row_delta:
            mid_neighborhood_row_delta = (
                (end_neighborhood_row_delta - start_neighborhood_row_delta) // 2
            ) + start_neighborhood_row_delta

        if split_neighborhood_col_delta:
            mid_neighborhood_col_delta = (
                (end_neighborhood_col_delta - start_neighborhood_col_delta) // 2
            ) + start_neighborhood_col_delta

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
                for neighborhood_row_delta_range in (
                    {
                        (start_neighborhood_row_delta, mid_neighborhood_row_delta),
                        (mid_neighborhood_row_delta + 1, end_neighborhood_row_delta),
                    }
                    if split_neighborhood_row_delta
                    else {
                        (start_neighborhood_row_delta, end_neighborhood_row_delta),
                    }
                ):
                    for neighborhood_col_delta_range in (
                        {
                            (start_neighborhood_col_delta, mid_neighborhood_col_delta),
                            (
                                mid_neighborhood_col_delta + 1,
                                end_neighborhood_col_delta,
                            ),
                        }
                        if split_neighborhood_col_delta
                        else {
                            (start_neighborhood_col_delta, end_neighborhood_col_delta),
                        }
                    ):
                        self.__recursive_set_scale_image_intensities(
                            image,
                            *dest_row_range,
                            *dest_col_range,
                            *neighborhood_row_delta_range,
                            *neighborhood_col_delta_range,
                        )

    def get(self) -> ImageMatrix:
        assert (
            self.__were_coefs_calculated
            and self.__was_empty_scaled_image_created
            and self.__were_intensities_calculated
        )

        return self.__scaled_image
