import argparse
from PIL import Image
import math
import pathlib
import sys

from collections.abc import Sequence

from bsplin.ds import ImageColorMode, ImageMatrix
from bsplin.scaling import BSplineScalingBuilder


sys.setrecursionlimit(50000)


def make_matrix_from_image_path(image_path: pathlib.Path) -> ImageMatrix:
    path = pathlib.Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    with Image.open(path) as image:
        color_mode = image.mode

        if color_mode not in {"L", "RGB", "RGBA"}:
            raise ValueError(
                "Unsupported image mode "
                f"{color_mode!r}. Supported modes are: L, RGB, RGBA"
            )

        if color_mode != "L":
            image = image.convert("L")
            color_mode = "L"

        width, height = image.size
        normalized_pixels: list[float] = []

        pixel_source = (
            image.get_flattened_data()
            if hasattr(image, "get_flattened_data")
            else image.getdata()
        )

        for pixel in pixel_source:
            normalized_pixels.append(float(pixel) / 255.0)

    return ImageMatrix(
        height, width, color_mode=color_mode, initial=normalized_pixels, type=float
    )


def make_channel_matrices_from_image_path(
    image_path: pathlib.Path,
) -> tuple[list[ImageMatrix], ImageColorMode]:
    path = pathlib.Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    with Image.open(path) as image:
        color_mode = image.mode

        if color_mode not in {"L", "RGB", "RGBA"}:
            raise ValueError(
                "Unsupported image mode "
                f"{color_mode!r}. Supported modes are: L, RGB, RGBA"
            )

        width, height = image.size
        n_channels = 1 if color_mode == "L" else (3 if color_mode == "RGB" else 4)
        channel_pixels: list[list[float]] = [[] for _ in range(n_channels)]

        pixel_source = (
            image.get_flattened_data()
            if hasattr(image, "get_flattened_data")
            else image.getdata()
        )

        for pixel in pixel_source:
            if color_mode == "L":
                channel_pixels[0].append(float(pixel) / 255.0)
                continue

            for channel_idx, channel_value in enumerate(pixel):
                channel_pixels[channel_idx].append(float(channel_value) / 255.0)

    channels = [
        ImageMatrix(
            height,
            width,
            color_mode="L",
            initial=channel_vector,
            type=float,
        )
        for channel_vector in channel_pixels
    ]

    return channels, color_mode


def scale_image_matrix(source_image: ImageMatrix, scale: float) -> ImageMatrix:
    builder = BSplineScalingBuilder()
    builder.set_prefilter(source_image)
    builder.set_empty_scaled_image(source_image, scale)
    builder.set_scaled_image_intensities(source_image)
    return builder.get()


type EncodedPixel = int | tuple[int, int, int] | tuple[int, int, int, int]


class ImageMatrixSaver:
    def __init__(self, channels: Sequence[ImageMatrix], color_mode: ImageColorMode) -> None:
        self.__channels = tuple(channels)
        assert len(self.__channels) > 0
        self.__color_mode = color_mode

        if self.__color_mode not in {"L", "RGB", "RGBA"}:
            raise ValueError(
                "Unsupported image mode "
                f"{self.__color_mode!r}. Supported modes are: L, RGB, RGBA"
            )

        expected_n_channels = (
            1 if self.__color_mode == "L" else (3 if self.__color_mode == "RGB" else 4)
        )
        assert len(self.__channels) == expected_n_channels

        self.__n_rows = self.__channels[0].n_rows
        self.__n_cols = self.__channels[0].n_cols
        for channel in self.__channels[1:]:
            assert channel.n_rows == self.__n_rows and channel.n_cols == self.__n_cols

        total_pixels = self.__n_rows * self.__n_cols
        if self.__color_mode == "L":
            self.__pixels: list[EncodedPixel] = [0] * total_pixels
        elif self.__color_mode == "RGB":
            self.__pixels = [(0, 0, 0)] * total_pixels
        else:
            self.__pixels = [(0, 0, 0, 0)] * total_pixels

    def save_to_path(self, dest_path: pathlib.Path) -> None:
        path = pathlib.Path(dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.__fill_pixels_recursive(
            row_start=0,
            row_end=self.__n_rows,
            col_start=0,
            col_end=self.__n_cols,
        )

        output = Image.new(
            self.__color_mode,
            (self.__n_cols, self.__n_rows),
        )
        output.putdata(self.__pixels)
        output.save(path)

    def __fill_pixels_recursive(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> None:
        row_len = row_end - row_start
        col_len = col_end - col_start

        if row_len <= 0 or col_len <= 0:
            return

        if row_len == 1 and col_len == 1:
            self.__set_pixel(row_start, col_start)
            return

        if row_len >= col_len and row_len > 1:
            row_mid = row_start + (row_len // 2)
            self.__fill_pixels_recursive(row_start, row_mid, col_start, col_end)
            self.__fill_pixels_recursive(row_mid, row_end, col_start, col_end)
            return

        col_mid = col_start + (col_len // 2)
        self.__fill_pixels_recursive(row_start, row_end, col_start, col_mid)
        self.__fill_pixels_recursive(row_start, row_end, col_mid, col_end)

    def __set_pixel(self, row: int, col: int) -> None:
        channel_values = [
            self.__sanitize_and_clamp(channel.get_by_pos(row, col))
            for channel in self.__channels
        ]
        pixel_idx = row * self.__n_cols + col
        self.__pixels[pixel_idx] = self.__encode_pixel(channel_values)

    @staticmethod
    def __sanitize_and_clamp(value: float) -> float:
        if not math.isfinite(value):
            return 0.0

        return min(1.0, max(0.0, value))

    def __encode_pixel(self, normalized_values: Sequence[float]) -> EncodedPixel:
        if self.__color_mode == "L":
            return int(round(normalized_values[0] * 255.0))

        if self.__color_mode == "RGB":
            return (
                int(round(normalized_values[0] * 255.0)),
                int(round(normalized_values[1] * 255.0)),
                int(round(normalized_values[2] * 255.0)),
            )

        return (
            int(round(normalized_values[0] * 255.0)),
            int(round(normalized_values[1] * 255.0)),
            int(round(normalized_values[2] * 255.0)),
            int(round(normalized_values[3] * 255.0)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=pathlib.Path)
    parser.add_argument("dest_path", type=pathlib.Path)
    parser.add_argument("scale", type=float)
    args = parser.parse_args(sys.argv[1:])

    if args.scale <= 0:
        raise ValueError("Scale must be greater than 0")

    source_channels, source_color_mode = make_channel_matrices_from_image_path(
        args.source_path
    )
    scaled_channels = [
        scale_image_matrix(source_channel, args.scale)
        for source_channel in source_channels
    ]

    ImageMatrixSaver(scaled_channels, source_color_mode).save_to_path(args.dest_path)


if __name__ == "__main__":
    main()
