#!/usr/bin/env python3
"""
split_dem.py

Split a DEM GeoTIFF into four equal-ish quadrants using hard-coded paths.
Requires: rasterio
"""

from pathlib import Path

import rasterio
from rasterio.windows import Window

# Hard-coded paths (no command-line args).
DEM_PATH = Path(r"data\USGS_13_n40w121_20250514.tif")
OUTPUT_DIR = Path(r"data\dem_quadrants")

def split_into_quadrants(src_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        height, width = src.height, src.width
        if height < 2 or width < 2:
            raise ValueError("DEM must be at least 2x2 pixels.")

        mid_row = (height + 1) // 2  # ceil division keeps coverage for odd sizes
        mid_col = (width + 1) // 2

        row_ranges = [(0, mid_row), (mid_row, height)]
        col_ranges = [(0, mid_col), (mid_col, width)]
        quadrants = [
            ("NW", 0, 0),
            ("NE", 0, 1),
            ("SW", 1, 0),
            ("SE", 1, 1),
        ]

        for label, row_idx, col_idx in quadrants:
            row_start, row_stop = row_ranges[row_idx]
            col_start, col_stop = col_ranges[col_idx]
            window = Window.from_slices(
                (row_start, row_stop),
                (col_start, col_stop),
            )

            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                height=row_stop - row_start,
                width=col_stop - col_start,
                transform=src.window_transform(window),
            )

            out_path = output_dir / f"{src_path.stem}_{label}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
            print(f"Wrote {out_path}")


def main() -> None:
    src_path = DEM_PATH
    if not src_path.exists():
        raise FileNotFoundError(f"No DEM found at {src_path}")

    output_dir = OUTPUT_DIR
    split_into_quadrants(src_path, output_dir)


if __name__ == "__main__":
    main()

