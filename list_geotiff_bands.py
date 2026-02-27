from pathlib import Path

import rasterio


TIF_PATH = Path(r"data\LF\LF_aoi.tif")


def list_bands(tif_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        print(f"File: {tif_path}")
        print(f"Driver: {src.driver}")
        print(f"Size: {src.width} x {src.height}")
        print(f"Band count: {src.count}")
        print()

        for band_index in range(1, src.count + 1):
            description = src.descriptions[band_index - 1] if src.descriptions else None
            dtype = src.dtypes[band_index - 1]
            name = description if description else "(no description)"
            print(f"Band {band_index}: {name} | dtype={dtype}")


def main() -> None:
    if not TIF_PATH.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {TIF_PATH}")

    list_bands(TIF_PATH)


if __name__ == "__main__":
    main()
