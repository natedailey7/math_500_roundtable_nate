"""
Rothermel surface fire ROS + flame length rasters using:
- DEM (for slope/aspect)
- LANDFIRE FBFM40 fuel model raster
- Wind speed raster
- Wind direction raster
- Fuel moisture raster (single band)

Uses the Pyretechnics library implementation of:
- Scott & Burgan (2005) standard fire behavior fuel models (FBFM40)
- Rothermel (1972) surface fire behavior
and returns max_spread_rate and max_flame_length per pixel.

Pyretechnics expects (per its docs):
- midflame_wind_speed: m/min
- upwind_direction: degrees clockwise from North
- slope: rise/run (unitless, e.g., 0.2 = 20% slope)
- aspect: degrees clockwise from North
- fuel_moisture tuple: (dead_1hr, dead_10hr, dead_100hr, dead_herb, live_herb, live_woody) as fractions (0-1)
See docs example in section 5.2.3. :contentReference[oaicite:2]{index=2}
"""

# =========================
# CONFIG — edit these paths
# =========================
DEM_PATH          = r"C:\path\to\dem.tif"
FUEL_MODEL_PATH   = r"C:\path\to\landfire_fbfm40.tif"      # integer codes like 101=GR1, 102=GR2, ...
WIND_SPEED_PATH   = r"C:\path\to\wind_speed.tif"
WIND_DIR_PATH     = r"C:\path\to\wind_dir.tif"
FUEL_MOISTURE_PATH= r"C:\path\to\fuel_moisture.tif"

ROS_OUT_PATH      = r"C:\path\to\ros_m_min.tif"            # Pyretechnics spread rate is in m/min
FL_OUT_PATH       = r"C:\path\to\flame_length_m.tif"       # meters

# =========================
# CONFIG — units/conventions
# =========================
# Wind speed raster units:
#   "mps" = meters/second
#   "kmhr" = km/hour
#   "mph" = miles/hour
WIND_SPEED_UNITS = "mps"

# Wind direction convention:
#   If your raster stores "direction wind is coming FROM" (meteorological, common): set True
#   If it stores "direction wind is blowing TO": set False
WIND_DIR_IS_FROM = True

# Wind direction is assumed degrees clockwise from North (0=N, 90=E).
# If your raster is 0=E, counterclockwise, etc., you must convert it before using this script.

# Fuel moisture raster units:
#   "fraction" = 0.08 means 8%
#   "percent"  = 8 means 8%
MOISTURE_UNITS = "fraction"

# =========================
# CONFIG — how to expand ONE moisture raster into the 6-part tuple
# =========================
# You only have one moisture raster; Pyretechnics/Rothermel needs 6 moistures.
# Common pragmatic approach: treat the raster as dead 1-hr moisture, and scale others.
DEAD_10HR_MULT  = 1.5
DEAD_100HR_MULT = 2.0

# Live moistures (constants). Adjust to scenario (curing/season).
LIVE_HERB_MOISTURE  = 0.90
LIVE_WOODY_MOISTURE = 0.60

# Clamp all moistures to [0, 3] just to avoid nonsense values
MOISTURE_MIN = 0.0
MOISTURE_MAX = 3.0

# =========================
# CONFIG — nonburnable handling
# =========================
# Common nonburnable codes you may see:
# 98 water, 99 rock; sometimes 90–97 are other nonburnables depending on workflow. :contentReference[oaicite:3]{index=3}
NONBURNABLE_CODES = {0, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}

# =========================
# END CONFIG
# =========================


import math
import numpy as np
import rasterio
import pyretechnics.conversion as conv
import pyretechnics.fuel_models as fm
import pyretechnics.surface_fire as sf


def _write_like(src_profile: dict, out_path: str, arr: np.ndarray, nodata: float = -9999.0) -> None:
    prof = src_profile.copy()
    prof.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        bigtiff="if_safer",
    )
    out = arr.astype("float32")
    out = np.where(np.isfinite(out), out, nodata).astype("float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)


def _slope_aspect_from_dem(dem_m: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      slope_rise_run (unitless)
      aspect_deg_clockwise_from_north (0..360)
    """
    dx = float(transform.a)
    dy = abs(float(transform.e))

    dz_dy, dz_dx = np.gradient(dem_m, dy, dx)

    # slope = tan(theta) = sqrt((dz/dx)^2+(dz/dy)^2)
    slope_rise_run = np.sqrt(dz_dx**2 + dz_dy**2).astype("float32")

    # aspect: direction of steepest downslope, clockwise from North
    aspect_rad = np.arctan2(-dz_dx, dz_dy)  # 0 at North
    aspect_rad = np.where(aspect_rad < 0, 2 * np.pi + aspect_rad, aspect_rad)
    aspect_deg = np.degrees(aspect_rad).astype("float32")

    return slope_rise_run, aspect_deg


def _wind_speed_to_m_min(ws: np.ndarray) -> np.ndarray:
    ws = ws.astype("float32")
    if WIND_SPEED_UNITS == "mps":
        return ws * 60.0
    if WIND_SPEED_UNITS == "kmhr":
        return conv.km_hr_to_m_min(ws)
    if WIND_SPEED_UNITS == "mph":
        # mph -> m/s -> m/min
        return ws * 0.44704 * 60.0
    raise ValueError(f"Unknown WIND_SPEED_UNITS: {WIND_SPEED_UNITS}")


def _upwind_dir_deg(wd_deg: np.ndarray) -> np.ndarray:
    """
    Pyretechnics wants upwind_direction (degrees clockwise from North).
    If input is "from", that is already upwind direction.
    If input is "to", convert to upwind by flipping 180 degrees.
    """
    wd = wd_deg.astype("float32") % 360.0
    if WIND_DIR_IS_FROM:
        return wd
    # wind blowing TO -> upwind is opposite
    return (wd + 180.0) % 360.0


def _moisture_tuple_from_one_raster(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = m.astype("float32")
    if MOISTURE_UNITS == "percent":
        m = m / 100.0
    m = np.clip(m, MOISTURE_MIN, MOISTURE_MAX)

    dead_1hr = m
    dead_10hr = np.clip(m * DEAD_10HR_MULT, MOISTURE_MIN, MOISTURE_MAX)
    dead_100hr = np.clip(m * DEAD_100HR_MULT, MOISTURE_MIN, MOISTURE_MAX)

    # dead_herbaceous: set to 0; fm.moisturize will handle for dynamic models per docs. :contentReference[oaicite:4]{index=4}
    dead_herb = np.zeros_like(m, dtype="float32")

    live_herb = np.full_like(m, LIVE_HERB_MOISTURE, dtype="float32")
    live_woody = np.full_like(m, LIVE_WOODY_MOISTURE, dtype="float32")

    return dead_1hr, dead_10hr, dead_100hr, dead_herb, live_herb, live_woody


def main() -> None:
    with rasterio.open(DEM_PATH) as dem_src, \
         rasterio.open(FUEL_MODEL_PATH) as fm_src, \
         rasterio.open(WIND_SPEED_PATH) as ws_src, \
         rasterio.open(WIND_DIR_PATH) as wd_src, \
         rasterio.open(FUEL_MOISTURE_PATH) as moist_src:

        profile = dem_src.profile

        # Quick sanity checks
        if (dem_src.width, dem_src.height) != (fm_src.width, fm_src.height):
            raise ValueError("Raster dimensions do not match. Reproject/resample to a common grid first.")
        if dem_src.transform != fm_src.transform:
            raise ValueError("Raster transforms do not match. Reproject/resample to a common grid first.")

        dem = dem_src.read(1).astype("float32")
        fuel_model = fm_src.read(1).astype("int32")
        wind_speed = ws_src.read(1).astype("float32")
        wind_dir = wd_src.read(1).astype("float32")
        moist = moist_src.read(1).astype("float32")

        # Nodata mask (combine)
        def bad(a: np.ndarray) -> np.ndarray:
            return ~np.isfinite(a) | (a <= -1e20)

        nodata_mask = bad(dem) | bad(wind_speed) | bad(wind_dir) | bad(moist)

        # Compute terrain
        slope_rise_run, aspect_deg = _slope_aspect_from_dem(dem, dem_src.transform)

        # Convert wind inputs
        midflame_wind_m_min = _wind_speed_to_m_min(wind_speed)
        upwind_deg = _upwind_dir_deg(wind_dir)

        # Expand moisture
        d1, d10, d100, dherb, lherb, lwood = _moisture_tuple_from_one_raster(moist)

        # Output arrays
        ros = np.full(dem.shape, np.nan, dtype="float32")  # m/min
        flame = np.full(dem.shape, np.nan, dtype="float32")  # m

        # Compute per unique fuel model code (fast-ish, avoids per-pixel Python loops)
        unique_codes = np.unique(fuel_model[~nodata_mask])
        for code in unique_codes:
            if int(code) in NONBURNABLE_CODES:
                continue

            try:
                base_fuel = fm.get_fuel_model(int(code))
            except Exception:
                # Unknown/unhandled fuel model code: leave as nodata
                continue

            sel = (fuel_model == code) & (~nodata_mask)
            if not np.any(sel):
                continue

            # Pull arrays for these pixels
            fm_tuple = (
                d1[sel],
                d10[sel],
                d100[sel],
                dherb[sel],
                lherb[sel],
                lwood[sel],
            )

            # Pyretechnics works with single fuel model dict + a moisture tuple
            # We'll compute behavior pixel-by-pixel for this code group.
            # (Still loops, but only over pixels of the same fuel model.)
            idxs = np.argwhere(sel)
            for (r, c) in idxs:
                m_tuple = (
                    float(d1[r, c]),
                    float(d10[r, c]),
                    float(d100[r, c]),
                    float(dherb[r, c]),
                    float(lherb[r, c]),
                    float(lwood[r, c]),
                )

                mf = fm.moisturize(base_fuel, m_tuple)
                sf_min = sf.calc_surface_fire_behavior_no_wind_no_slope(mf)
                sf_max = sf.calc_surface_fire_behavior_max(
                    sf_min,
                    float(midflame_wind_m_min[r, c]),
                    float(upwind_deg[r, c]),
                    float(slope_rise_run[r, c]),
                    float(aspect_deg[r, c]),
                    surface_lw_ratio_model="rothermel",
                )

                ros[r, c] = float(sf_max["max_spread_rate"])       # m/min :contentReference[oaicite:5]{index=5}
                flame[r, c] = float(sf_max["max_flame_length"])    # m :contentReference[oaicite:6]{index=6}

        _write_like(profile, ROS_OUT_PATH, ros)
        _write_like(profile, FL_OUT_PATH, flame)

    print("Done.")
    print(f"ROS (m/min): {ROS_OUT_PATH}")
    print(f"Flame length (m): {FL_OUT_PATH}")


if __name__ == "__main__":
    main()