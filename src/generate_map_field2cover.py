"""Convert a Fields2Cover GeoJSON field to a pathCoverage occupancy map.

Fields2Cover represents a field as GeoJSON polygons: the first ring is the
field boundary and the remaining rings are holes.  pathCoverage uses a uint8
occupancy image whose origin is at the lower-left, with 255 for free space and
0 for obstacles.  This module bridges those two representations.

The conversion is deliberately independent of Fields2Cover's Python binding.
Only numpy and Pillow (already used by pathCoverage) are required.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from PIL import Image, ImageDraw


CoordinateSystem = Literal["auto", "planar", "wgs84"]
Point = tuple[float, float]
Polygon = list[list[Point]]


@dataclass(frozen=True)
class MapMetadata:
    """Information needed to map source coordinates onto the output image.

    For a projected point ``(x, y)``, its pathCoverage world coordinate is
    ``(x - origin_x, y - origin_y)``.  Dividing that result by ``pixel_size``
    gives the lower-left-origin pixel coordinate.
    """

    pixel_size: float
    width_pixels: int
    height_pixels: int
    origin_x: float
    origin_y: float
    source_coordinate_system: str
    projected_coordinate_system: str
    utm_zone: int | None = None
    utm_hemisphere: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point(raw_point: Sequence[Any]) -> Point:
    if len(raw_point) < 2:
        raise ValueError(f"坐标点至少需要 x、y 两个分量：{raw_point!r}")
    try:
        point = (float(raw_point[0]), float(raw_point[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无效坐标点：{raw_point!r}") from exc
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"坐标点必须是有限数值：{raw_point!r}")
    return point


def _parse_polygon(raw_rings: Sequence[Any]) -> Polygon:
    if not raw_rings:
        raise ValueError("Polygon 的 coordinates 不能为空。")
    polygon: Polygon = []
    for raw_ring in raw_rings:
        ring = [_point(raw_point) for raw_point in raw_ring]
        # GeoJSON rings normally repeat the first point.  ImageDraw closes an
        # open ring as well, so accept both forms while rejecting degenerate
        # rings.
        unique_points = set(ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring)
        if len(unique_points) < 3:
            raise ValueError("Polygon 的每个 ring 至少需要三个不同的点。")
        polygon.append(ring)
    return polygon


def load_fields2cover_json(json_path: str | Path) -> list[Polygon]:
    """Load all Polygon/MultiPolygon features from a Fields2Cover JSON file."""

    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误：{path}（{exc}）") from exc

    if document.get("type") != "FeatureCollection":
        raise ValueError("Fields2Cover JSON 顶层必须是 FeatureCollection。")

    polygons: list[Polygon] = []
    for feature_index, feature in enumerate(document.get("features", [])):
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry:
            continue
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        try:
            if geometry_type == "Polygon":
                polygons.append(_parse_polygon(coordinates))
            elif geometry_type == "MultiPolygon":
                polygons.extend(_parse_polygon(raw_polygon) for raw_polygon in coordinates)
            else:
                raise ValueError(f"不支持的 geometry 类型 {geometry_type!r}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {feature_index} 个 feature 无效：{exc}") from exc

    if not polygons:
        raise ValueError(f"{path} 中没有可用的 Polygon 或 MultiPolygon。")
    return polygons


def _all_points(polygons: Iterable[Polygon]) -> list[Point]:
    return [point for polygon in polygons for ring in polygon for point in ring]


def _looks_like_wgs84(points: Sequence[Point]) -> bool:
    """Conservatively distinguish GPS data from small local test fields.

    Fields2Cover JSON does not store a CRS.  Its sample GPS fields span much
    less than one degree and are far from (0, 0), while local examples such as
    test_ring.json span several coordinate units.  Ambiguous inputs should use
    the explicit ``coordinate_system`` argument.
    """

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    valid_lon_lat = all(-180.0 <= x <= 180.0 for x in xs) and all(-90.0 <= y <= 90.0 for y in ys)
    small_extent = max(xs) - min(xs) <= 1.0 and max(ys) - min(ys) <= 1.0
    away_from_origin = abs(sum(xs) / len(xs)) > 20.0 or abs(sum(ys) / len(ys)) > 20.0
    return valid_lon_lat and small_extent and away_from_origin


def _utm_zone(longitude: float) -> int:
    return min(60, max(1, int(math.floor((longitude + 180.0) / 6.0)) + 1))


def _wgs84_to_utm(longitude: float, latitude: float, zone: int) -> Point:
    """Project WGS84 longitude/latitude to UTM using the standard TM series."""

    if not (-80.0 <= latitude <= 84.0):
        raise ValueError("UTM 仅支持纬度范围 [-80, 84]；请改用 planar 或预先投影数据。")

    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    scale = 0.9996

    lat = math.radians(latitude)
    lon = math.radians(longitude)
    central_lon = math.radians((zone - 1) * 6 - 180 + 3)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n = semi_major / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = second_eccentricity_sq * cos_lat * cos_lat
    a = cos_lat * (lon - central_lon)
    e2 = eccentricity_sq
    meridian = semi_major * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat)
        - (35 * e2**3 / 3072) * math.sin(6 * lat)
    )

    easting = scale * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * second_eccentricity_sq) * a**5 / 120
    ) + 500000.0
    northing = scale * (
        meridian
        + n
        * tan_lat
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * second_eccentricity_sq) * a**6 / 720
        )
    )
    if latitude < 0:
        northing += 10000000.0
    return easting, northing


def _project_polygons(
    polygons: list[Polygon], coordinate_system: CoordinateSystem
) -> tuple[list[Polygon], str, int | None, str | None]:
    points = _all_points(polygons)
    if coordinate_system not in ("auto", "planar", "wgs84"):
        raise ValueError("coordinate_system 必须是 auto、planar 或 wgs84。")
    use_wgs84 = coordinate_system == "wgs84" or (
        coordinate_system == "auto" and _looks_like_wgs84(points)
    )
    if not use_wgs84:
        return polygons, "planar", None, None

    mean_lon = sum(point[0] for point in points) / len(points)
    mean_lat = sum(point[1] for point in points) / len(points)
    zone = _utm_zone(mean_lon)
    hemisphere = "N" if mean_lat >= 0 else "S"
    projected = [
        [[_wgs84_to_utm(lon, lat, zone) for lon, lat in ring] for ring in polygon]
        for polygon in polygons
    ]
    return projected, f"UTM zone {zone}{hemisphere} (WGS84)", zone, hemisphere


def generate_map_from_fields2cover(
    json_path: str | Path,
    pixel_size: float,
    *,
    coordinate_system: CoordinateSystem = "auto",
    padding: float = 0.0,
) -> tuple[np.ndarray, MapMetadata]:
    """Rasterize a Fields2Cover JSON file into a pathCoverage occupancy map.

    Args:
        json_path: Fields2Cover-style GeoJSON FeatureCollection.
        pixel_size: Metres (or planar source units) represented by one pixel.
        coordinate_system: ``wgs84`` projects GPS input to UTM, ``planar``
            preserves x/y, and ``auto`` recognizes the repository sample data.
        padding: Obstacle margin around the field, in projected world units.

    Returns:
        ``(occupancy_map, metadata)``.  The array has lower-left origin:
        ``occupancy_map[y, x]`` is 255 in the field and 0 outside/in holes.
    """

    if not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size 必须是大于 0 的有限数值。")
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("padding 必须是非负有限数值。")

    polygons = load_fields2cover_json(json_path)
    projected, projected_crs, zone, hemisphere = _project_polygons(polygons, coordinate_system)
    points = _all_points(projected)
    min_x = min(point[0] for point in points) - padding
    min_y = min(point[1] for point in points) - padding
    max_x = max(point[0] for point in points) + padding
    max_y = max(point[1] for point in points) + padding
    width = max(1, int(math.ceil((max_x - min_x) / pixel_size)))
    height = max(1, int(math.ceil((max_y - min_y) / pixel_size)))

    # Build one mask per polygon before unioning them, so a hole in one feature
    # cannot erase the free region contributed by an overlapping feature.
    free_mask = np.zeros((height, width), dtype=np.bool_)

    def image_point(point: Point) -> tuple[float, float]:
        x = (point[0] - min_x) / pixel_size
        # Pillow has a top-left origin.  y=height is intentionally allowed for
        # the lower geometric boundary; polygon clipping fills the last row.
        y = height - (point[1] - min_y) / pixel_size
        return x, y

    for polygon in projected:
        image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(image)
        draw.polygon([image_point(point) for point in polygon[0]], fill=1)
        for hole in polygon[1:]:
            draw.polygon([image_point(point) for point in hole], fill=0)
        # Flip into pathCoverage's lower-left-origin ndarray convention.
        free_mask |= np.flipud(np.asarray(image, dtype=np.bool_))

    occupancy_map = np.where(free_mask, 255, 0).astype(np.uint8)
    detected_source = "wgs84" if zone is not None else "planar"
    metadata = MapMetadata(
        pixel_size=pixel_size,
        width_pixels=width,
        height_pixels=height,
        origin_x=min_x,
        origin_y=min_y,
        source_coordinate_system=detected_source,
        projected_coordinate_system=projected_crs,
        utm_zone=zone,
        utm_hemisphere=hemisphere,
    )
    return occupancy_map, metadata


def save_map_png(occupancy_map: np.ndarray, output_path: str | Path) -> None:
    """Save a lower-left-origin pathCoverage map as an 8-bit grayscale PNG."""

    if occupancy_map.ndim != 2:
        raise ValueError("occupancy_map 必须是二维数组。")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(occupancy_map).astype(np.uint8), mode="L").save(path)


def save_metadata(metadata: MapMetadata, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")


# CamelCase entry points keep this module consistent with generate_map.py.
generateMapFromFields2Cover = generate_map_from_fields2cover
saveMapPNG = save_map_png


def generateMap(
    json_path: str | Path,
    pixel_size: float,
    *,
    coordinate_system: CoordinateSystem = "auto",
    padding: float = 0.0,
) -> np.ndarray:
    """Compatibility wrapper returning only the occupancy array."""

    occupancy_map, _ = generate_map_from_fields2cover(
        json_path,
        pixel_size,
        coordinate_system=coordinate_system,
        padding=padding,
    )
    return occupancy_map


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Fields2Cover JSON 转成 pathCoverage 灰度地图。")
    parser.add_argument("json_path", help="Fields2Cover GeoJSON 输入路径")
    parser.add_argument("output_path", nargs="?", default="map.png", help="输出 PNG（默认 map.png）")
    parser.add_argument("--pixel-size", type=float, default=0.5, help="每像素对应的米数/平面单位（默认 0.5）")
    parser.add_argument("--padding", type=float, default=0.0, help="地块周围的黑色障碍边距（默认 0）")
    parser.add_argument(
        "--coordinate-system",
        choices=("auto", "planar", "wgs84"),
        default="auto",
        help="输入坐标类型（默认 auto）",
    )
    parser.add_argument(
        "--metadata",
        help="坐标变换元数据路径；默认与 PNG 同名并添加 .json 后缀",
    )
    args = parser.parse_args()

    occupancy_map, metadata = generate_map_from_fields2cover(
        args.json_path,
        args.pixel_size,
        coordinate_system=args.coordinate_system,
        padding=args.padding,
    )
    save_map_png(occupancy_map, args.output_path)
    metadata_path = args.metadata or f"{args.output_path}.json"
    save_metadata(metadata, metadata_path)
    print(
        f"已生成 {args.output_path}: {metadata.width_pixels} x {metadata.height_pixels} px, "
        f"坐标系 {metadata.projected_coordinate_system}"
    )
    print(f"坐标元数据：{metadata_path}")


if __name__ == "__main__":
    main()
