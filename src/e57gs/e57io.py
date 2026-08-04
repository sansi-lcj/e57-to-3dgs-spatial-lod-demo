"""Read Realsee E57 metadata and embedded panoramas without altering the source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
import pye57


E57_HEADER = struct.Struct("<8sIIQQQQ")


@dataclass(frozen=True)
class E57FileHeader:
    signature: str
    major: int
    minor: int
    physical_length: int
    xml_physical_offset: int
    xml_logical_length: int
    page_size: int


@dataclass(frozen=True)
class ScanInfo:
    index: int
    guid: str
    name: str
    point_count: int
    point_fields: tuple[str, ...]
    rotation_wxyz: tuple[float, float, float, float]
    translation_m: tuple[float, float, float]
    local_bounds_m: dict[str, float]


@dataclass(frozen=True)
class ImageInfo:
    index: int
    guid: str
    name: str
    associated_data3d_guid: str
    width: int
    height: int
    pixel_width: float
    pixel_height: float
    jpeg_byte_count: int
    has_independent_pose: bool


@dataclass(frozen=True)
class E57Metadata:
    path: str
    sha256: str
    file_header: E57FileHeader
    root_guid: str
    library_version: str
    has_coordinate_metadata: bool
    scans: tuple[ScanInfo, ...]
    images: tuple[ImageInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_file_header(path: Path) -> E57FileHeader:
    with path.open("rb") as source:
        raw_header = source.read(E57_HEADER.size)
    if len(raw_header) != E57_HEADER.size:
        raise ValueError(f"E57 header is truncated: {path}")
    signature, major, minor, length, xml_offset, xml_length, page_size = E57_HEADER.unpack(raw_header)
    if signature != b"ASTM-E57":
        raise ValueError(f"Unexpected E57 signature: {signature!r}")
    actual_length = path.stat().st_size
    if actual_length != length:
        raise ValueError(f"E57 length mismatch: header={length}, actual={actual_length}")
    if page_size <= 4 or page_size & (page_size - 1):
        raise ValueError(f"Unexpected E57 page size: {page_size}")
    return E57FileHeader(
        signature=signature.decode("ascii"),
        major=major,
        minor=minor,
        physical_length=length,
        xml_physical_offset=xml_offset,
        xml_logical_length=xml_length,
        page_size=page_size,
    )


def _string(node: Any, field: str) -> str:
    return str(node[field].value())


def _integer(node: Any, field: str) -> int:
    return int(node[field].value())


def _float(node: Any, field: str) -> float:
    return float(node[field].value())


def read_metadata(path: Path, include_sha256: bool = True) -> E57Metadata:
    path = path.resolve()
    file_header = read_file_header(path)
    with pye57.E57(str(path)) as e57:
        root = e57.root
        scans: list[ScanInfo] = []
        for index in range(e57.scan_count):
            node = e57.data3d[index]
            header = e57.get_header(index)
            bounds = {
                "x_min": float(header.xMinimum),
                "x_max": float(header.xMaximum),
                "y_min": float(header.yMinimum),
                "y_max": float(header.yMaximum),
                "z_min": float(header.zMinimum),
                "z_max": float(header.zMaximum),
            }
            scans.append(
                ScanInfo(
                    index=index,
                    guid=_string(node, "guid"),
                    name=_string(node, "name"),
                    point_count=int(header.point_count),
                    point_fields=tuple(header.point_fields),
                    rotation_wxyz=tuple(float(value) for value in header.rotation),
                    translation_m=tuple(float(value) for value in header.translation),
                    local_bounds_m=bounds,
                )
            )

        images: list[ImageInfo] = []
        images_node = root["images2D"]
        for index in range(len(images_node)):
            node = images_node[index]
            representation = node["sphericalRepresentation"]
            blob = representation["jpegImage"]
            images.append(
                ImageInfo(
                    index=index,
                    guid=_string(node, "guid"),
                    name=_string(node, "name"),
                    associated_data3d_guid=_string(node, "associatedData3DGuid"),
                    width=_integer(representation, "imageWidth"),
                    height=_integer(representation, "imageHeight"),
                    pixel_width=_float(representation, "pixelWidth"),
                    pixel_height=_float(representation, "pixelHeight"),
                    jpeg_byte_count=int(blob.byteCount()),
                    has_independent_pose=bool(node.isDefined("pose")),
                )
            )

        metadata = E57Metadata(
            path=str(path),
            sha256=hash_file(path) if include_sha256 else "",
            file_header=file_header,
            root_guid=_string(root, "guid"),
            library_version=_string(root, "e57LibraryVersion"),
            has_coordinate_metadata=bool(root.isDefined("coordinateMetadata")),
            scans=tuple(scans),
            images=tuple(images),
        )
    validate_associations(metadata)
    return metadata


def validate_associations(metadata: E57Metadata) -> None:
    scan_guids = [scan.guid for scan in metadata.scans]
    image_guids = [image.guid for image in metadata.images]
    if len(set(scan_guids)) != len(scan_guids):
        raise ValueError("Data3D GUIDs are not unique")
    if len(set(image_guids)) != len(image_guids):
        raise ValueError("Image2D GUIDs are not unique")
    scans_by_guid = {scan.guid: scan for scan in metadata.scans}
    for image in metadata.images:
        scan = scans_by_guid.get(image.associated_data3d_guid)
        if scan is None:
            raise ValueError(f"Image {image.guid} references a missing Data3D GUID")
        if image.name != scan.name:
            raise ValueError(f"Image/scan name mismatch for GUID {scan.guid}")
        if image.width != image.height * 2:
            raise ValueError(f"Image {image.guid} is not a 2:1 panorama")
        if not np.isclose(image.pixel_width, 2 * np.pi / image.width, atol=1e-15):
            raise ValueError(f"Unexpected pixelWidth for image {image.guid}")
        if not np.isclose(image.pixel_height, np.pi / image.height, atol=1e-15):
            raise ValueError(f"Unexpected pixelHeight for image {image.guid}")


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "unnamed"


def extract_panoramas(path: Path, output_dir: Path, overwrite: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with pye57.E57(str(path.resolve())) as e57:
        images_node = e57.root["images2D"]
        for index in range(len(images_node)):
            node = images_node[index]
            name = _string(node, "name")
            guid = _string(node, "guid")
            associated_guid = _string(node, "associatedData3DGuid")
            blob = node["sphericalRepresentation"]["jpegImage"]
            output_path = output_dir / f"{index:02d}_{safe_name(name)}.jpg"
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            jpeg_bytes = np.asarray(blob.read_buffer(), dtype=np.uint8).tobytes()
            temporary_path = output_path.with_suffix(".jpg.partial")
            temporary_path.write_bytes(jpeg_bytes)
            temporary_path.replace(output_path)
            extracted.append(
                {
                    "index": index,
                    "guid": guid,
                    "name": name,
                    "associated_data3d_guid": associated_guid,
                    "path": str(output_path),
                    "bytes": len(jpeg_bytes),
                    "sha256": sha256(jpeg_bytes).hexdigest(),
                }
            )
    return extracted


def _crc32c_table() -> tuple[int, ...]:
    polynomial = 0x82F63B78
    values: list[int] = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (polynomial if crc & 1 else 0)
        values.append(crc)
    return tuple(values)


CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def verify_page_checksums(path: Path, page_size: int | None = None) -> dict[str, int]:
    if page_size is None:
        page_size = read_file_header(path).page_size
    checked = 0
    failures = 0
    with path.open("rb") as source:
        while page := source.read(page_size):
            checked += 1
            if len(page) != page_size:
                failures += 1
                break
            expected = struct.unpack(">I", page[-4:])[0]
            if crc32c(page[:-4]) != expected:
                failures += 1
    return {"pages_checked": checked, "failures": failures, "page_size": page_size}
