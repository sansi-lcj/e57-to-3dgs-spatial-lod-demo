"""End-to-end preparation of a Realsee E57 for local 3DGS training."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .colmap_model import ColmapImage, validate_with_pycolmap, write_colmap_model
from .e57io import extract_panoramas, hash_file, read_metadata, verify_page_checksums
from .geometry import (
    colmap_world_to_camera,
    matrix_to_quaternion_wxyz,
    overlapping_virtual_views,
    quaternion_wxyz_to_matrix,
    recover_camera_center,
)
from .panorama import render_panorama_views
from .points import (
    point_cloud_stats,
    registered_point_cloud,
    voxel_downsample,
    write_binary_ply,
)
from .qa import crop_reprojection_overlay, scan_panorama_reprojection


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        result[path.relative_to(root).as_posix()] = hash_file(path)
    return result


def _relative_artifact_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _relative_artifact_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_artifact_paths(item, root) for item in value]
    if isinstance(value, tuple):
        return [_relative_artifact_paths(item, root) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(root):
                return path.relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
    return value


def prepare_dataset(
    source_e57: Path,
    output_dir: Path,
    *,
    output_size: int = 2048,
    fov_deg: float = 90.0,
    num_steps_yaw: int = 4,
    pitches_deg: tuple[float, ...] = (-35.0, 0.0, 35.0),
    voxel_size_m: float = 0.03,
    polar_cap_deg: float = 20.0,
    verify_crc: bool = True,
) -> dict[str, Any]:
    source_e57 = source_e57.resolve()
    output_dir = output_dir.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_dir} or {partial_dir}"
        )
    partial_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    metadata = read_metadata(source_e57, include_sha256=True)
    crc_result = verify_page_checksums(source_e57) if verify_crc else None
    if crc_result is not None and crc_result["failures"]:
        raise ValueError(f"E57 CRC verification failed: {crc_result}")

    panorama_dir = partial_dir / "panoramas"
    panorama_records = extract_panoramas(source_e57, panorama_dir)
    panorama_by_guid = {record["guid"]: record for record in panorama_records}
    for image in metadata.images:
        record = panorama_by_guid[image.guid]
        decoded = cv2.imread(record["path"], cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != (image.height, image.width):
            raise ValueError(f"Decoded panorama dimensions do not match metadata: {image.guid}")

    full_cloud, scan_stats = registered_point_cloud(source_e57)
    init_cloud = voxel_downsample(full_cloud, voxel_size_m)
    pointcloud_dir = partial_dir / "pointcloud"
    full_ply = pointcloud_dir / "registered_full.ply"
    init_ply = pointcloud_dir / f"init_voxel_{int(round(voxel_size_m * 1000)):03d}mm.ply"
    write_binary_ply(full_ply, full_cloud)
    write_binary_ply(init_ply, init_cloud)

    views = overlapping_virtual_views(
        num_steps_yaw=num_steps_yaw,
        pitches_deg=pitches_deg,
    )
    scans_by_guid = {scan.guid: scan for scan in metadata.scans}
    rendered_records = []
    colmap_images: list[ColmapImage] = []
    image_pose_records: list[dict[str, Any]] = []
    image_id = 1
    for image in metadata.images:
        panorama_record = panorama_by_guid[image.guid]
        scan = scans_by_guid[image.associated_data3d_guid]
        rendered = render_panorama_views(
            Path(panorama_record["path"]),
            partial_dir / "images",
            partial_dir / "masks_colmap",
            partial_dir / "masks_train",
            panorama_index=image.index,
            panorama_guid=image.guid,
            scan_guid=scan.guid,
            panorama_name=scan.name,
            output_size=output_size,
            fov_deg=fov_deg,
            polar_cap_deg=polar_cap_deg,
            views=views,
        )
        rotation_ws = quaternion_wxyz_to_matrix(scan.rotation_wxyz)
        center_world = np.asarray(scan.translation_m, dtype=np.float64)
        for rendered_view in rendered:
            view = views[rendered_view.view_index]
            rotation_cw, translation_cw = colmap_world_to_camera(
                rotation_ws, center_world, view.camera_from_pano
            )
            quaternion_cw = matrix_to_quaternion_wxyz(rotation_cw)
            recovered_center = recover_camera_center(rotation_cw, translation_cw)
            if not np.allclose(recovered_center, center_world, atol=1e-10):
                raise ValueError("COLMAP camera center round-trip failed")
            record = ColmapImage(
                image_id=image_id,
                image_name=rendered_view.image_name,
                quaternion_wxyz=tuple(float(value) for value in quaternion_cw),
                translation=tuple(float(value) for value in translation_cw),
                panorama_guid=image.guid,
                scan_guid=scan.guid,
                view_index=view.index,
            )
            colmap_images.append(record)
            image_pose_records.append(
                record.to_dict()
                | {
                    "rotation_cw": rotation_cw.astype(float).tolist(),
                    "camera_center_world_m": center_world.astype(float).tolist(),
                    "yaw_deg": view.yaw_deg,
                    "pitch_deg": view.pitch_deg,
                }
            )
            rendered_records.append(rendered_view.to_dict())
            image_id += 1

    focal = output_size / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    binary_model_dir = partial_dir / "sparse" / "0"
    text_model_dir = partial_dir / "sparse_txt" / "0"
    write_colmap_model(
        binary_model_dir,
        text_model_dir,
        width=output_size,
        height=output_size,
        focal=float(focal),
        images=colmap_images,
        cloud=init_cloud,
    )
    colmap_validation = validate_with_pycolmap(binary_model_dir)
    expected_counts = (1, len(colmap_images), len(init_cloud.xyz))
    actual_counts = (
        colmap_validation["num_cameras"],
        colmap_validation["num_registered_images"],
        colmap_validation["num_points3d"],
    )
    if actual_counts != expected_counts:
        raise ValueError(f"COLMAP validation counts mismatch: {actual_counts} != {expected_counts}")

    metadata_dir = partial_dir / "metadata"
    _write_json(metadata_dir / "source_e57.json", metadata.to_dict())
    _write_json(metadata_dir / "views.json", rendered_records)
    _write_json(metadata_dir / "image_poses.json", image_pose_records)
    _write_json(
        metadata_dir / "rig_manifest.json",
        {
            "coordinate_convention": {
                "scan_local": "+X right, +Y down, +Z forward",
                "world": "right-handed, Z-up, meters",
                "colmap": "world-to-camera; +X right, +Y down, +Z forward",
            },
            "virtual_sensors": [
                {
                    "view_index": view.index,
                    "yaw_deg": view.yaw_deg,
                    "pitch_deg": view.pitch_deg,
                    "camera_from_pano": view.camera_from_pano.astype(float).tolist(),
                }
                for view in views
            ],
            "frames": [
                {
                    "panorama_guid": image.guid,
                    "scan_guid": image.associated_data3d_guid,
                    "scan_rotation_wxyz": list(scans_by_guid[image.associated_data3d_guid].rotation_wxyz),
                    "camera_center_world_m": list(scans_by_guid[image.associated_data3d_guid].translation_m),
                }
                for image in metadata.images
            ],
        },
    )

    qa_dir = partial_dir / "qa"
    panorama_qa = []
    for image in metadata.images:
        panorama_qa.append(
            scan_panorama_reprojection(
                source_e57,
                scans_by_guid[image.associated_data3d_guid].index,
                Path(panorama_by_guid[image.guid]["path"]),
                qa_dir / "panorama_reprojection" / f"scan_{image.index:02d}.jpg",
            )
        )
    crop_qa = []
    selected_views = {4, 5}
    for record, pose in zip(colmap_images, image_pose_records, strict=True):
        if record.view_index not in selected_views:
            continue
        crop_qa.append(
            crop_reprojection_overlay(
                init_cloud,
                partial_dir / "images" / record.image_name,
                qa_dir / "crop_reprojection" / record.image_name,
                np.asarray(pose["rotation_cw"], dtype=np.float64),
                np.asarray(record.translation, dtype=np.float64),
                float(focal),
            )
        )
    qa_summary = _relative_artifact_paths(
        {"panorama_reprojection": panorama_qa, "crop_reprojection": crop_qa},
        partial_dir,
    )
    _write_json(qa_dir / "qa_summary.json", qa_summary)

    finished = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "source": {
            "path": str(source_e57),
            "size_bytes": metadata.file_header.physical_length,
            "sha256": metadata.sha256,
            "crc": crc_result,
        },
        "configuration": {
            "perspective_size": output_size,
            "perspective_fov_deg": fov_deg,
            "perspective_yaw_steps": num_steps_yaw,
            "perspective_pitches_deg": list(pitches_deg),
            "voxel_size_m": voxel_size_m,
            "polar_cap_deg": polar_cap_deg,
        },
        "counts": {
            "scans": len(metadata.scans),
            "panoramas": len(metadata.images),
            "perspective_images": len(colmap_images),
            "full_points": len(full_cloud.xyz),
            "initialization_points": len(init_cloud.xyz),
        },
        "pointcloud": {
            "scan_stats": scan_stats,
            "full": point_cloud_stats(full_cloud),
            "initialization": point_cloud_stats(init_cloud),
            "full_ply": str(full_ply.relative_to(partial_dir)),
            "initialization_ply": str(init_ply.relative_to(partial_dir)),
        },
        "colmap_validation": colmap_validation,
        "qa": qa_summary,
    }
    manifest["files_sha256"] = _tree_hashes(partial_dir)
    _write_json(partial_dir / "manifest.json", manifest)
    partial_dir.replace(output_dir)
    return manifest
