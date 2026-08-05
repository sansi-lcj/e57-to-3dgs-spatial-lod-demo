import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";

function distanceToBounds2D(point, bounds) {
  const dx = Math.max(bounds.min[0] - point.x, 0, point.x - bounds.max[0]);
  const dy = Math.max(bounds.min[1] - point.y, 0, point.y - bounds.max[1]);
  return Math.hypot(dx, dy);
}

export class SpatialLodManager {
  constructor({
    scene,
    manifestUrl,
    viewerCenter,
    maxConcurrent = 2,
    maxLoadedBytes = 64 * 1024 * 1024,
    maxLoadedSplats = 4_000_000,
    fullLoad = false,
    onBaseProgress = () => {},
    onReady = () => {},
    onState = () => {},
  }) {
    this.scene = scene;
    this.manifestUrl = new URL(manifestUrl, window.location.href);
    this.viewerCenter = viewerCenter.clone();
    this.maxConcurrent = maxConcurrent;
    this.maxLoadedBytes = maxLoadedBytes;
    this.maxLoadedSplats = maxLoadedSplats;
    this.fullLoad = fullLoad;
    this.onBaseProgress = onBaseProgress;
    this.onReady = onReady;
    this.onState = onState;
    this.manifest = null;
    this.baseMesh = null;
    this.records = [];
    this.activeLoads = 0;
    this.sourcePosition = new THREE.Vector3();
    this.sourceDirection = new THREE.Vector3(1, 0, 0);
  }

  async initialize(sourcePosition, sourceDirection) {
    const response = await fetch(this.manifestUrl, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`LoD manifest returned ${response.status}`);
    }
    // Follow redirects and resolve all tile URLs from the final manifest URL.
    this.manifestUrl = new URL(response.url);
    this.manifest = await response.json();
    if (
      this.manifest.version !== 1 ||
      this.manifest.scheme !== "additive-spatial-lod" ||
      !this.manifest.base ||
      !Array.isArray(this.manifest.levels)
    ) {
      throw new Error("Unsupported LoD manifest");
    }
    this.records = this.manifest.levels.flatMap((level, levelIndex) =>
      level.assets.map((asset) => ({
        asset,
        level,
        levelIndex,
        key: `${level.id}/${asset.id}`,
        state: "idle",
        wanted: false,
        distance: Number.POSITIVE_INFINITY,
        mesh: null,
      })),
    );
    this.sourcePosition.copy(sourcePosition);
    this.sourceDirection.copy(sourceDirection).normalize();
    this.baseMesh = this.createMesh(this.manifest.base, this.onBaseProgress);
    await this.baseMesh.initialized;
    this.onReady(this.snapshot());
    this.update(sourcePosition, sourceDirection);
  }

  createMesh(asset, onProgress) {
    const mesh = new SplatMesh({
      url: new URL(asset.url, this.manifestUrl).toString(),
      nonLod: true,
      editable: false,
      raycastable: false,
      onProgress,
    });
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(this.viewerCenter).multiplyScalar(-1);
    this.scene.add(mesh);
    return mesh;
  }

  update(sourcePosition, sourceDirection) {
    if (!this.manifest) return;
    this.sourcePosition.copy(sourcePosition);
    this.sourceDirection.copy(sourceDirection).normalize();

    if (this.fullLoad) {
      for (const record of this.records) {
        record.wanted = true;
        record.priority = record.levelIndex;
      }
      this.pump();
      this.emitState();
      return;
    }

    const candidates = [];
    for (const record of this.records) {
      record.distance = distanceToBounds2D(this.sourcePosition, record.asset.bounds);
      const retaining = record.state === "loaded" || record.state === "loading";
      const proximityThreshold = retaining
        ? record.level.unloadDistanceM
        : record.level.loadDistanceM;
      const centerX = (record.asset.bounds.min[0] + record.asset.bounds.max[0]) * 0.5;
      const centerY = (record.asset.bounds.min[1] + record.asset.bounds.max[1]) * 0.5;
      const toTileX = centerX - this.sourcePosition.x;
      const toTileY = centerY - this.sourcePosition.y;
      const centerDistance = Math.hypot(toTileX, toTileY);
      const viewDistance = retaining
        ? record.level.retainViewDistanceM
        : record.level.viewDistanceM;
      const viewHalfAngle = retaining
        ? record.level.retainViewHalfAngleDegrees
        : record.level.viewHalfAngleDegrees;
      const forwardLength = Math.hypot(this.sourceDirection.x, this.sourceDirection.y);
      const cosine =
        centerDistance > 0 && forwardLength > 0
          ? (toTileX * this.sourceDirection.x + toTileY * this.sourceDirection.y) /
            (centerDistance * forwardLength)
          : 1;
      const visible =
        centerDistance <= viewDistance &&
        cosine >= Math.cos(THREE.MathUtils.degToRad(viewHalfAngle));
      record.candidate = record.distance <= proximityThreshold || visible;
      const angleDegrees = THREE.MathUtils.radToDeg(
        Math.acos(THREE.MathUtils.clamp(cosine, -1, 1)),
      );
      record.priority = visible
        ? angleDegrees * 0.25 + centerDistance * 0.05 + record.levelIndex * 0.01
        : 100 + record.distance + record.levelIndex * 0.01;
      if (record.candidate) candidates.push(record);
    }

    const selected = new Set();
    let selectedBytes = this.manifest.base.bytes;
    let selectedSplats = this.manifest.base.splats;
    candidates.sort((left, right) => left.priority - right.priority);
    for (const record of candidates) {
      if (
        selectedBytes + record.asset.bytes > this.maxLoadedBytes ||
        selectedSplats + record.asset.splats > this.maxLoadedSplats
      ) {
        continue;
      }
      selected.add(record);
      selectedBytes += record.asset.bytes;
      selectedSplats += record.asset.splats;
    }

    for (const record of this.records) {
      record.wanted = selected.has(record);
      if (record.state === "loaded" && !record.wanted) {
        this.disposeRecord(record);
      } else if (record.state === "error" && !record.wanted) {
        record.state = "idle";
      }
    }
    this.pump();
    this.emitState();
  }

  pump() {
    while (this.activeLoads < this.maxConcurrent) {
      const next = this.records
        .filter((record) => record.wanted && record.state === "idle")
        .sort((left, right) => left.priority - right.priority)[0];
      if (!next) break;
      this.loadRecord(next);
    }
  }

  async loadRecord(record) {
    record.state = "loading";
    this.activeLoads += 1;
    record.mesh = this.createMesh(record.asset);
    this.emitState();
    try {
      await record.mesh.initialized;
      if (record.wanted) {
        record.state = "loaded";
      } else {
        this.disposeRecord(record);
      }
    } catch (error) {
      if (record.mesh) {
        this.scene.remove(record.mesh);
        record.mesh.dispose();
        record.mesh = null;
      }
      record.state = "error";
      console.error(`LoD tile failed: ${record.key}`, error);
    } finally {
      this.activeLoads -= 1;
      this.pump();
      this.emitState();
    }
  }

  disposeRecord(record) {
    if (record.mesh) {
      this.scene.remove(record.mesh);
      record.mesh.dispose();
      record.mesh = null;
    }
    record.state = "idle";
  }

  snapshot() {
    const loaded = this.records.filter((record) => record.state === "loaded");
    const loading = this.records.filter((record) => record.state === "loading");
    const queued = this.records.filter(
      (record) => record.wanted && record.state === "idle",
    );
    const failed = this.records.filter((record) => record.state === "error");
    const baseSplats = this.baseMesh?.isInitialized ? this.manifest.base.splats : 0;
    const baseBytes = this.baseMesh?.isInitialized ? this.manifest.base.bytes : 0;
    return {
      totalSplats: this.manifest?.totalSplats ?? 0,
      totalAssets: (this.manifest?.assetCount ?? 1) - 1,
      budgetBytes: this.maxLoadedBytes,
      budgetSplats: this.maxLoadedSplats,
      loadedSplats:
        baseSplats + loaded.reduce((sum, record) => sum + record.asset.splats, 0),
      loadedBytes:
        baseBytes + loaded.reduce((sum, record) => sum + record.asset.bytes, 0),
      loadedTiles: loaded.length,
      loadingTiles: loading.length,
      queuedTiles: queued.length,
      failedTiles: failed.length,
    };
  }

  emitState() {
    if (this.manifest) this.onState(this.snapshot());
  }
}
