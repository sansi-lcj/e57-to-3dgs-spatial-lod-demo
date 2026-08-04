import * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { SpatialLodManager } from "./lod-manager.js";
import "./style.css";

const LOD_MANIFEST_URL =
  import.meta.env.VITE_LOD_MANIFEST_URL ||
  new URL(`${import.meta.env.BASE_URL}lod/manifest.json`, window.location.href).toString();
const SOURCE_BOUNDS = {
  min: new THREE.Vector3(-11.0118418, -0.6854115, -1.2702199),
  max: new THREE.Vector3(2.637929, 9.1443319, 3.1349726),
};

const STATIONS = [
  { position: [-0.0138174408, 0.0084268138, -0.0216614986], direction: [0.1875702176, 0.9822511967, 0] },
  { position: [-2.212699597, -0.1412638532, -0.0252423697], direction: [0.9706490776, 0.2405002458, 0] },
  { position: [-4.739134304, 0.99969107, -0.0386287641], direction: [-0.386088657, 0.9224616788, 0] },
  { position: [-7.766730022, 1.076041064, -0.0408268142], direction: [-0.9950703263, 0.09917179872, 0] },
  { position: [-10.17182389, 1.715469393, -0.0456017047], direction: [-0.9980704356, -0.06209191215, 0] },
  { position: [-2.679071068, 1.776616208, -0.036079848], direction: [0.9941103039, -0.1083729838, 0] },
  { position: [0.1174978548, 2.49053333, -0.0380397843], direction: [0.9881714405, -0.1533532005, 0] },
  { position: [-3.814322989, 2.38052495, -0.0379709521], direction: [-0.9858116486, 0.1678552757, 0] },
  { position: [-5.041539255, 3.37312649, -0.0407496896], direction: [-0.9969765425, -0.07770311214, 0] },
  { position: [-6.892330301, 3.352360706, -0.0415347658], direction: [-0.995841651, -0.09110107602, 0] },
  { position: [-8.894918416, 3.40446912, -0.0446612679], direction: [-0.9987231347, -0.05051831503, 0] },
  { position: [-8.871702664, 5.006782315, -0.0500306228], direction: [-0.9996912349, 0.02484823626, 0] },
  { position: [-6.428888773, 4.879351334, -0.0450697089], direction: [-0.9983338759, -0.0577015792, 0] },
  { position: [-4.62650268, 5.108420926, -0.0431204596], direction: [-0.2713579824, 0.9624784909, 0] },
  { position: [-3.282998616, 5.74057962, -0.0425214847], direction: [-0.3348388502, 0.9422754079, 0] },
  { position: [-3.140747438, 7.141896876, -0.04816744], direction: [0.7037128222, -0.7104845276, 0] },
  { position: [-4.692709833, 7.20386039, -0.0494680928], direction: [0.9998477567, -0.01744887914, 0] },
  { position: [-6.778610036, 7.196328169, -0.0508934971], direction: [0.9999633567, -0.008560679684, 0] },
  { position: [-8.965873574, 7.30305383, -0.0523442044], direction: [0.9949598103, -0.100274503, 0] },
];

const canvas = document.querySelector("#viewport");
const statusText = document.querySelector("#status-text");
const statusDot = document.querySelector("#status-dot");
const splatCount = document.querySelector("#splat-count");
const tileStatus = document.querySelector("#tile-status");
const errorPanel = document.querySelector("#error-panel");
const stationSlider = document.querySelector("#station-slider");
const stationLabel = document.querySelector("#station-label");
const overviewButton = document.querySelector("#overview");
const roamButton = document.querySelector("#roam");
const modeLabel = document.querySelector("#mode-label");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.025, 100);
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));

const spark = new SparkRenderer({
  renderer,
  // Keep projection and footprint faithful for millimetre-scale Gaussian
  // surfels. There is no post-process blur or artificial focal inflation.
  sortRadial: false,
  focalAdjustment: 1,
  blurAmount: 0,
  maxPixelRadius: 96,
});
scene.add(spark);

let roaming = false;
let draggingLook = false;
const lookEuler = new THREE.Euler(0, 0, 0, "YXZ");
const forwardDirection = new THREE.Vector3();
const rightDirection = new THREE.Vector3();
const viewerLookDirection = new THREE.Vector3();
const sourceLookDirection = new THREE.Vector3();

// Source coordinates are right-handed, Z-up and measured in metres. Rotate to
// Three.js Y-up while preserving handedness: (x, y, z) -> (x, z, -y).
const sourceCenter = SOURCE_BOUNDS.min.clone().add(SOURCE_BOUNDS.max).multiplyScalar(0.5);
const viewerCenter = new THREE.Vector3(sourceCenter.x, sourceCenter.z, -sourceCenter.y);

function sourcePointToViewer([x, y, z]) {
  return new THREE.Vector3(x, z, -y).sub(viewerCenter);
}

function sourceDirectionToViewer([x, y, z]) {
  return new THREE.Vector3(x, z, -y).normalize();
}

function viewerPointToSource(position) {
  return new THREE.Vector3(
    position.x + sourceCenter.x,
    sourceCenter.y - position.z,
    position.y + sourceCenter.z,
  );
}

function cameraDirectionToSource() {
  camera.getWorldDirection(viewerLookDirection);
  return sourceLookDirection
    .set(viewerLookDirection.x, -viewerLookDirection.z, viewerLookDirection.y)
    .normalize();
}

let currentStation = 18;

function setStation(index) {
  currentStation = THREE.MathUtils.clamp(index, 0, STATIONS.length - 1);
  const station = STATIONS[currentStation];
  const position = sourcePointToViewer(station.position);
  const direction = sourceDirectionToViewer(station.direction);
  camera.position.copy(position);
  camera.lookAt(position.clone().add(direction));
  camera.updateMatrixWorld();
  stationSlider.value = String(currentStation);
  stationLabel.textContent = `${currentStation + 1} / ${STATIONS.length}`;
  overviewButton.setAttribute("aria-pressed", "false");
}

function setOverview() {
  camera.position.set(11.5, 8.5, 13.5);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  overviewButton.setAttribute("aria-pressed", "true");
}

function setStatus(message, state = "loading") {
  statusText.textContent = message;
  statusDot.dataset.state = state;
}

setStation(currentStation);

const lodManager = new SpatialLodManager({
  scene,
  manifestUrl: LOD_MANIFEST_URL,
  viewerCenter,
  maxConcurrent: 2,
  maxLoadedBytes: 64 * 1024 * 1024,
  maxLoadedSplats: 4_000_000,
  onBaseProgress: (event) => {
    if (event.lengthComputable && event.total > 0) {
      const percentage = Math.round((event.loaded / event.total) * 100);
      setStatus(`正在载入基础层 ${percentage}%`);
    }
  },
  onReady: () => {
    setStatus("基础层已就绪 · 正在细化", "ready");
    document.body.classList.add("ready");
  },
  onState: (state) => {
    splatCount.textContent = state.loadedSplats.toLocaleString("zh-CN");
    splatCount.title = `已加载 ${state.loadedSplats.toLocaleString("zh-CN")} / ${state.totalSplats.toLocaleString("zh-CN")}`;
    tileStatus.textContent = `${state.loadedTiles} 已载 · ${state.loadingTiles + state.queuedTiles} 待载`;
    tileStatus.title = `细节分块 ${state.loadedTiles} / ${state.totalAssets}，当前 ${(state.loadedBytes / 1024 / 1024).toFixed(1)} MiB`;
    if (state.failedTiles > 0) {
      setStatus(`${state.failedTiles} 个细节分块载入失败`, "error");
    } else if (state.loadingTiles + state.queuedTiles > 0) {
      setStatus(`场景已就绪 · 正在细化 ${state.loadedTiles}/${state.totalAssets}`, "ready");
    } else {
      setStatus(`当前区域细化完成 · ${state.loadedTiles} 块`, "ready");
    }
  },
});

lodManager
  .initialize(viewerPointToSource(camera.position), cameraDirectionToSource())
  .catch((error) => {
    console.error(error);
    setStatus("场景载入失败", "error");
    errorPanel.hidden = false;
    errorPanel.textContent = `无法载入 LoD 场景：${error.message}`;
  });

document.querySelector("#station-previous").addEventListener("click", () => {
  setStation(currentStation - 1);
});
document.querySelector("#station-next").addEventListener("click", () => {
  setStation(currentStation + 1);
});
stationSlider.addEventListener("input", () => setStation(Number(stationSlider.value)));
overviewButton.addEventListener("click", setOverview);

function setRoaming(nextRoaming) {
  roaming = nextRoaming;
  document.body.classList.toggle("roaming", roaming);
  roamButton.setAttribute("aria-pressed", String(roaming));
  roamButton.textContent = roaming ? "漫游中 · Esc 退出" : "进入自由漫游";
  modeLabel.textContent = roaming ? "自由漫游中" : "自由漫游";
  if (roaming) renderer.domElement.focus();
  else pressedKeys.clear();
}

roamButton.addEventListener("click", () => setRoaming(!roaming));
renderer.domElement.addEventListener("click", () => setRoaming(true));
renderer.domElement.addEventListener("pointerdown", (event) => {
  if (!roaming || event.button !== 0) return;
  draggingLook = true;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!roaming || !draggingLook) return;
  lookEuler.setFromQuaternion(camera.quaternion);
  lookEuler.y -= event.movementX * 0.00156;
  lookEuler.x -= event.movementY * 0.00156;
  lookEuler.x = THREE.MathUtils.clamp(lookEuler.x, -Math.PI * 0.472, Math.PI * 0.472);
  camera.quaternion.setFromEuler(lookEuler);
});
renderer.domElement.addEventListener("pointerup", (event) => {
  draggingLook = false;
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
});
document.querySelector("#fullscreen").addEventListener("click", async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await document.documentElement.requestFullscreen();
  }
});

const pressedKeys = new Set();
const movementKeys = new Set(["w", "a", "s", "d", "q", "e", "shift"]);
window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (key === "escape" && roaming) {
    setRoaming(false);
    return;
  }
  if (!movementKeys.has(key)) return;
  pressedKeys.add(key);
  if (roaming) event.preventDefault();
});
window.addEventListener("keyup", (event) => pressedKeys.delete(event.key.toLowerCase()));
window.addEventListener("blur", () => pressedKeys.clear());

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const clock = new THREE.Clock();
let lodUpdateElapsed = 0;
renderer.setAnimationLoop(() => {
  const deltaSeconds = Math.min(clock.getDelta(), 0.05);
  if (roaming) {
    const speedMetresPerSecond = pressedKeys.has("shift") ? 4.5 : 1.5;
    const distance = speedMetresPerSecond * deltaSeconds;
    const forward = Number(pressedKeys.has("w")) - Number(pressedKeys.has("s"));
    const right = Number(pressedKeys.has("d")) - Number(pressedKeys.has("a"));
    const vertical = Number(pressedKeys.has("e")) - Number(pressedKeys.has("q"));
    camera.getWorldDirection(forwardDirection);
    forwardDirection.y = 0;
    if (forwardDirection.lengthSq() > 0) forwardDirection.normalize();
    rightDirection.crossVectors(forwardDirection, camera.up).normalize();
    if (forward) camera.position.addScaledVector(forwardDirection, forward * distance);
    if (right) camera.position.addScaledVector(rightDirection, right * distance);
    if (vertical) camera.position.y += vertical * distance;
  }
  lodUpdateElapsed += deltaSeconds;
  if (lodUpdateElapsed >= 0.25) {
    lodManager.update(viewerPointToSource(camera.position), cameraDirectionToSource());
    lodUpdateElapsed = 0;
  }
  renderer.render(scene, camera);
});
