import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultManifest = path.resolve(scriptDirectory, "../public/lod/manifest.json");
const manifestPath = path.resolve(process.argv[2] ?? defaultManifest);
const assetRoot = path.dirname(manifestPath);

function fail(message) {
  throw new Error(`LoD validation failed: ${message}`);
}

function readHeader(bytes) {
  let raw;
  try {
    raw = gunzipSync(bytes);
  } catch (error) {
    fail(`SPZ is not a valid gzip stream (${error.message})`);
  }
  if (raw.length < 16) fail("SPZ header is truncated");
  return {
    magic: `0x${raw.readUInt32LE(0).toString(16).padStart(8, "0")}`,
    version: raw.readUInt32LE(4),
    numSplats: raw.readUInt32LE(8),
    shDegree: raw[12],
    fractionalBits: raw[13],
    flags: raw[14],
  };
}

function localAssetPath(url) {
  const relativeUrl = decodeURIComponent(url.split(/[?#]/, 1)[0]);
  if (path.isAbsolute(relativeUrl)) fail(`absolute asset URL is not allowed: ${url}`);
  const resolved = path.resolve(assetRoot, relativeUrl);
  const relative = path.relative(assetRoot, resolved);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    fail(`asset escapes manifest directory: ${url}`);
  }
  return resolved;
}

const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
if (manifest.version !== 1 || manifest.scheme !== "additive-spatial-lod") {
  fail("unsupported manifest version or scheme");
}
if (!manifest.base || !Array.isArray(manifest.levels)) fail("missing base or levels");

const assets = [
  { ...manifest.base, level: "d0" },
  ...manifest.levels.flatMap((level) =>
    level.assets.map((asset) => ({ ...asset, level: level.id })),
  ),
];
const seenUrls = new Set();
let totalBytes = 0;
let totalSplats = 0;
for (const asset of assets) {
  if (seenUrls.has(asset.url)) fail(`duplicate URL: ${asset.url}`);
  seenUrls.add(asset.url);
  const filePath = localAssetPath(asset.url);
  const bytes = await fs.readFile(filePath);
  const digest = crypto.createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== asset.bytes) fail(`${asset.url}: byte count mismatch`);
  if (digest !== asset.sha256) fail(`${asset.url}: SHA-256 mismatch`);
  const header = readHeader(bytes);
  if (header.magic !== "0x5053474e" || ![2, 3].includes(header.version)) {
    fail(`${asset.url}: invalid SPZ header`);
  }
  if (header.numSplats !== asset.splats) fail(`${asset.url}: splat count mismatch`);
  if (header.shDegree !== manifest.maxSh || header.fractionalBits !== manifest.fractionalBits) {
    fail(`${asset.url}: encoding settings mismatch`);
  }
  totalBytes += bytes.length;
  totalSplats += asset.splats;
}

if (assets.length !== manifest.assetCount) fail("assetCount mismatch");
if (totalBytes !== manifest.totalBytes) fail("totalBytes mismatch");
if (totalSplats !== manifest.totalSplats) fail("totalSplats mismatch");
if (manifest.initialBytes !== manifest.base.bytes) fail("initialBytes must equal base bytes");

console.log(
  JSON.stringify(
    {
      manifest: manifestPath,
      assets: assets.length,
      splats: totalSplats,
      totalBytes,
      initialBytes: manifest.initialBytes,
      sha256: crypto.createHash("sha256").update(await fs.readFile(manifestPath)).digest("hex"),
    },
    null,
    2,
  ),
);
