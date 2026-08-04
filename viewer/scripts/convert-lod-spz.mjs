import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { transcodeSpz } from "@sparkjsdev/spark";

const [, , manifestArgument, outputArgument, maxShArgument] = process.argv;
if (!manifestArgument || !outputArgument) {
  console.log("Usage: node scripts/convert-lod-spz.mjs MANIFEST.source.json OUTPUT_DIR [MAX_SH]");
  process.exit(1);
}

const manifestPath = path.resolve(manifestArgument);
const outputDir = path.resolve(outputArgument);
const maxSh = maxShArgument === undefined ? 3 : Number(maxShArgument);
if (!Number.isInteger(maxSh) || maxSh < 0 || maxSh > 3) {
  throw new Error(`MAX_SH must be an integer from 0 to 3, got ${maxShArgument}`);
}
try {
  await fs.access(outputDir);
  throw new Error(`Refusing to overwrite existing output directory: ${outputDir}`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const sourceManifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
const assets = [
  { ...sourceManifest.base, level: "d0" },
  ...sourceManifest.levels.flatMap((level) =>
    level.assets.map((asset) => ({ ...asset, level: level.id })),
  ),
];

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function readHeader(fileBytes) {
  const raw = gunzipSync(fileBytes);
  const magic = raw.readUInt32LE(0);
  const header = {
    magic: `0x${magic.toString(16).padStart(8, "0")}`,
    version: raw.readUInt32LE(4),
    numSplats: raw.readUInt32LE(8),
    shDegree: raw[12],
    fractionalBits: raw[13],
    flags: raw[14],
  };
  if (magic !== 0x5053474e || ![2, 3].includes(header.version)) {
    throw new Error(`Unexpected SPZ header: ${JSON.stringify(header)}`);
  }
  return header;
}

await fs.mkdir(outputDir, { recursive: false });
const convertedByUrl = new Map();
let completed = 0;
for (const asset of assets) {
  const inputBytes = new Uint8Array(await fs.readFile(asset.sourcePath));
  const converted = await transcodeSpz({
    inputs: [
      {
        fileBytes: inputBytes,
        pathOrUrl: asset.sourcePath,
        transform: {
          translate: [0, 0, 0],
          quaternion: [0, 0, 0, 1],
          scale: 1,
        },
      },
    ],
    maxSh,
    fractionalBits: 12,
  });
  if ((converted.clippedCount ?? 0) > 0) {
    throw new Error(`${asset.url} clipped ${converted.clippedCount} splats`);
  }
  const header = readHeader(converted.fileBytes);
  if (header.numSplats !== asset.splats || header.shDegree !== maxSh) {
    throw new Error(
      `${asset.url} header mismatch: ${JSON.stringify(header)} vs ${asset.splats} splats`,
    );
  }
  const outputPath = path.join(outputDir, asset.url);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.tmp`;
  await fs.writeFile(temporaryPath, converted.fileBytes);
  await fs.rename(temporaryPath, outputPath);
  convertedByUrl.set(asset.url, {
    bytes: converted.fileBytes.length,
    sha256: sha256(converted.fileBytes),
    header,
  });
  completed += 1;
  console.log(
    `[${completed}/${assets.length}] ${asset.url}: ${asset.splats.toLocaleString()} splats, ${(converted.fileBytes.length / 1024 / 1024).toFixed(2)} MiB`,
  );
}

function finalizeAsset(asset) {
  const converted = convertedByUrl.get(asset.url);
  if (!converted) throw new Error(`Missing converted asset: ${asset.url}`);
  const { sourcePath: _sourcePath, ...clean } = asset;
  return {
    ...clean,
    url: `${clean.url}?v=${converted.sha256.slice(0, 12)}`,
    ...converted,
  };
}

const manifest = {
  ...sourceManifest,
  source: path.basename(sourceManifest.source),
  maxSh,
  fractionalBits: 12,
  base: finalizeAsset(sourceManifest.base),
  levels: sourceManifest.levels.map((level) => ({
    ...level,
    assets: level.assets.map(finalizeAsset),
  })),
};
manifest.totalBytes = assets.reduce(
  (sum, asset) => sum + convertedByUrl.get(asset.url).bytes,
  0,
);
manifest.initialBytes = manifest.base.bytes;
manifest.assetCount = assets.length;
await fs.writeFile(
  path.join(outputDir, "manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
console.log(
  JSON.stringify(
    {
      output: outputDir,
      assets: manifest.assetCount,
      splats: manifest.totalSplats,
      initialBytes: manifest.initialBytes,
      totalBytes: manifest.totalBytes,
    },
    null,
    2,
  ),
);
