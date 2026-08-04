import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { transcodeSpz } from "@sparkjsdev/spark";

const [, , inputArgument, outputArgument, maxShArgument] = process.argv;
if (!inputArgument || inputArgument === "--help" || inputArgument === "-h") {
  console.log("Usage: npm run convert -- INPUT.ply [OUTPUT.spz] [MAX_SH]");
  process.exit(inputArgument ? 0 : 1);
}

const inputPath = path.resolve(inputArgument);
const outputPath = path.resolve(
  outputArgument ?? inputPath.replace(/\.[^.]+$/, ".spz"),
);
const maxSh = maxShArgument === undefined ? 1 : Number(maxShArgument);
if (!Number.isInteger(maxSh) || maxSh < 0 || maxSh > 3) {
  throw new Error(`MAX_SH must be an integer from 0 to 3, got ${maxShArgument}`);
}
const inputBytes = new Uint8Array(await fs.readFile(inputPath));

const converted = await transcodeSpz({
  inputs: [
    {
      fileBytes: inputBytes,
      pathOrUrl: inputPath,
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

if (converted.clippedCount > 0) {
  throw new Error(
    `SPZ fixed-point range clipped ${converted.clippedCount} splats; output was not written`,
  );
}

const raw = gunzipSync(converted.fileBytes);
const magic = raw.readUInt32LE(0);
const header = {
  magic: `0x${magic.toString(16).padStart(8, "0")}`,
  version: raw.readUInt32LE(4),
  numSplats: raw.readUInt32LE(8),
  shDegree: raw[12],
  fractionalBits: raw[13],
  flags: raw[14],
};
// Spark 2.1.0's published encoder emits v3; current source builds may emit v2.
// Both layouts are supported by the same viewer.
if (magic !== 0x5053474e || ![2, 3].includes(header.version)) {
  throw new Error(`Unexpected SPZ header: ${JSON.stringify(header)}`);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, converted.fileBytes);
console.log(
  JSON.stringify(
    {
      input: inputPath,
      output: outputPath,
      outputBytes: converted.fileBytes.length,
      clippedSplats: converted.clippedCount ?? 0,
      header,
    },
    null,
    2,
  ),
);
