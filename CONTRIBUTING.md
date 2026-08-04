# Contributing

This repository is a small, reproducible 3DGS demonstration. Keep changes focused on
the E57-to-Gaussian pipeline, the additive spatial LoD manifest, and the Three.js viewer.

Before opening a pull request:

```bash
uv sync --locked --dev
uv run pytest -q

cd viewer
npm ci
npm run validate:lod
npm run build
```

Do not commit source captures, private panoramas, E57/PLY files, training checkpoints,
`artifacts/`, `output/`, `node_modules/`, or `viewer/dist/`. The only binary scene assets
intentionally tracked by this demo are the files under `viewer/public/lod/`.
