# Public asset policy

The repository contains a browser-ready LoD package under `viewer/public/lod/` so a fresh
GitHub checkout can run the demo without a separate data server. It is 64 SPZ files plus a
manifest, about 95.6 MiB in total; every file is below GitHub's individual large-file limit.

The following remain local-only and are ignored by Git:

- original E57 captures and embedded panoramas;
- source and trained PLY files;
- COLMAP/OpenSplat datasets, checkpoints, logs, and temporary conversion output;
- screenshots and browser profiles outside the curated demo image.

The published SPZ files are derived from a real scan and may reveal the captured premises.
Before making this repository public, confirm that the owner of the scan and all visible
content has approved publication. Removing an E57 from the repository does not make a
derived Gaussian scene anonymous.

The code license has deliberately not been selected in this working tree. Choose and add a
license before accepting external contributions. Third-party packages retain their own
licenses; the pipeline report calls out OpenSplat's AGPLv3 boundary and the viewer's MIT
dependencies.
