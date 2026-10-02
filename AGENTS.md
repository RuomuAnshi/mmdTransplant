# Repository publication rules

- Publish only generic plugin code, synthetic tests and generic documentation.
- Never commit real model names, filenames, local asset paths, material inventories,
  screenshots, model files, textures, motion files, scenes or asset-specific reports.
- Keep real-asset validation scripts in `scripts/local/` and results in `output/`.
  Both directories are private and ignored. Do not force-add ignored files.
- Construct public test fixtures in memory without third-party assets or names.
- Inspect the complete staged file list and run
  `python3 scripts/check_public_tree.py` before committing or pushing.
- Commit messages and public descriptions must not identify private assets.
- Rewriting an already published branch requires explicit user authorization.
