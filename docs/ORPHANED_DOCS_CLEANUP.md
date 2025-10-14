# Orphaned Documentation Cleanup

## Issue

Build warnings were appearing for a non-existent module:
```
WARNING: Failed to import starling.inference.bme_example.
Possible hints:
* AttributeError: module 'starling.inference' has no attribute 'bme_example'
* ModuleNotFoundError: No module named 'starling.inference.bme_example'
```

## Root Cause

The file `docs/autosummary/starling.inference.bme_example.rst` existed in the documentation directory but referenced a module that doesn't exist in the source code.

This was likely a leftover from:
- An old example file that was removed from the codebase
- A renamed module that wasn't cleaned up
- Autosummary generation from an earlier version

## Investigation

**Checked:**
1. ✅ No references to `bme_example` in `docs/api.rst`
2. ✅ Not listed in `docs/autosummary/starling.inference.rst`
3. ✅ No `bme_example.py` file exists in `starling/inference/`
4. ✅ No code references to `bme_example` anywhere in the source

**Source files in `starling/inference/`:**
- `__init__.py`
- `benchmark_mds.py`
- `constraints.py`
- `encode_sequences.py`
- `evaluate_vae.py`
- `generation.py`
- `model_loading.py`

## Solution

**Removed the orphaned file:**
```bash
rm docs/autosummary/starling.inference.bme_example.rst
```

## Result

✅ Build warning eliminated
✅ Documentation structure cleaned up
✅ No functional impact (file was never referenced)

## Prevention

To prevent similar issues in the future:

1. **Clean rebuild**: Run `make clean` before `make html` to remove stale generated files
2. **Autosummary overwrite**: The config now has `autosummary_generate_overwrite = True` which helps keep generated files in sync
3. **Version control**: Consider adding generated `.rst` files to `.gitignore` if they're auto-generated

## Verification

To verify the fix:
```bash
cd docs
make clean
make html
```

The warning should no longer appear in the build output.
