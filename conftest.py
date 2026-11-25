"""Ensures `import src...` resolves when running `uv run pytest` from the
repo root, regardless of pytest's rootdir auto-detection. This file having
no other content is intentional -- its mere presence at the repo root is
what makes pytest add the root to sys.path during collection."""
