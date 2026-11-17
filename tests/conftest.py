import os

# macOS: torch and faiss each bundle libomp; loading both aborts the process.
# Test-only workaround so the full suite can import both.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
