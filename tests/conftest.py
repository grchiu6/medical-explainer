import os

# macOS: torch and faiss each bundle libomp; loading both aborts the process
# (KMP_DUPLICATE_LIB_OK) and faiss index.search segfaults with multiple OpenMP
# threads when both are loaded (OMP_NUM_THREADS=1). Test-only workarounds; they
# must be set before faiss/torch are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
