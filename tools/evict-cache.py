"""Drop page cache for files under the given dirs (no root needed: posix_fadvise DONTNEED on files we can read).
On GB300 the HBM is NUMA node 1, and page cache that spills there steals GPU memory from vLLM."""
import os, sys
n = 0
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fdatasync(fd) if os.access(p, os.W_OK) else None
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
            except OSError:
                pass
            finally:
                os.close(fd)
print(f"evicted page cache for {n} files")
