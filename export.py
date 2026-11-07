"""Turn a solver dump (average strategy floats) into a compact uint8 table."""
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
a = np.fromfile(src, dtype=np.float32)
n = a.size // (128 * 4)
a = a.reshape(n, 128, 4)
t = a.sum(axis=2, keepdims=True)
p = np.where(t > 0, a / np.maximum(t, 1e-12), 0)
out = np.round(p * 255).astype(np.uint8)
np.save(dst, out)
print(dst, out.shape, "infosets touched", int((t[..., 0] > 0).sum()), "of", n * 128)
