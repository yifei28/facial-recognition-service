import redis
r = redis.Redis(host='127.0.0.1', port=6379, db=0)

import numpy as np
import os

emb = np.random.rand(512).astype(np.float32)

# 原始二进制存取
# r.set('guard:123:emb', emb.tobytes())
emb_dir = "embeddings/"
for fname in os.listdir(emb_dir):
    if fname.endswith(".npy"):
        guard_id = fname[:-4]
        path = os.path.join(emb_dir, fname)
        emb = np.load(path)

r.set('guard:test:emb', emb.tobytes())

for key in r.scan_iter(match='*', count=100):
    print(key, r.type(key))