import numpy as np, sys
from PIL import Image
import os

names = sys.argv[1:]
prev = None
for n in names:
    im = np.asarray(Image.open(n).convert('L')).astype(np.int16)
    print(n, os.path.getsize(n), 'shape', im.shape, 'mean %.2f' % im.mean())
    if prev is not None:
        d = np.abs(im - prev).astype(np.float32)
        print('   vs prev: mean abs diff %.2f  pct>50 %.2f%%  max %d' % (d.mean(), (d > 50).mean() * 100, d.max()))
    prev = im