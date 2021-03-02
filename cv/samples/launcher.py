import os
import time
from math import ceil

batch_size = 200
interval = 20  # ceil(60000/batch_size)

cmd = 'python -m torch.distributed.launch --use_env pytorch_conv3L_mnist.py ' \
      '--n_iters=500 ' \
      '--burnin=50 ' \
      '--batch_size=' + str(batch_size) + ' ' \
      '--batch_size_test=' + str(batch_size) + ' ' \
      '--n_test_interval=' + str(interval)

_s = time.time()
os.system(cmd)
_e = time.time()
print('finished in {} minutes'.format((_e - _s)/60))
