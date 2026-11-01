"""Lossless, on-demand temporal tensors for larger training corpora."""
import os
from pathlib import Path
import re
import numpy as np
import torch


class DiskTensor:
    def __init__(self,path,shape):
        self.path=Path(path);self.shape=tuple(shape)

    def to(self,device):
        data=np.load(self.path,mmap_mode='r',allow_pickle=False)
        if data.shape!=self.shape or data.dtype!=np.float32:raise ValueError('Temporal tensor cache contract changed')
        # Copy to the requested device. Augmentation must never mutate the cache.
        return torch.tensor(data,device=device,dtype=torch.float32)


def disk_pair(directory,digest,x,y):
    if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Content-addressed input required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);result=[]
    for role,array in [('x',x),('y',y)]:
        array=np.asarray(array,dtype=np.float32);path=directory/f'{digest}-{role}.npy'
        if not path.exists():
            temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
            try:
                with temporary.open('wb') as stream:np.save(stream,array,allow_pickle=False)
                if not path.exists():os.replace(temporary,path)
            finally:
                if temporary.exists():temporary.unlink()
        result.append(DiskTensor(path,array.shape))
    return tuple(result)
