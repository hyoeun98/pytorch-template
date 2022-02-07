import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname): # dir가 없다면 mkdir 
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle: # read text mode
        return json.load(handle, object_hook=OrderedDict) # json파일을 객체로 return

def write_json(content, fname): 
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False) # content를 json 파일로 저장

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader): # repeat = data loader를 iterator로 변환
        yield from loader # loader를 하나씩 yield

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count() # 가용 cuda gpu 수
    if n_gpu_use > 0 and n_gpu == 0: # 가용 gpu가 0일 때
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu: # 가용 gpu가 적을 때
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu') # 가용 gpu가 없다면 cpu 사용
    list_ids = list(range(n_gpu_use))  # 가용 gpu list
    return device, list_ids # gpu와, gpu list

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer # tensorboard의 writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self): # 모든 데이터 초기화
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value) # tensorboard update
        self._data.total[key] += value * n # total update 
        self._data.counts[key] += n # counts update
        self._data.average[key] = self._data.total[key] / self._data.counts[key] # avg update

    def avg(self, key): # key column의 avg
        return self._data.average[key]

    def result(self): # dataframe avg리턴
        return dict(self._data.average)
