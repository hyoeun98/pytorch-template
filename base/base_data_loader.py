import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn, # batch size로 묶는 fn
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs) # 샘플링된 train set을 dataloader로 리턴

    def _split_sampler(self, split): # train, valid set sampling
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples) # len(dataset)

        np.random.seed(0)
        np.random.shuffle(idx_full) # idx_full을 랜덤하게 shuffle

        if isinstance(split, int): #split이 int라면
            assert split > 0 # split이 음수일 때 에러
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset." # split이 len(dataset)보다 크면 에러
            len_valid = split 
        else: # split이 실수일 때
            len_valid = int(self.n_samples * split) # dataset에서 split의 비율 (int형변환)

        valid_idx = idx_full[0:len_valid] # validation set index
        train_idx = np.delete(idx_full, np.arange(0, len_valid)) # train set = dataset - validation set

        train_sampler = SubsetRandomSampler(train_idx) # 완전 랜덤한 샘플링
        valid_sampler = SubsetRandomSampler(valid_idx) # 완전 랜덤한 샘플링

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False # shuffle은 샘플러와 같이 사용할 수 없다
        self.n_samples = len(train_idx) # train set의 개수

        return train_sampler, valid_sampler # sampling된 valid, train set

    def split_validation(self): #샘플링된 valid set을 dataloader로 return
        if self.valid_sampler is None: # valid set이 없을 시 None
            return None
        else: 
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
