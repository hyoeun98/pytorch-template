import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # sub class에서 구현해야 함
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) # requires_grad = True인 param의 list
        params = sum([np.prod(p.size()) for p in model_parameters]) # 파라미터의 총 개수
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
