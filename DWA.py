import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abstract_weighting import AbsWeighting


class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """

    def __init__(self, args):
        super(DWA, self).__init__()
        self.T = 2

        self.device = args['device']
        self.task_num = args['num_task']

    def backward(self, losses, **kwargs):

        self.epoch = kwargs['epoch']
        self.train_loss_buffer = kwargs['train_loss_buffer']


        if self.epoch > 1:
            w_i = t.Tensor(
                self.train_loss_buffer[:, self.epoch - 1] / self.train_loss_buffer[:, self.epoch - 2]).to(self.device)
            batch_weight = self.task_num * F.softmax(w_i / self.T, dim=-1)
        else:
            batch_weight = t.ones(len(losses))


        return batch_weight.detach().tolist()
