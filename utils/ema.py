import torch as t
from copy import deepcopy
from contextlib import contextmanager

class ExponentialMovingAverage:
    """
    实现模型参数的指数移动平均
    """
    def __init__(self, parameters, decay=0.999):
        """
        Args:
            parameters: 模型参数
            decay: EMA衰减率
        """
        self.decay = decay
        self.shadow_params = {}
        self.collected_params = {}
        
        # 初始化影子参数
        self.params = {name: param for name, param in parameters.named_parameters() if param.requires_grad}
        for name, param in self.params.items():
            self.shadow_params[name] = param.data.clone()
                
    def update(self):
        """
        更新影子参数
        """
        for name, param in self.params.items():
            self.shadow_params[name] = (
                self.decay * self.shadow_params[name] + 
                (1 - self.decay) * param.data
            )
                
    @contextmanager
    def average_parameters(self):
        """
        临时将模型参数替换为影子参数的上下文管理器
        """
        # 保存当前参数
        for name, param in self.params.items():
            self.collected_params[name] = param.data.clone()
            param.data = self.shadow_params[name]
                
        yield
        
        # 恢复原始参数
        for name, param in self.params.items():
            param.data = self.collected_params[name] 