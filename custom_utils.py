from tensorpack import HyperParamSetter
import operator
import numpy as np

class CyclicLearningRateSetter(HyperParamSetter):
    """
    Cyclic Learning Rate: https://arxiv.org/pdf/1506.01186.pdf
    mode: triangular2 or exp_range. detail in paper.
    """

    def __init__(self, param, base_lr=0.001, max_lr=0.006, step_size=2000., mode="triangular2", step_based=True):
        self._step = step_based
        self.mode = mode
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = float(step_size)
        self.scale_fn = lambda x: 1 / (2. ** (x - 1)) if mode == 'triangular2' else lambda x: gamma**(x)
        super(CyclicLearningRateSetter, self).__init__(param)

    def clr(self):
        cycle_num = np.floor(1 + float(self.global_step) / (2 * self.step_size))
        step_ratio = np.abs(float(self.global_step) / self.step_size - 2 * cycle_num + 1)
        if self.mode == 'triangular2':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-step_ratio)) * self.scale_fn(cycle_num)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-step_ratio)) * self.scale_fn(self.global_step)

    def _get_value_to_set(self):
        v = self.clr()
        return v

    def _trigger_epoch(self):
        if not self._step:
            self.trigger()

    def _trigger_step(self):
        if self._step:
            self.trigger()
