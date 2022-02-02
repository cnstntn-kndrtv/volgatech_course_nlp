# -*- coding: utf-8 -*-
"""Some utilities."""

import os
from numpy import random as np_random
import random
import torch


def reset_random_seeds(seed: int = 42):
    """
    Set random seeds.

    Args:
        seed (int, optional): seed. Defaults to 42.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np_random.seed(seed)
    random.seed(seed)
    # tensorflow.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
