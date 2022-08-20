import torch
from torch.utils.data import RandomSampler
import numpy as np

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):

        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


class SubsetSequentialRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
        self.sequence = list(range(len(self.indices)))

    def __iter__(self):
        np.random.seed(100)
        np.random.shuffle(self.sequence)
        return (self.indices[i] for i in self.sequence)

    def __len__(self):
        return len(self.indices)


class BalancedSubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, indices, weights, generator=None, replacement = True):
        self.indices = indices
        self.generator = generator
        self.num_samples = len(self.indices)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        self.weights = weights_tensor
        self.replacement = replacement



    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())
        # for i in torch.randperm(len(self.indices), generator=self.generator):
        #     yield self.indices[i]

    def __len__(self):
        return len(self.indices)
