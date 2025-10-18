import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log: bool = True):
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        a_log_prob = None
        if with_log:
            s, a, a_log_prob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float, device=self.device)
        a = torch.tensor(np.asarray(a), dtype=torch.float, device=self.device)
        if with_log:
            a_log_prob = torch.tensor(np.asarray(a_log_prob), dtype=torch.float, device=self.device)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float, device=self.device)
        r = torch.tensor(np.asarray(r), dtype=torch.float, device=self.device).view(batch_size, 1)
        dw = torch.tensor(np.asarray(dw), dtype=torch.float, device=self.device).view(batch_size, 1)
        done = torch.tensor(np.asarray(done), dtype=torch.float, device=self.device).view(batch_size, 1)
        if with_log:
            return s, a, a_log_prob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class ReplayBufferDiscreteAction(ReplayBuffer):
    def __init__(self, capacity: int):
        super(ReplayBufferDiscreteAction, self).__init__(capacity,
                                                        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    def sample(self, batch_size: int = None, sequential: bool = True, with_log: bool = True):
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        a_log_prob = None
        if with_log:
            s, a, a_log_prob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.int64)
        if with_log:
            a_log_prob = torch.tensor(np.asarray(a_log_prob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r).reshape((batch_size, 1)), dtype=torch.float)
        dw = torch.tensor(np.asarray(dw).reshape((batch_size, 1)), dtype=torch.float)
        done = torch.tensor(np.asarray(done).reshape((batch_size, 1)), dtype=torch.float)
        if with_log:
            return s, a, a_log_prob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done
