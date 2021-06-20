import random
import torch
from torch.autograd import Variable

class BufferPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_buffer_batches = 0
            self.buffers = []

    def query(self, buffer_batch):
        if self.pool_size == 0:
            return buffer_batch
        return_buffers = []
        for buffer in buffer_batch.data:
            buffer = torch.unsqueeze(buffer, 0)
            if self.num_buffer_batches < self.pool_size:
                self.num_buffer_batches = self.num_buffer_batches + 1
                self.buffers.append(buffer)
                return_buffers.append(buffer)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.buffers[random_id].clone()
                    self.buffers[random_id] = buffer
                    return_buffers.append(tmp)
                else:
                    return_buffers.append(buffer)
        return_buffers = Variable(torch.cat(return_buffers, 0))
        return return_buffers
