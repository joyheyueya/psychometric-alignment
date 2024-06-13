from utils import num_tokens_from_messages

class Memory:
    def __init__(self,max_tokens):
        self.max_tokens = max_tokens
        self.mem = []
        self.start_indices = []
        self.all_mem = []
        self.all_mem_start_indices = []

    def append(self, data, to_last_trajectory=False):
        # If appending to the last trajectory
        if to_last_trajectory:
            if not self.mem or not self.start_indices:
                raise ValueError("No trajectory exists in memory to append to!")
            self.mem.append(data)
            self.all_mem.append(data)
        # If appending a new trajectory
        else:
            self.start_indices.append(len(self.mem))
            self.all_mem_start_indices.append(len(self.all_mem))
            self.mem.extend(data)
            self.all_mem.extend(data)

        # Delete the first trajectory until the total number of tokens in memory is less than self.max_tokens
        if self.max_tokens > -1:  
#             print('self.max_tokens', self.max_tokens)
            while num_tokens_from_messages(self.mem) > self.max_tokens:
                start_index = self.start_indices.pop(0)
                end_index = self.start_indices[0] if self.start_indices else len(self.mem)
                del self.mem[start_index:end_index]

                # Adjust the start indices of the remaining trajectories
                for i in range(len(self.start_indices)):
                    self.start_indices[i] -= (end_index - start_index)

    def __repr__(self):
        return f"Memory(mem={self.mem}, start_indices={self.start_indices})"