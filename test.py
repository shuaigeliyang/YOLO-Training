import torch



print(torch.cuda.memory_summary())
print(torch.cuda.memory_allocated())  # 当前分配的显存
print(torch.cuda.max_memory_allocated())  # 最大分配过的显存
print(torch.cuda.memory_reserved())  # 缓存分配器管理的总内存