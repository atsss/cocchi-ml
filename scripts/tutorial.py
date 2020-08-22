import torch
import numpy as np

# x = torch.empty(5, 3)
# print(x)
#
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
#
# x = torch.tensor([5.5, 3])
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x)
#
# print(x.size())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
