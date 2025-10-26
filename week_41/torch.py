import norch

tensor1 = norch.Tensor([[1, 2, 3], [3, 2, 1]])
tensor2 = norch.Tensor([[3, 2, 1], [1, 2, 3]])

result = tensor1 + tensor2
print(result[0, 0])