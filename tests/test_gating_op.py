import torch
from grouped_gemm.ops import permute, unpermute

# indices = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.int32, device='cuda')
# input_act = torch.tensor([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]], dtype=torch.float32, device='cuda')
# probs = torch.ones_like(indices, dtype=torch.float32)

num_token = 72
num_expert = 8
hidden_size = 12
num_topK = 2

indices = torch.stack([torch.randperm(num_expert, dtype=torch.int32, device="cuda")[:num_topK] for _ in range(num_token)])
# indices = indices.to(torch.int32).cuda()
input_act = torch.rand((num_token, hidden_size), dtype=torch.float32, device="cuda")

probs = torch.rand(num_token, num_topK, dtype=torch.float32).cuda()
row_sums = probs.sum(dim=1, keepdim=True)
probs = probs / row_sums
probs.requires_grad_(True)

permuted_inputs, row_id_map = permute(input_act, indices)
unpermute_outputs = unpermute(permuted_inputs, row_id_map, probs)

flatten_indices = indices.view(-1)
sorted_indices = torch.argsort(flatten_indices, stable=True)

print("input_act : %s" % input_act.cpu().detach().numpy())
# print("permuted_inputs: %s" % permuted_inputs)
# print("row_id_map : %s" % row_id_map)
# print("sorted_indices: %s" % sorted_indices)
# print("unpermute_outputs: %s" % unpermute_outputs)