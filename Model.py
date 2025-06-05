import torch
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, mock_output=False):
        super(MockModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mock_output = mock_output
        self.num_blocks = num_blocks
        self.blocks = torch.nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_blocks)
        ])

    def forward(self, seq, candidates):
        for block in self.blocks:
            seq, _ = block(seq, seq, seq)

        logits = torch.matmul(seq[:, -1, :].unsqueeze(1), candidates).transpose(2, 1) # (B, N_c, 1)

        return logits


if __name__ == "__main__":
    n_blocks = 4
    n_heads = 3
    embed_dim = 196
    B =1
    max_seq_len = 128
    N_c = 3000
    candidates = torch.rand(B,embed_dim, N_c) # user - candidates variant
    x = torch.rand(B, max_seq_len, embed_dim)
    inputs = (x, candidates)
    model =  MockModel(embed_dim, n_blocks, n_heads)
    model.eval()
    #print(model(x,candidates).shape)
    torch.onnx.export(
        model,
        inputs,
        "multi_input_model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['x', 'candidates'],
        output_names=['output'],
    )


