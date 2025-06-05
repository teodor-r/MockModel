import torch
import torch.nn as nn
import onnxruntime as ort
import json

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
    embed_dim = 192
    B = 1
    max_seq_len = 128
    N_c = 3000
    candidates = torch.rand(B,embed_dim, N_c) # user <-> candidates
    x = torch.rand(B, max_seq_len, embed_dim)
    inputs = (x, candidates)
    model = MockModel(embed_dim,n_heads, n_blocks)
    model.eval()
    #print(model(x,candidates).shape)
    torch.onnx.export(
        model,
        inputs,
        "model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['x', 'candidates'],
        output_names=['logits'],
    )

    onnx_session = ort.InferenceSession("model.onnx")
    input_onnx_info  = [{
        "name": i.name,
        "type": i.type,
        "shape": i.shape,
    }for i in onnx_session.get_inputs()]
    output_onnx_info  = [{
        "name": i.name,
        "type": i.type,
        "shape": i.shape,
    }for i in onnx_session.get_outputs()]

    io_info  = {"inputs": input_onnx_info, "outputs": output_onnx_info}

    with open("onnx_io_info.json", "w") as f:
        json.dump(io_info, f)

    onnx_inputs = {
        "x": x.cpu().numpy(),
        "candidates": candidates.cpu().detach().numpy()
    }


    onnx_logits_list  = onnx_session.run(None, onnx_inputs)[0]
    onnx_logits = torch.from_numpy(onnx_logits_list).to("cpu")
    print(onnx_logits.shape)

    with open("input.json", "w") as f:
        for obj_ in onnx_inputs:
            onnx_inputs[obj_] = onnx_inputs[obj_].tolist()
        json.dump(onnx_inputs, f)

    with open("output.json", "w") as f:
        json.dump({"logits": onnx_logits_list[0].tolist()}, f)


