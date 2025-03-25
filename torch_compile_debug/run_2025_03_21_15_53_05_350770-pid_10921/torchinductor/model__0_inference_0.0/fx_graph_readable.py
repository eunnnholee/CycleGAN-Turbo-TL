class <lambda>(torch.nn.Module):
    def forward(self):
         # File: /home/elicer/cyclegan-turbo/src/cyclegan_turbo.py:249 in forward, code: dummy_text_emb = torch.zeros((batch_size, token_length, hidden_dim), device=x_t.device)
        full_default: "f32[4, 77, 1024]" = torch.ops.aten.full.default([4, 77, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (full_default,)
        