class <lambda>(torch.nn.Module):
    def forward(self):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:554 in torch_dynamo_resume_in_previous_timestep_at_553, code: prev_t = torch.tensor(-1)
        _tensor_constant0: "i64[]" = self._tensor_constant0;  _tensor_constant0 = None
        full_default: "i64[]" = torch.ops.aten.full.default([], -1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        return (full_default,)
        