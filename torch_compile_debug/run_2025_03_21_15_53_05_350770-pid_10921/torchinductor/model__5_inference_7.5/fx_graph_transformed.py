class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "i64[]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:440 in torch_dynamo_resume_in_step_at_439, code: alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        ge: "b8[]" = torch.ops.aten.ge.Scalar(arg1_1, 0);  arg1_1 = None
        return (arg0_1, ge)
        