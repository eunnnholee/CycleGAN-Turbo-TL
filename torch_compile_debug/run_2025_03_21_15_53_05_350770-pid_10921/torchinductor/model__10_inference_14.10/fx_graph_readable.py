class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:331 in torch_dynamo_resume_in__get_variance_at_330, code: current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        div: "f32[]" = torch.ops.aten.div.Tensor(arg1_1, arg0_1)
        sub: "f32[]" = torch.ops.aten.sub.Tensor(1, div);  div = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:336 in torch_dynamo_resume_in__get_variance_at_330, code: variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(1, arg0_1);  arg0_1 = None
        sub_2: "f32[]" = torch.ops.aten.sub.Tensor(1, arg1_1);  arg1_1 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sub_1, sub_2);  sub_1 = sub_2 = None
        mul: "f32[]" = torch.ops.aten.mul.Tensor(div_1, sub);  div_1 = sub = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:339 in torch_dynamo_resume_in__get_variance_at_330, code: variance = torch.clamp(variance, min=1e-20)
        clamp_min: "f32[]" = torch.ops.aten.clamp_min.default(mul, 1e-20);  mul = None
        return (clamp_min,)
        