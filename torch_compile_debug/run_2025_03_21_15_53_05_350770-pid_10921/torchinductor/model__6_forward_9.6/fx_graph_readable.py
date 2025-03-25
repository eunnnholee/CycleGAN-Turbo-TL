class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[]", primals_2: "f32[]", primals_3: "f32[4, 32, 32]", primals_4: "f32[4, 32, 32]", primals_5: "i64[]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:441 in torch_dynamo_resume_in_step_at_440, code: beta_prod_t = 1 - alpha_prod_t
        sub: "f32[]" = torch.ops.aten.sub.Tensor(1, primals_2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:442 in torch_dynamo_resume_in_step_at_440, code: beta_prod_t_prev = 1 - alpha_prod_t_prev
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(1, primals_1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:443 in torch_dynamo_resume_in_step_at_440, code: current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        div: "f32[]" = torch.ops.aten.div.Tensor(primals_2, primals_1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:444 in torch_dynamo_resume_in_step_at_440, code: current_beta_t = 1 - current_alpha_t
        sub_2: "f32[]" = torch.ops.aten.sub.Tensor(1, div)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:449 in torch_dynamo_resume_in_step_at_440, code: pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pow_1: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 0.5)
        mul: "f32[4, 32, 32]" = torch.ops.aten.mul.Tensor(pow_1, primals_3);  pow_1 = primals_3 = None
        sub_3: "f32[4, 32, 32]" = torch.ops.aten.sub.Tensor(primals_4, mul);  mul = None
        pow_2: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(primals_2, 0.5)
        div_1: "f32[4, 32, 32]" = torch.ops.aten.div.Tensor(sub_3, pow_2);  sub_3 = pow_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:470 in torch_dynamo_resume_in_step_at_440, code: pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        pow_3: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 0.5)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(pow_3, sub_2);  pow_3 = sub_2 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(mul_1, sub);  mul_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:471 in torch_dynamo_resume_in_step_at_440, code: current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pow_4: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(div, 0.5);  div = None
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(pow_4, sub_1);  pow_4 = sub_1 = None
        div_3: "f32[]" = torch.ops.aten.div.Tensor(mul_2, sub);  mul_2 = sub = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:475 in torch_dynamo_resume_in_step_at_440, code: pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        mul_3: "f32[4, 32, 32]" = torch.ops.aten.mul.Tensor(div_2, div_1);  div_2 = None
        mul_4: "f32[4, 32, 32]" = torch.ops.aten.mul.Tensor(div_3, primals_4);  div_3 = primals_4 = None
        add: "f32[4, 32, 32]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:479 in torch_dynamo_resume_in_step_at_440, code: if t > 0:
        gt: "b8[]" = torch.ops.aten.gt.Scalar(primals_5, 0);  primals_5 = None
        return (gt, div_1, add, primals_1, primals_2)
        