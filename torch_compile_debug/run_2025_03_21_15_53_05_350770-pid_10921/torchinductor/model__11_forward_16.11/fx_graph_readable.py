class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[]", primals_2: "f32[4, 32, 32]", primals_3: "f32[4, 32, 32]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:490 in torch_dynamo_resume_in_step_at_490, code: variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
        pow_1: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 0.5);  primals_1 = None
        mul: "f32[4, 32, 32]" = torch.ops.aten.mul.Tensor(pow_1, primals_2);  pow_1 = primals_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:492 in torch_dynamo_resume_in_step_at_490, code: pred_prev_sample = pred_prev_sample + variance
        add: "f32[4, 32, 32]" = torch.ops.aten.add.Tensor(primals_3, mul);  primals_3 = mul = None
        return (add,)
        