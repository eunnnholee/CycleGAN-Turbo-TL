class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:552 in torch_dynamo_resume_in_previous_timestep_at_552, code: index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
        select: "i64[]" = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:553 in torch_dynamo_resume_in_previous_timestep_at_552, code: if index == self.timesteps.shape[0] - 1:
        eq: "b8[]" = torch.ops.aten.eq.Scalar(select, 0)
        return (select, eq)
        