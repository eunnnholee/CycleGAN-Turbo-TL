class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1]", arg1_1: "i64[]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddpm.py:552 in previous_timestep, code: index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
        eq: "b8[1]" = torch.ops.aten.eq.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (eq,)
        