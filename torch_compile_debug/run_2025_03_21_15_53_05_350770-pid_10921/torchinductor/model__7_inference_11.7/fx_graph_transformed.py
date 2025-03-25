class <lambda>(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 32, 32]" = torch.ops.prims.inductor_random.default([4, 32, 32], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        return (inductor_random_default,)
        