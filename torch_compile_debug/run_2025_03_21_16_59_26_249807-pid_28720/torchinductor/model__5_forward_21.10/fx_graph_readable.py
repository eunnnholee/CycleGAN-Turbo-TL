class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 3, 256, 256]", primals_2: "f32[3]", primals_3: "f32[3]", primals_4: "f32[768, 3, 32, 32]", primals_5: "f32[768]", primals_6: "f32[50, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[2304]", primals_12: "f32[2304, 768]", primals_13: "f32[768, 768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[3072, 768]", primals_18: "f32[3072]", primals_19: "f32[768, 3072]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[2304]", primals_24: "f32[2304, 768]", primals_25: "f32[768, 768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[3072, 768]", primals_30: "f32[3072]", primals_31: "f32[768, 3072]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[2304]", primals_36: "f32[2304, 768]", primals_37: "f32[768, 768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[3072, 768]", primals_42: "f32[3072]", primals_43: "f32[768, 3072]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[2304]", primals_48: "f32[2304, 768]", primals_49: "f32[768, 768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[3072, 768]", primals_54: "f32[3072]", primals_55: "f32[768, 3072]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[2304]", primals_60: "f32[2304, 768]", primals_61: "f32[768, 768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[3072, 768]", primals_66: "f32[3072]", primals_67: "f32[768, 3072]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[2304]", primals_72: "f32[2304, 768]", primals_73: "f32[768, 768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[3072, 768]", primals_78: "f32[3072]", primals_79: "f32[768, 3072]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[2304]", primals_84: "f32[2304, 768]", primals_85: "f32[768, 768]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[3072, 768]", primals_90: "f32[3072]", primals_91: "f32[768, 3072]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[2304]", primals_96: "f32[2304, 768]", primals_97: "f32[768, 768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[3072, 768]", primals_102: "f32[3072]", primals_103: "f32[768, 3072]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[2304]", primals_108: "f32[2304, 768]", primals_109: "f32[768, 768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[3072, 768]", primals_114: "f32[3072]", primals_115: "f32[768, 3072]", primals_116: "f32[768]", primals_117: "f32[768]", primals_118: "f32[768]", primals_119: "f32[2304]", primals_120: "f32[2304, 768]", primals_121: "f32[768, 768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[3072, 768]", primals_126: "f32[3072]", primals_127: "f32[768, 3072]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[2304]", primals_132: "f32[2304, 768]", primals_133: "f32[768, 768]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[3072, 768]", primals_138: "f32[3072]", primals_139: "f32[768, 3072]", primals_140: "f32[768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[2304]", primals_144: "f32[2304, 768]", primals_145: "f32[768, 768]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[3072, 768]", primals_150: "f32[3072]", primals_151: "f32[768, 3072]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768]", primals_155: "f32[768, 512]", primals_156: "f32[256, 768, 3, 3]", primals_157: "f32[256]", primals_158: "f32[6912]", primals_159: "f32[256]", primals_160: "f32[256, 1, 4, 4]", primals_161: "f32[1, 256, 1, 1]", primals_162: "f32[1]", primals_163: "f32[256]", primals_164: "f32[1]", primals_165: "f32[256, 768, 3, 3]", primals_166: "f32[256]", primals_167: "f32[6912]", primals_168: "f32[256]", primals_169: "f32[256, 1, 4, 4]", primals_170: "f32[1, 256, 1, 1]", primals_171: "f32[1]", primals_172: "f32[256]", primals_173: "f32[1]", primals_174: "f32[256, 512]", primals_175: "f32[256]", primals_176: "f32[512]", primals_177: "f32[256]", primals_178: "f32[1, 256]", primals_179: "f32[1]", primals_180: "f32[256]", primals_181: "f32[1]"):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[7]" = torch.ops.prims.inductor_seeds.default(7, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_2: "f32[2, 1, 1, 1]" = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:23 in rand_brightness, code: x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        sub: "f32[2, 1, 1, 1]" = torch.ops.aten.sub.Tensor(inductor_random_default_2, 0.5);  inductor_random_default_2 = None
        add: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(primals_1, sub);  primals_1 = sub = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:28 in rand_saturation, code: x_mean = x.mean(dim=1, keepdim=True)
        mean: "f32[2, 1, 256, 256]" = torch.ops.aten.mean.dim(add, [1], True)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:29 in rand_saturation, code: x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        sub_1: "f32[2, 3, 256, 256]" = torch.ops.aten.sub.Tensor(add, mean);  add = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1: "f32[2, 1, 1, 1]" = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:29 in rand_saturation, code: x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        mul: "f32[2, 1, 1, 1]" = torch.ops.aten.mul.Tensor(inductor_random_default_1, 2);  inductor_random_default_1 = None
        mul_1: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(sub_1, mul);  sub_1 = mul = None
        add_1: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_1, mean);  mul_1 = mean = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:34 in rand_contrast, code: x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        mean_1: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(add_1, [1, 2, 3], True)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:35 in rand_contrast, code: x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        sub_2: "f32[2, 3, 256, 256]" = torch.ops.aten.sub.Tensor(add_1, mean_1);  add_1 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default: "f32[2, 1, 1, 1]" = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:35 in rand_contrast, code: x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        add_2: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(inductor_random_default, 0.5);  inductor_random_default = None
        mul_2: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(sub_2, add_2);  sub_2 = add_2 = None
        add_3: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_2, mean_1);  mul_2 = mean_1 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_randint_default_3: "i64[2, 1, 1]" = torch.ops.prims.inductor_randint.default(-32, 33, [2, 1, 1], inductor_lookup_seed_default_3);  inductor_lookup_seed_default_3 = None
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_randint_default_2: "i64[2, 1, 1]" = torch.ops.prims.inductor_randint.default(-32, 33, [2, 1, 1], inductor_lookup_seed_default_4);  inductor_lookup_seed_default_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:44 in rand_translation, code: torch.arange(x.size(0), dtype=torch.long, device=x.device),
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:45 in rand_translation, code: torch.arange(x.size(2), dtype=torch.long, device=x.device),
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:43 in rand_translation, code: grid_batch, grid_x, grid_y = torch.meshgrid(
        view: "i64[2, 1, 1]" = torch.ops.aten.view.default(iota, [-1, 1, 1]);  iota = None
        expand: "i64[2, 256, 256]" = torch.ops.aten.expand.default(view, [2, 256, 256])
        view_1: "i64[1, 256, 1]" = torch.ops.aten.view.default(iota_1, [1, -1, 1])
        expand_1: "i64[2, 256, 256]" = torch.ops.aten.expand.default(view_1, [2, 256, 256]);  view_1 = None
        view_2: "i64[1, 1, 256]" = torch.ops.aten.view.default(iota_1, [1, 1, -1]);  iota_1 = None
        expand_2: "i64[2, 256, 256]" = torch.ops.aten.expand.default(view_2, [2, 256, 256]);  view_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:48 in rand_translation, code: grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        add_4: "i64[2, 256, 256]" = torch.ops.aten.add.Tensor(expand_1, inductor_randint_default_3);  expand_1 = inductor_randint_default_3 = None
        add_5: "i64[2, 256, 256]" = torch.ops.aten.add.Tensor(add_4, 1);  add_4 = None
        clamp_min: "i64[2, 256, 256]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
        clamp_max: "i64[2, 256, 256]" = torch.ops.aten.clamp_max.default(clamp_min, 257);  clamp_min = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:49 in rand_translation, code: grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        add_6: "i64[2, 256, 256]" = torch.ops.aten.add.Tensor(expand_2, inductor_randint_default_2);  expand_2 = inductor_randint_default_2 = None
        add_7: "i64[2, 256, 256]" = torch.ops.aten.add.Tensor(add_6, 1);  add_6 = None
        clamp_min_1: "i64[2, 256, 256]" = torch.ops.aten.clamp_min.default(add_7, 0);  add_7 = None
        clamp_max_1: "i64[2, 256, 256]" = torch.ops.aten.clamp_max.default(clamp_min_1, 257);  clamp_min_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[2, 3, 258, 258]" = torch.ops.aten.constant_pad_nd.default(add_3, [1, 1, 1, 1, 0, 0, 0, 0], 0.0);  add_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:51 in rand_translation, code: x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        permute: "f32[2, 258, 258, 3]" = torch.ops.aten.permute.default(constant_pad_nd, [0, 2, 3, 1]);  constant_pad_nd = None
        clone: "f32[2, 258, 258, 3]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        index: "f32[2, 256, 256, 3]" = torch.ops.aten.index.Tensor(clone, [expand, clamp_max, clamp_max_1]);  clone = expand = clamp_max = clamp_max_1 = None
        permute_1: "f32[2, 3, 256, 256]" = torch.ops.aten.permute.default(index, [0, 3, 1, 2]);  index = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_randint_default_1: "i64[2, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 257, [2, 1, 1], inductor_lookup_seed_default_5);  inductor_lookup_seed_default_5 = None
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6);  inductor_seeds_default = None
        inductor_randint_default: "i64[2, 1, 1]" = torch.ops.prims.inductor_randint.default(0, 257, [2, 1, 1], inductor_lookup_seed_default_6);  inductor_lookup_seed_default_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:61 in rand_cutout, code: torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:59 in rand_cutout, code: grid_batch, grid_x, grid_y = torch.meshgrid(
        expand_3: "i64[2, 128, 128]" = torch.ops.aten.expand.default(view, [2, 128, 128]);  view = None
        view_4: "i64[1, 128, 1]" = torch.ops.aten.view.default(iota_4, [1, -1, 1])
        expand_4: "i64[2, 128, 128]" = torch.ops.aten.expand.default(view_4, [2, 128, 128]);  view_4 = None
        view_5: "i64[1, 1, 128]" = torch.ops.aten.view.default(iota_4, [1, 1, -1]);  iota_4 = None
        expand_5: "i64[2, 128, 128]" = torch.ops.aten.expand.default(view_5, [2, 128, 128]);  view_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:64 in rand_cutout, code: grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        add_8: "i64[2, 128, 128]" = torch.ops.aten.add.Tensor(expand_4, inductor_randint_default_1);  expand_4 = inductor_randint_default_1 = None
        sub_3: "i64[2, 128, 128]" = torch.ops.aten.sub.Tensor(add_8, 64);  add_8 = None
        clamp_min_2: "i64[2, 128, 128]" = torch.ops.aten.clamp_min.default(sub_3, 0);  sub_3 = None
        clamp_max_2: "i64[2, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:65 in rand_cutout, code: grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        add_9: "i64[2, 128, 128]" = torch.ops.aten.add.Tensor(expand_5, inductor_randint_default);  expand_5 = inductor_randint_default = None
        sub_4: "i64[2, 128, 128]" = torch.ops.aten.sub.Tensor(add_9, 64);  add_9 = None
        clamp_min_3: "i64[2, 128, 128]" = torch.ops.aten.clamp_min.default(sub_4, 0);  sub_4 = None
        clamp_max_3: "i64[2, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:66 in rand_cutout, code: mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        full_default: "f32[2, 256, 256]" = torch.ops.aten.full.default([2, 256, 256], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:67 in rand_cutout, code: mask[grid_batch, grid_x, grid_y] = 0
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put: "f32[2, 256, 256]" = torch.ops.aten.index_put.default(full_default, [expand_3, clamp_max_2, clamp_max_3], full_default_1);  full_default = expand_3 = clamp_max_2 = clamp_max_3 = full_default_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:68 in rand_cutout, code: x = x * mask.unsqueeze(1)
        unsqueeze_1: "f32[2, 1, 256, 256]" = torch.ops.aten.unsqueeze.default(index_put, 1);  index_put = None
        mul_3: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(permute_1, unsqueeze_1);  permute_1 = unsqueeze_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/DiffAugment_pytorch.py:18 in DiffAugment, code: x = x.contiguous()
        clone_1: "f32[2, 3, 256, 256]" = torch.ops.aten.clone.default(mul_3, memory_format = torch.contiguous_format);  mul_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:128 in __call__, code: x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        mul_4: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(clone_1, 0.5);  clone_1 = None
        add_10: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_4, 0.5);  mul_4 = None
        _adaptive_avg_pool2d: "f32[2, 3, 224, 224]" = torch.ops.aten._adaptive_avg_pool2d.default(add_10, [224, 224]);  add_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:129 in __call__, code: x = x - self.image_mean[:, None, None].to(x.device)
        unsqueeze_2: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        unsqueeze_3: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        device_put: "f32[3, 1, 1]" = torch.ops.prims.device_put.default(unsqueeze_3, device(type='cuda', index=0));  unsqueeze_3 = None
        sub_5: "f32[2, 3, 224, 224]" = torch.ops.aten.sub.Tensor(_adaptive_avg_pool2d, device_put);  _adaptive_avg_pool2d = device_put = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:130 in __call__, code: x /= self.image_std[:, None, None].to(x.device)
        unsqueeze_4: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 1);  primals_3 = None
        unsqueeze_5: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        device_put_1: "f32[3, 1, 1]" = torch.ops.prims.device_put.default(unsqueeze_5, device(type='cuda', index=0));  unsqueeze_5 = None
        div: "f32[2, 3, 224, 224]" = torch.ops.aten.div.Tensor(sub_5, device_put_1);  sub_5 = device_put_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:106 in forward_custom, code: x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        convolution: "f32[2, 768, 7, 7]" = torch.ops.aten.convolution.default(div, primals_4, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1);  div = primals_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:107 in forward_custom, code: x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        view_6: "f32[2, 768, 49]" = torch.ops.aten.view.default(convolution, [2, 768, -1]);  convolution = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:108 in forward_custom, code: x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        permute_2: "f32[2, 49, 768]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:109 in forward_custom, code: x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        full_default_2: "f32[2, 1, 768]" = torch.ops.aten.full.default([2, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add_11: "f32[2, 1, 768]" = torch.ops.aten.add.Tensor(primals_5, full_default_2);  primals_5 = full_default_2 = None
        cat: "f32[2, 50, 768]" = torch.ops.aten.cat.default([add_11, permute_2], 1);  add_11 = permute_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:110 in forward_custom, code: x = x + self.model.positional_embedding.to(x.dtype)
        add_12: "f32[2, 50, 768]" = torch.ops.aten.add.Tensor(cat, primals_6);  cat = primals_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        var_mean = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem: "f32[2, 50, 1]" = var_mean[0]
        getitem_1: "f32[2, 50, 1]" = var_mean[1];  var_mean = None
        add_13: "f32[2, 50, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[2, 50, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_6: "f32[2, 50, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_1);  add_12 = getitem_1 = None
        mul_5: "f32[2, 50, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
        mul_6: "f32[2, 50, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_7);  mul_5 = primals_7 = None
        add_14: "f32[2, 50, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_8);  mul_6 = primals_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:113 in forward_custom, code: x = x.permute(1, 0, 2)  # NLD -> LND
        permute_3: "f32[50, 2, 768]" = torch.ops.aten.permute.default(add_14, [1, 0, 2]);  add_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_2: "f32[50, 2, 768]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
        getitem_2: "f32[50, 2, 1]" = var_mean_1[0]
        getitem_3: "f32[50, 2, 1]" = var_mean_1[1];  var_mean_1 = None
        add_15: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_7: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_2, getitem_3);  clone_2 = getitem_3 = None
        mul_7: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
        mul_8: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_9);  mul_7 = primals_9 = None
        add_16: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_10);  mul_8 = primals_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_7: "f32[100, 768]" = torch.ops.aten.view.default(add_16, [100, 768]);  add_16 = None
        permute_4: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_11, view_7, permute_4);  primals_11 = view_7 = permute_4 = None
        view_8: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm, [50, 2, 2304]);  addmm = None
        view_9: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_8, [50, 2, 3, 768]);  view_8 = None
        unsqueeze_6: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_9, 0);  view_9 = None
        permute_5: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_5, -2);  permute_5 = None
        clone_3: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_3, 0, 0)
        select_1: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_3, 0, 1)
        select_2: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_3, 0, 2);  clone_3 = None
        view_10: "f32[50, 24, 64]" = torch.ops.aten.view.default(select, [50, 24, 64]);  select = None
        permute_6: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        view_11: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_1, [50, 24, 64]);  select_1 = None
        permute_7: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        view_12: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_2, [50, 24, 64]);  select_2 = None
        permute_8: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        view_13: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_6, [2, 12, 50, 64]);  permute_6 = None
        view_14: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_7, [2, 12, 50, 64]);  permute_7 = None
        view_15: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_8, [2, 12, 50, 64]);  permute_8 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_13, view_14, view_15, None, False);  view_13 = view_14 = view_15 = None
        getitem_4: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_9: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_4, [2, 0, 1, 3]);  getitem_4 = None
        clone_4: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_16: "f32[100, 768]" = torch.ops.aten.view.default(clone_4, [100, 768]);  clone_4 = None
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        addmm_1: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_14, view_16, permute_10);  primals_14 = view_16 = permute_10 = None
        view_17: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_1, [50, 2, 768]);  addmm_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_17: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(permute_3, view_17);  permute_3 = view_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_5: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
        var_mean_2 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
        getitem_8: "f32[50, 2, 1]" = var_mean_2[0]
        getitem_9: "f32[50, 2, 1]" = var_mean_2[1];  var_mean_2 = None
        add_18: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_5, getitem_9);  clone_5 = getitem_9 = None
        mul_9: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_2);  sub_8 = rsqrt_2 = None
        mul_10: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_15);  mul_9 = primals_15 = None
        add_19: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_16);  mul_10 = primals_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_18: "f32[100, 768]" = torch.ops.aten.view.default(add_19, [100, 768]);  add_19 = None
        permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        addmm_2: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_18, view_18, permute_11);  primals_18 = view_18 = permute_11 = None
        view_19: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_2, [50, 2, 3072]);  addmm_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_11: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_19, 1.702)
        sigmoid: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_11);  mul_11 = None
        mul_12: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_19, sigmoid);  view_19 = sigmoid = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_20: "f32[100, 3072]" = torch.ops.aten.view.default(mul_12, [100, 3072]);  mul_12 = None
        permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        addmm_3: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_20, view_20, permute_12);  primals_20 = view_20 = permute_12 = None
        view_21: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_3, [50, 2, 768]);  addmm_3 = None
        add_20: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_17, view_21);  add_17 = view_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_6: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
        var_mean_3 = torch.ops.aten.var_mean.correction(clone_6, [2], correction = 0, keepdim = True)
        getitem_10: "f32[50, 2, 1]" = var_mean_3[0]
        getitem_11: "f32[50, 2, 1]" = var_mean_3[1];  var_mean_3 = None
        add_21: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_9: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_6, getitem_11);  clone_6 = getitem_11 = None
        mul_13: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_3);  sub_9 = rsqrt_3 = None
        mul_14: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_13, primals_21);  mul_13 = primals_21 = None
        add_22: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_14, primals_22);  mul_14 = primals_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_22: "f32[100, 768]" = torch.ops.aten.view.default(add_22, [100, 768]);  add_22 = None
        permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_4: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_13);  primals_23 = view_22 = permute_13 = None
        view_23: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_4, [50, 2, 2304]);  addmm_4 = None
        view_24: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_23, [50, 2, 3, 768]);  view_23 = None
        unsqueeze_7: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_24, 0);  view_24 = None
        permute_14: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_1: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_14, -2);  permute_14 = None
        clone_7: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_7, 0, 0)
        select_4: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_7, 0, 1)
        select_5: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_25: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_3, [50, 24, 64]);  select_3 = None
        permute_15: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        view_26: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_4, [50, 24, 64]);  select_4 = None
        permute_16: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_26, [1, 0, 2]);  view_26 = None
        view_27: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_5, [50, 24, 64]);  select_5 = None
        permute_17: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        view_28: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_15, [2, 12, 50, 64]);  permute_15 = None
        view_29: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_16, [2, 12, 50, 64]);  permute_16 = None
        view_30: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_17, [2, 12, 50, 64]);  permute_17 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_28, view_29, view_30, None, False);  view_28 = view_29 = view_30 = None
        getitem_12: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_18: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_12, [2, 0, 1, 3]);  getitem_12 = None
        clone_8: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_31: "f32[100, 768]" = torch.ops.aten.view.default(clone_8, [100, 768]);  clone_8 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        addmm_5: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_26, view_31, permute_19);  primals_26 = view_31 = permute_19 = None
        view_32: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_5, [50, 2, 768]);  addmm_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_23: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_20, view_32);  add_20 = view_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_9: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
        var_mean_4 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
        getitem_16: "f32[50, 2, 1]" = var_mean_4[0]
        getitem_17: "f32[50, 2, 1]" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_10: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_9, getitem_17);  clone_9 = getitem_17 = None
        mul_15: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = rsqrt_4 = None
        mul_16: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_27);  mul_15 = primals_27 = None
        add_25: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_28);  mul_16 = primals_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_33: "f32[100, 768]" = torch.ops.aten.view.default(add_25, [100, 768]);  add_25 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        addmm_6: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_30, view_33, permute_20);  primals_30 = view_33 = permute_20 = None
        view_34: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_6, [50, 2, 3072]);  addmm_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_17: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_34, 1.702)
        sigmoid_1: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_17);  mul_17 = None
        mul_18: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_34, sigmoid_1);  view_34 = sigmoid_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_35: "f32[100, 3072]" = torch.ops.aten.view.default(mul_18, [100, 3072]);  mul_18 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        addmm_7: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_32, view_35, permute_21);  primals_32 = view_35 = permute_21 = None
        view_36: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_7, [50, 2, 768]);  addmm_7 = None
        add_26: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_23, view_36);  add_23 = view_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_10: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        var_mean_5 = torch.ops.aten.var_mean.correction(clone_10, [2], correction = 0, keepdim = True)
        getitem_18: "f32[50, 2, 1]" = var_mean_5[0]
        getitem_19: "f32[50, 2, 1]" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_5: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_11: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_10, getitem_19);  clone_10 = getitem_19 = None
        mul_19: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
        mul_20: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_19, primals_33);  mul_19 = primals_33 = None
        add_28: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_20, primals_34);  mul_20 = primals_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_37: "f32[100, 768]" = torch.ops.aten.view.default(add_28, [100, 768]);  add_28 = None
        permute_22: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_8: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_35, view_37, permute_22);  primals_35 = view_37 = permute_22 = None
        view_38: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_8, [50, 2, 2304]);  addmm_8 = None
        view_39: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_38, [50, 2, 3, 768]);  view_38 = None
        unsqueeze_8: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_39, 0);  view_39 = None
        permute_23: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_2: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_23, -2);  permute_23 = None
        clone_11: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_11, 0, 0)
        select_7: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_11, 0, 1)
        select_8: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_40: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_6, [50, 24, 64]);  select_6 = None
        permute_24: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_40, [1, 0, 2]);  view_40 = None
        view_41: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_7, [50, 24, 64]);  select_7 = None
        permute_25: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_41, [1, 0, 2]);  view_41 = None
        view_42: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_8, [50, 24, 64]);  select_8 = None
        permute_26: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        view_43: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_24, [2, 12, 50, 64]);  permute_24 = None
        view_44: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_25, [2, 12, 50, 64]);  permute_25 = None
        view_45: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_26, [2, 12, 50, 64]);  permute_26 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_43, view_44, view_45, None, False);  view_43 = view_44 = view_45 = None
        getitem_20: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_27: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_20, [2, 0, 1, 3]);  getitem_20 = None
        clone_12: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_46: "f32[100, 768]" = torch.ops.aten.view.default(clone_12, [100, 768]);  clone_12 = None
        permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        addmm_9: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_38, view_46, permute_28);  primals_38 = view_46 = permute_28 = None
        view_47: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_9, [50, 2, 768]);  addmm_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_29: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_26, view_47);  add_26 = view_47 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_13: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_29, memory_format = torch.contiguous_format)
        var_mean_6 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
        getitem_24: "f32[50, 2, 1]" = var_mean_6[0]
        getitem_25: "f32[50, 2, 1]" = var_mean_6[1];  var_mean_6 = None
        add_30: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_13, getitem_25);  clone_13 = getitem_25 = None
        mul_21: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
        mul_22: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_39);  mul_21 = primals_39 = None
        add_31: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_40);  mul_22 = primals_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_48: "f32[100, 768]" = torch.ops.aten.view.default(add_31, [100, 768]);  add_31 = None
        permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        addmm_10: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_42, view_48, permute_29);  primals_42 = view_48 = permute_29 = None
        view_49: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_10, [50, 2, 3072]);  addmm_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_23: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_49, 1.702)
        sigmoid_2: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_23);  mul_23 = None
        mul_24: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_49, sigmoid_2);  view_49 = sigmoid_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_50: "f32[100, 3072]" = torch.ops.aten.view.default(mul_24, [100, 3072]);  mul_24 = None
        permute_30: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        addmm_11: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_44, view_50, permute_30);  primals_44 = view_50 = permute_30 = None
        view_51: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_11, [50, 2, 768]);  addmm_11 = None
        add_32: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_29, view_51);  add_29 = view_51 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_14: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format)
        var_mean_7 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
        getitem_26: "f32[50, 2, 1]" = var_mean_7[0]
        getitem_27: "f32[50, 2, 1]" = var_mean_7[1];  var_mean_7 = None
        add_33: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_14, getitem_27);  clone_14 = getitem_27 = None
        mul_25: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = rsqrt_7 = None
        mul_26: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_45);  mul_25 = primals_45 = None
        add_34: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_46);  mul_26 = primals_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_52: "f32[100, 768]" = torch.ops.aten.view.default(add_34, [100, 768]);  add_34 = None
        permute_31: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        addmm_12: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_47, view_52, permute_31);  primals_47 = view_52 = permute_31 = None
        view_53: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_12, [50, 2, 2304]);  addmm_12 = None
        view_54: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_53, [50, 2, 3, 768]);  view_53 = None
        unsqueeze_9: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_54, 0);  view_54 = None
        permute_32: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_3: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_32, -2);  permute_32 = None
        clone_15: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_15, 0, 0)
        select_10: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_15, 0, 1)
        select_11: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_15, 0, 2);  clone_15 = None
        view_55: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_9, [50, 24, 64]);  select_9 = None
        permute_33: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_55, [1, 0, 2]);  view_55 = None
        view_56: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_10, [50, 24, 64]);  select_10 = None
        permute_34: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_56, [1, 0, 2]);  view_56 = None
        view_57: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_11, [50, 24, 64]);  select_11 = None
        permute_35: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        view_58: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_33, [2, 12, 50, 64]);  permute_33 = None
        view_59: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_34, [2, 12, 50, 64]);  permute_34 = None
        view_60: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_35, [2, 12, 50, 64]);  permute_35 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_58, view_59, view_60, None, False);  view_58 = view_59 = view_60 = None
        getitem_28: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3]);  getitem_28 = None
        clone_16: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_61: "f32[100, 768]" = torch.ops.aten.view.default(clone_16, [100, 768]);  clone_16 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        addmm_13: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_50, view_61, permute_37);  primals_50 = view_61 = permute_37 = None
        view_62: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_13, [50, 2, 768]);  addmm_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_35: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_32, view_62);  add_32 = view_62 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_17: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
        var_mean_8 = torch.ops.aten.var_mean.correction(clone_17, [2], correction = 0, keepdim = True)
        getitem_32: "f32[50, 2, 1]" = var_mean_8[0]
        getitem_33: "f32[50, 2, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_17, getitem_33);  clone_17 = getitem_33 = None
        mul_27: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = rsqrt_8 = None
        mul_28: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_51);  mul_27 = primals_51 = None
        add_37: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_52);  mul_28 = primals_52 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_63: "f32[100, 768]" = torch.ops.aten.view.default(add_37, [100, 768]);  add_37 = None
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        addmm_14: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_54, view_63, permute_38);  primals_54 = view_63 = permute_38 = None
        view_64: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_14, [50, 2, 3072]);  addmm_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_29: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_64, 1.702)
        sigmoid_3: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_29);  mul_29 = None
        mul_30: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_64, sigmoid_3);  view_64 = sigmoid_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_65: "f32[100, 3072]" = torch.ops.aten.view.default(mul_30, [100, 3072]);  mul_30 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        addmm_15: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_56, view_65, permute_39);  primals_56 = view_65 = permute_39 = None
        view_66: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_15, [50, 2, 768]);  addmm_15 = None
        add_38: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_35, view_66);  add_35 = view_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:119 in forward_custom, code: x1.append(x.permute(1, 0, 2))
        permute_40: "f32[2, 50, 768]" = torch.ops.aten.permute.default(add_38, [1, 0, 2])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_18: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
        var_mean_9 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
        getitem_34: "f32[50, 2, 1]" = var_mean_9[0]
        getitem_35: "f32[50, 2, 1]" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_9: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_15: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_18, getitem_35);  clone_18 = getitem_35 = None
        mul_31: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_31, primals_57);  mul_31 = primals_57 = None
        add_40: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_32, primals_58);  mul_32 = primals_58 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_67: "f32[100, 768]" = torch.ops.aten.view.default(add_40, [100, 768]);  add_40 = None
        permute_41: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_16: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_59, view_67, permute_41);  primals_59 = view_67 = permute_41 = None
        view_68: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_16, [50, 2, 2304]);  addmm_16 = None
        view_69: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_68, [50, 2, 3, 768]);  view_68 = None
        unsqueeze_10: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_69, 0);  view_69 = None
        permute_42: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_4: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_42, -2);  permute_42 = None
        clone_19: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_19, 0, 0)
        select_13: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_19, 0, 1)
        select_14: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_19, 0, 2);  clone_19 = None
        view_70: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_12, [50, 24, 64]);  select_12 = None
        permute_43: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_70, [1, 0, 2]);  view_70 = None
        view_71: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_13, [50, 24, 64]);  select_13 = None
        permute_44: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_71, [1, 0, 2]);  view_71 = None
        view_72: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_14, [50, 24, 64]);  select_14 = None
        permute_45: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        view_73: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_43, [2, 12, 50, 64]);  permute_43 = None
        view_74: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_44, [2, 12, 50, 64]);  permute_44 = None
        view_75: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_45, [2, 12, 50, 64]);  permute_45 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_73, view_74, view_75, None, False);  view_73 = view_74 = view_75 = None
        getitem_36: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_36, [2, 0, 1, 3]);  getitem_36 = None
        clone_20: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_76: "f32[100, 768]" = torch.ops.aten.view.default(clone_20, [100, 768]);  clone_20 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        addmm_17: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_62, view_76, permute_47);  primals_62 = view_76 = permute_47 = None
        view_77: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_17, [50, 2, 768]);  addmm_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_41: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_38, view_77);  add_38 = view_77 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_21: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
        var_mean_10 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
        getitem_40: "f32[50, 2, 1]" = var_mean_10[0]
        getitem_41: "f32[50, 2, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_21, getitem_41);  clone_21 = getitem_41 = None
        mul_33: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_34: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_63);  mul_33 = primals_63 = None
        add_43: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_64);  mul_34 = primals_64 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_78: "f32[100, 768]" = torch.ops.aten.view.default(add_43, [100, 768]);  add_43 = None
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        addmm_18: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_66, view_78, permute_48);  primals_66 = view_78 = permute_48 = None
        view_79: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_18, [50, 2, 3072]);  addmm_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_35: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_79, 1.702)
        sigmoid_4: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_35);  mul_35 = None
        mul_36: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_79, sigmoid_4);  view_79 = sigmoid_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_80: "f32[100, 3072]" = torch.ops.aten.view.default(mul_36, [100, 3072]);  mul_36 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        addmm_19: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_68, view_80, permute_49);  primals_68 = view_80 = permute_49 = None
        view_81: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_19, [50, 2, 768]);  addmm_19 = None
        add_44: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_41, view_81);  add_41 = view_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_22: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
        var_mean_11 = torch.ops.aten.var_mean.correction(clone_22, [2], correction = 0, keepdim = True)
        getitem_42: "f32[50, 2, 1]" = var_mean_11[0]
        getitem_43: "f32[50, 2, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_11: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_17: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_22, getitem_43);  clone_22 = getitem_43 = None
        mul_37: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_38: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_69);  mul_37 = primals_69 = None
        add_46: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_70);  mul_38 = primals_70 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_82: "f32[100, 768]" = torch.ops.aten.view.default(add_46, [100, 768]);  add_46 = None
        permute_50: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_20: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_71, view_82, permute_50);  primals_71 = view_82 = permute_50 = None
        view_83: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_20, [50, 2, 2304]);  addmm_20 = None
        view_84: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_83, [50, 2, 3, 768]);  view_83 = None
        unsqueeze_11: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_84, 0);  view_84 = None
        permute_51: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_5: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_51, -2);  permute_51 = None
        clone_23: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_23, 0, 0)
        select_16: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_23, 0, 1)
        select_17: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_23, 0, 2);  clone_23 = None
        view_85: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_15, [50, 24, 64]);  select_15 = None
        permute_52: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_85, [1, 0, 2]);  view_85 = None
        view_86: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_16, [50, 24, 64]);  select_16 = None
        permute_53: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_86, [1, 0, 2]);  view_86 = None
        view_87: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_17, [50, 24, 64]);  select_17 = None
        permute_54: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        view_88: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_52, [2, 12, 50, 64]);  permute_52 = None
        view_89: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_53, [2, 12, 50, 64]);  permute_53 = None
        view_90: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_54, [2, 12, 50, 64]);  permute_54 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_88, view_89, view_90, None, False);  view_88 = view_89 = view_90 = None
        getitem_44: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_55: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_44, [2, 0, 1, 3]);  getitem_44 = None
        clone_24: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_91: "f32[100, 768]" = torch.ops.aten.view.default(clone_24, [100, 768]);  clone_24 = None
        permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        addmm_21: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_74, view_91, permute_56);  primals_74 = view_91 = permute_56 = None
        view_92: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_21, [50, 2, 768]);  addmm_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_47: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_44, view_92);  add_44 = view_92 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_25: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_47, memory_format = torch.contiguous_format)
        var_mean_12 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
        getitem_48: "f32[50, 2, 1]" = var_mean_12[0]
        getitem_49: "f32[50, 2, 1]" = var_mean_12[1];  var_mean_12 = None
        add_48: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_18: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_25, getitem_49);  clone_25 = getitem_49 = None
        mul_39: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_40: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_39, primals_75);  mul_39 = primals_75 = None
        add_49: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_40, primals_76);  mul_40 = primals_76 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_93: "f32[100, 768]" = torch.ops.aten.view.default(add_49, [100, 768]);  add_49 = None
        permute_57: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        addmm_22: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_78, view_93, permute_57);  primals_78 = view_93 = permute_57 = None
        view_94: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_22, [50, 2, 3072]);  addmm_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_41: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_94, 1.702)
        sigmoid_5: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_41);  mul_41 = None
        mul_42: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_94, sigmoid_5);  view_94 = sigmoid_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_95: "f32[100, 3072]" = torch.ops.aten.view.default(mul_42, [100, 3072]);  mul_42 = None
        permute_58: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        addmm_23: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_80, view_95, permute_58);  primals_80 = view_95 = permute_58 = None
        view_96: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_23, [50, 2, 768]);  addmm_23 = None
        add_50: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_47, view_96);  add_47 = view_96 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_26: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
        var_mean_13 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
        getitem_50: "f32[50, 2, 1]" = var_mean_13[0]
        getitem_51: "f32[50, 2, 1]" = var_mean_13[1];  var_mean_13 = None
        add_51: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_13: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_19: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_26, getitem_51);  clone_26 = getitem_51 = None
        mul_43: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
        mul_44: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_81);  mul_43 = primals_81 = None
        add_52: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_82);  mul_44 = primals_82 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_97: "f32[100, 768]" = torch.ops.aten.view.default(add_52, [100, 768]);  add_52 = None
        permute_59: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        addmm_24: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_83, view_97, permute_59);  primals_83 = view_97 = permute_59 = None
        view_98: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_24, [50, 2, 2304]);  addmm_24 = None
        view_99: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_98, [50, 2, 3, 768]);  view_98 = None
        unsqueeze_12: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_99, 0);  view_99 = None
        permute_60: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_6: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_60, -2);  permute_60 = None
        clone_27: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_27, 0, 0)
        select_19: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_27, 0, 1)
        select_20: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_27, 0, 2);  clone_27 = None
        view_100: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_18, [50, 24, 64]);  select_18 = None
        permute_61: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_100, [1, 0, 2]);  view_100 = None
        view_101: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_19, [50, 24, 64]);  select_19 = None
        permute_62: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_101, [1, 0, 2]);  view_101 = None
        view_102: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_20, [50, 24, 64]);  select_20 = None
        permute_63: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        view_103: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_61, [2, 12, 50, 64]);  permute_61 = None
        view_104: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_62, [2, 12, 50, 64]);  permute_62 = None
        view_105: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_63, [2, 12, 50, 64]);  permute_63 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_103, view_104, view_105, None, False);  view_103 = view_104 = view_105 = None
        getitem_52: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_64: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_52, [2, 0, 1, 3]);  getitem_52 = None
        clone_28: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106: "f32[100, 768]" = torch.ops.aten.view.default(clone_28, [100, 768]);  clone_28 = None
        permute_65: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        addmm_25: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_86, view_106, permute_65);  primals_86 = view_106 = permute_65 = None
        view_107: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_25, [50, 2, 768]);  addmm_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_53: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_50, view_107);  add_50 = view_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_29: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_53, memory_format = torch.contiguous_format)
        var_mean_14 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
        getitem_56: "f32[50, 2, 1]" = var_mean_14[0]
        getitem_57: "f32[50, 2, 1]" = var_mean_14[1];  var_mean_14 = None
        add_54: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_20: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_57);  clone_29 = getitem_57 = None
        mul_45: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = rsqrt_14 = None
        mul_46: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_87);  mul_45 = primals_87 = None
        add_55: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_88);  mul_46 = primals_88 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_108: "f32[100, 768]" = torch.ops.aten.view.default(add_55, [100, 768]);  add_55 = None
        permute_66: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        addmm_26: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_90, view_108, permute_66);  primals_90 = view_108 = permute_66 = None
        view_109: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_26, [50, 2, 3072]);  addmm_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_47: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_109, 1.702)
        sigmoid_6: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_47);  mul_47 = None
        mul_48: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_109, sigmoid_6);  view_109 = sigmoid_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_110: "f32[100, 3072]" = torch.ops.aten.view.default(mul_48, [100, 3072]);  mul_48 = None
        permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        addmm_27: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_92, view_110, permute_67);  primals_92 = view_110 = permute_67 = None
        view_111: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_27, [50, 2, 768]);  addmm_27 = None
        add_56: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_53, view_111);  add_53 = view_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_30: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
        var_mean_15 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
        getitem_58: "f32[50, 2, 1]" = var_mean_15[0]
        getitem_59: "f32[50, 2, 1]" = var_mean_15[1];  var_mean_15 = None
        add_57: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_15: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_21: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_30, getitem_59);  clone_30 = getitem_59 = None
        mul_49: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = rsqrt_15 = None
        mul_50: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_93);  mul_49 = primals_93 = None
        add_58: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_94);  mul_50 = primals_94 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_112: "f32[100, 768]" = torch.ops.aten.view.default(add_58, [100, 768]);  add_58 = None
        permute_68: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        addmm_28: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_95, view_112, permute_68);  primals_95 = view_112 = permute_68 = None
        view_113: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_28, [50, 2, 2304]);  addmm_28 = None
        view_114: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_113, [50, 2, 3, 768]);  view_113 = None
        unsqueeze_13: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_114, 0);  view_114 = None
        permute_69: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_7: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_69, -2);  permute_69 = None
        clone_31: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_31, 0, 0)
        select_22: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_31, 0, 1)
        select_23: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_31, 0, 2);  clone_31 = None
        view_115: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_21, [50, 24, 64]);  select_21 = None
        permute_70: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_115, [1, 0, 2]);  view_115 = None
        view_116: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_22, [50, 24, 64]);  select_22 = None
        permute_71: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_116, [1, 0, 2]);  view_116 = None
        view_117: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_23, [50, 24, 64]);  select_23 = None
        permute_72: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_117, [1, 0, 2]);  view_117 = None
        view_118: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_70, [2, 12, 50, 64]);  permute_70 = None
        view_119: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_71, [2, 12, 50, 64]);  permute_71 = None
        view_120: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_72, [2, 12, 50, 64]);  permute_72 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_118, view_119, view_120, None, False);  view_118 = view_119 = view_120 = None
        getitem_60: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_73: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [2, 0, 1, 3]);  getitem_60 = None
        clone_32: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_121: "f32[100, 768]" = torch.ops.aten.view.default(clone_32, [100, 768]);  clone_32 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        addmm_29: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_98, view_121, permute_74);  primals_98 = view_121 = permute_74 = None
        view_122: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_29, [50, 2, 768]);  addmm_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_59: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_56, view_122);  add_56 = view_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_33: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
        var_mean_16 = torch.ops.aten.var_mean.correction(clone_33, [2], correction = 0, keepdim = True)
        getitem_64: "f32[50, 2, 1]" = var_mean_16[0]
        getitem_65: "f32[50, 2, 1]" = var_mean_16[1];  var_mean_16 = None
        add_60: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_22: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_33, getitem_65);  clone_33 = getitem_65 = None
        mul_51: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = rsqrt_16 = None
        mul_52: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_99);  mul_51 = primals_99 = None
        add_61: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_100);  mul_52 = primals_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_123: "f32[100, 768]" = torch.ops.aten.view.default(add_61, [100, 768]);  add_61 = None
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        addmm_30: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_102, view_123, permute_75);  primals_102 = view_123 = permute_75 = None
        view_124: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_30, [50, 2, 3072]);  addmm_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_53: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_124, 1.702)
        sigmoid_7: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_53);  mul_53 = None
        mul_54: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_124, sigmoid_7);  view_124 = sigmoid_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_125: "f32[100, 3072]" = torch.ops.aten.view.default(mul_54, [100, 3072]);  mul_54 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        addmm_31: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_104, view_125, permute_76);  primals_104 = view_125 = permute_76 = None
        view_126: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_31, [50, 2, 768]);  addmm_31 = None
        add_62: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_59, view_126);  add_59 = view_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:119 in forward_custom, code: x1.append(x.permute(1, 0, 2))
        permute_77: "f32[2, 50, 768]" = torch.ops.aten.permute.default(add_62, [1, 0, 2])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_34: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
        var_mean_17 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
        getitem_66: "f32[50, 2, 1]" = var_mean_17[0]
        getitem_67: "f32[50, 2, 1]" = var_mean_17[1];  var_mean_17 = None
        add_63: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_17: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_23: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_34, getitem_67);  clone_34 = getitem_67 = None
        mul_55: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
        mul_56: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_105);  mul_55 = primals_105 = None
        add_64: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_106);  mul_56 = primals_106 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_127: "f32[100, 768]" = torch.ops.aten.view.default(add_64, [100, 768]);  add_64 = None
        permute_78: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        addmm_32: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_107, view_127, permute_78);  primals_107 = view_127 = permute_78 = None
        view_128: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_32, [50, 2, 2304]);  addmm_32 = None
        view_129: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_128, [50, 2, 3, 768]);  view_128 = None
        unsqueeze_14: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_129, 0);  view_129 = None
        permute_79: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_8: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_79, -2);  permute_79 = None
        clone_35: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_35, 0, 0)
        select_25: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_35, 0, 1)
        select_26: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_35, 0, 2);  clone_35 = None
        view_130: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_24, [50, 24, 64]);  select_24 = None
        permute_80: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_130, [1, 0, 2]);  view_130 = None
        view_131: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_25, [50, 24, 64]);  select_25 = None
        permute_81: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_131, [1, 0, 2]);  view_131 = None
        view_132: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_26, [50, 24, 64]);  select_26 = None
        permute_82: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_132, [1, 0, 2]);  view_132 = None
        view_133: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_80, [2, 12, 50, 64]);  permute_80 = None
        view_134: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_81, [2, 12, 50, 64]);  permute_81 = None
        view_135: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_82, [2, 12, 50, 64]);  permute_82 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_133, view_134, view_135, None, False);  view_133 = view_134 = view_135 = None
        getitem_68: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_83: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_68, [2, 0, 1, 3]);  getitem_68 = None
        clone_36: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_136: "f32[100, 768]" = torch.ops.aten.view.default(clone_36, [100, 768]);  clone_36 = None
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        addmm_33: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_110, view_136, permute_84);  primals_110 = view_136 = permute_84 = None
        view_137: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_33, [50, 2, 768]);  addmm_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_65: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_62, view_137);  add_62 = view_137 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_37: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
        var_mean_18 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
        getitem_72: "f32[50, 2, 1]" = var_mean_18[0]
        getitem_73: "f32[50, 2, 1]" = var_mean_18[1];  var_mean_18 = None
        add_66: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_24: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_37, getitem_73);  clone_37 = getitem_73 = None
        mul_57: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_18);  sub_24 = rsqrt_18 = None
        mul_58: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_111);  mul_57 = primals_111 = None
        add_67: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_112);  mul_58 = primals_112 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_138: "f32[100, 768]" = torch.ops.aten.view.default(add_67, [100, 768]);  add_67 = None
        permute_85: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        addmm_34: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_114, view_138, permute_85);  primals_114 = view_138 = permute_85 = None
        view_139: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_34, [50, 2, 3072]);  addmm_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_59: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_139, 1.702)
        sigmoid_8: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_59);  mul_59 = None
        mul_60: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_139, sigmoid_8);  view_139 = sigmoid_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_140: "f32[100, 3072]" = torch.ops.aten.view.default(mul_60, [100, 3072]);  mul_60 = None
        permute_86: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        addmm_35: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_116, view_140, permute_86);  primals_116 = view_140 = permute_86 = None
        view_141: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_35, [50, 2, 768]);  addmm_35 = None
        add_68: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_65, view_141);  add_65 = view_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_38: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
        var_mean_19 = torch.ops.aten.var_mean.correction(clone_38, [2], correction = 0, keepdim = True)
        getitem_74: "f32[50, 2, 1]" = var_mean_19[0]
        getitem_75: "f32[50, 2, 1]" = var_mean_19[1];  var_mean_19 = None
        add_69: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_19: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_25: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_38, getitem_75);  clone_38 = getitem_75 = None
        mul_61: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_19);  sub_25 = rsqrt_19 = None
        mul_62: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_61, primals_117);  mul_61 = primals_117 = None
        add_70: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_62, primals_118);  mul_62 = primals_118 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_142: "f32[100, 768]" = torch.ops.aten.view.default(add_70, [100, 768]);  add_70 = None
        permute_87: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        addmm_36: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_119, view_142, permute_87);  primals_119 = view_142 = permute_87 = None
        view_143: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_36, [50, 2, 2304]);  addmm_36 = None
        view_144: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_143, [50, 2, 3, 768]);  view_143 = None
        unsqueeze_15: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_144, 0);  view_144 = None
        permute_88: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_9: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_88, -2);  permute_88 = None
        clone_39: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_39, 0, 0)
        select_28: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_39, 0, 1)
        select_29: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_39, 0, 2);  clone_39 = None
        view_145: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_27, [50, 24, 64]);  select_27 = None
        permute_89: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_145, [1, 0, 2]);  view_145 = None
        view_146: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_28, [50, 24, 64]);  select_28 = None
        permute_90: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_146, [1, 0, 2]);  view_146 = None
        view_147: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_29, [50, 24, 64]);  select_29 = None
        permute_91: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_147, [1, 0, 2]);  view_147 = None
        view_148: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_89, [2, 12, 50, 64]);  permute_89 = None
        view_149: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_90, [2, 12, 50, 64]);  permute_90 = None
        view_150: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_91, [2, 12, 50, 64]);  permute_91 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_148, view_149, view_150, None, False);  view_148 = view_149 = view_150 = None
        getitem_76: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_92: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_76, [2, 0, 1, 3]);  getitem_76 = None
        clone_40: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_151: "f32[100, 768]" = torch.ops.aten.view.default(clone_40, [100, 768]);  clone_40 = None
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        addmm_37: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_122, view_151, permute_93);  primals_122 = view_151 = permute_93 = None
        view_152: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_37, [50, 2, 768]);  addmm_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_71: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_68, view_152);  add_68 = view_152 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_41: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_71, memory_format = torch.contiguous_format)
        var_mean_20 = torch.ops.aten.var_mean.correction(clone_41, [2], correction = 0, keepdim = True)
        getitem_80: "f32[50, 2, 1]" = var_mean_20[0]
        getitem_81: "f32[50, 2, 1]" = var_mean_20[1];  var_mean_20 = None
        add_72: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_41, getitem_81);  clone_41 = getitem_81 = None
        mul_63: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_20);  sub_26 = rsqrt_20 = None
        mul_64: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_123);  mul_63 = primals_123 = None
        add_73: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_124);  mul_64 = primals_124 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_153: "f32[100, 768]" = torch.ops.aten.view.default(add_73, [100, 768]);  add_73 = None
        permute_94: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        addmm_38: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_126, view_153, permute_94);  primals_126 = view_153 = permute_94 = None
        view_154: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_38, [50, 2, 3072]);  addmm_38 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_65: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_154, 1.702)
        sigmoid_9: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_65);  mul_65 = None
        mul_66: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_154, sigmoid_9);  view_154 = sigmoid_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_155: "f32[100, 3072]" = torch.ops.aten.view.default(mul_66, [100, 3072]);  mul_66 = None
        permute_95: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        addmm_39: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_128, view_155, permute_95);  primals_128 = view_155 = permute_95 = None
        view_156: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_39, [50, 2, 768]);  addmm_39 = None
        add_74: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_71, view_156);  add_71 = view_156 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_42: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_74, memory_format = torch.contiguous_format)
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
        getitem_82: "f32[50, 2, 1]" = var_mean_21[0]
        getitem_83: "f32[50, 2, 1]" = var_mean_21[1];  var_mean_21 = None
        add_75: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_21: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_27: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_83);  clone_42 = getitem_83 = None
        mul_67: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_21);  sub_27 = rsqrt_21 = None
        mul_68: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_129);  mul_67 = primals_129 = None
        add_76: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_130);  mul_68 = primals_130 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_157: "f32[100, 768]" = torch.ops.aten.view.default(add_76, [100, 768]);  add_76 = None
        permute_96: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        addmm_40: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_131, view_157, permute_96);  primals_131 = view_157 = permute_96 = None
        view_158: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_40, [50, 2, 2304]);  addmm_40 = None
        view_159: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_158, [50, 2, 3, 768]);  view_158 = None
        unsqueeze_16: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_159, 0);  view_159 = None
        permute_97: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_10: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_97, -2);  permute_97 = None
        clone_43: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_43, 0, 0)
        select_31: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_43, 0, 1)
        select_32: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_43, 0, 2);  clone_43 = None
        view_160: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_30, [50, 24, 64]);  select_30 = None
        permute_98: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_160, [1, 0, 2]);  view_160 = None
        view_161: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_31, [50, 24, 64]);  select_31 = None
        permute_99: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_161, [1, 0, 2]);  view_161 = None
        view_162: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_32, [50, 24, 64]);  select_32 = None
        permute_100: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_162, [1, 0, 2]);  view_162 = None
        view_163: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_98, [2, 12, 50, 64]);  permute_98 = None
        view_164: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_99, [2, 12, 50, 64]);  permute_99 = None
        view_165: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_100, [2, 12, 50, 64]);  permute_100 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_163, view_164, view_165, None, False);  view_163 = view_164 = view_165 = None
        getitem_84: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_101: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_84, [2, 0, 1, 3]);  getitem_84 = None
        clone_44: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_166: "f32[100, 768]" = torch.ops.aten.view.default(clone_44, [100, 768]);  clone_44 = None
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        addmm_41: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_134, view_166, permute_102);  primals_134 = view_166 = permute_102 = None
        view_167: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_41, [50, 2, 768]);  addmm_41 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_77: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_74, view_167);  add_74 = view_167 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_45: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
        var_mean_22 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
        getitem_88: "f32[50, 2, 1]" = var_mean_22[0]
        getitem_89: "f32[50, 2, 1]" = var_mean_22[1];  var_mean_22 = None
        add_78: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_28: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_45, getitem_89);  clone_45 = getitem_89 = None
        mul_69: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = rsqrt_22 = None
        mul_70: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_69, primals_135);  mul_69 = primals_135 = None
        add_79: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_136);  mul_70 = primals_136 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_168: "f32[100, 768]" = torch.ops.aten.view.default(add_79, [100, 768]);  add_79 = None
        permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        addmm_42: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_138, view_168, permute_103);  primals_138 = view_168 = permute_103 = None
        view_169: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_42, [50, 2, 3072]);  addmm_42 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_71: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_169, 1.702)
        sigmoid_10: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_71);  mul_71 = None
        mul_72: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_169, sigmoid_10);  view_169 = sigmoid_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_170: "f32[100, 3072]" = torch.ops.aten.view.default(mul_72, [100, 3072]);  mul_72 = None
        permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        addmm_43: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_140, view_170, permute_104);  primals_140 = view_170 = permute_104 = None
        view_171: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_43, [50, 2, 768]);  addmm_43 = None
        add_80: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_77, view_171);  add_77 = view_171 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_46: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
        var_mean_23 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
        getitem_90: "f32[50, 2, 1]" = var_mean_23[0]
        getitem_91: "f32[50, 2, 1]" = var_mean_23[1];  var_mean_23 = None
        add_81: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_23: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_29: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_46, getitem_91);  clone_46 = getitem_91 = None
        mul_73: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = rsqrt_23 = None
        mul_74: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_141);  mul_73 = primals_141 = None
        add_82: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_142);  mul_74 = primals_142 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:188 in attention, code: return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        view_172: "f32[100, 768]" = torch.ops.aten.view.default(add_82, [100, 768]);  add_82 = None
        permute_105: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        addmm_44: "f32[100, 2304]" = torch.ops.aten.addmm.default(primals_143, view_172, permute_105);  primals_143 = view_172 = permute_105 = None
        view_173: "f32[50, 2, 2304]" = torch.ops.aten.view.default(addmm_44, [50, 2, 2304]);  addmm_44 = None
        view_174: "f32[50, 2, 3, 768]" = torch.ops.aten.view.default(view_173, [50, 2, 3, 768]);  view_173 = None
        unsqueeze_17: "f32[1, 50, 2, 3, 768]" = torch.ops.aten.unsqueeze.default(view_174, 0);  view_174 = None
        permute_106: "f32[3, 50, 2, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_11: "f32[3, 50, 2, 768]" = torch.ops.aten.squeeze.dim(permute_106, -2);  permute_106 = None
        clone_47: "f32[3, 50, 2, 768]" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_47, 0, 0)
        select_34: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_47, 0, 1)
        select_35: "f32[50, 2, 768]" = torch.ops.aten.select.int(clone_47, 0, 2);  clone_47 = None
        view_175: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_33, [50, 24, 64]);  select_33 = None
        permute_107: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_175, [1, 0, 2]);  view_175 = None
        view_176: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_34, [50, 24, 64]);  select_34 = None
        permute_108: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_176, [1, 0, 2]);  view_176 = None
        view_177: "f32[50, 24, 64]" = torch.ops.aten.view.default(select_35, [50, 24, 64]);  select_35 = None
        permute_109: "f32[24, 50, 64]" = torch.ops.aten.permute.default(view_177, [1, 0, 2]);  view_177 = None
        view_178: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_107, [2, 12, 50, 64]);  permute_107 = None
        view_179: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_108, [2, 12, 50, 64]);  permute_108 = None
        view_180: "f32[2, 12, 50, 64]" = torch.ops.aten.view.default(permute_109, [2, 12, 50, 64]);  permute_109 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_178, view_179, view_180, None, False);  view_178 = view_179 = view_180 = None
        getitem_92: "f32[2, 12, 50, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_110: "f32[50, 2, 12, 64]" = torch.ops.aten.permute.default(getitem_92, [2, 0, 1, 3]);  getitem_92 = None
        clone_48: "f32[50, 2, 12, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_181: "f32[100, 768]" = torch.ops.aten.view.default(clone_48, [100, 768]);  clone_48 = None
        permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        addmm_45: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_146, view_181, permute_111);  primals_146 = view_181 = permute_111 = None
        view_182: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_45, [50, 2, 768]);  addmm_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:191 in forward, code: x = x + self.attention(self.ln_1(x))
        add_83: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_80, view_182);  add_80 = view_182 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_49: "f32[50, 2, 768]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
        var_mean_24 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
        getitem_96: "f32[50, 2, 1]" = var_mean_24[0]
        getitem_97: "f32[50, 2, 1]" = var_mean_24[1];  var_mean_24 = None
        add_84: "f32[50, 2, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[50, 2, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30: "f32[50, 2, 768]" = torch.ops.aten.sub.Tensor(clone_49, getitem_97);  clone_49 = getitem_97 = None
        mul_75: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_24);  sub_30 = rsqrt_24 = None
        mul_76: "f32[50, 2, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_147);  mul_75 = primals_147 = None
        add_85: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_148);  mul_76 = primals_148 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_183: "f32[100, 768]" = torch.ops.aten.view.default(add_85, [100, 768]);  add_85 = None
        permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        addmm_46: "f32[100, 3072]" = torch.ops.aten.addmm.default(primals_150, view_183, permute_112);  primals_150 = view_183 = permute_112 = None
        view_184: "f32[50, 2, 3072]" = torch.ops.aten.view.default(addmm_46, [50, 2, 3072]);  addmm_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:169 in forward, code: return x * torch.sigmoid(1.702 * x)
        mul_77: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_184, 1.702)
        sigmoid_11: "f32[50, 2, 3072]" = torch.ops.aten.sigmoid.default(mul_77);  mul_77 = None
        mul_78: "f32[50, 2, 3072]" = torch.ops.aten.mul.Tensor(view_184, sigmoid_11);  view_184 = sigmoid_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:192 in forward, code: x = x + self.mlp(self.ln_2(x))
        view_185: "f32[100, 3072]" = torch.ops.aten.view.default(mul_78, [100, 3072]);  mul_78 = None
        permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        addmm_47: "f32[100, 768]" = torch.ops.aten.addmm.default(primals_152, view_185, permute_113);  primals_152 = view_185 = permute_113 = None
        view_186: "f32[50, 2, 768]" = torch.ops.aten.view.default(addmm_47, [50, 2, 768]);  addmm_47 = None
        add_86: "f32[50, 2, 768]" = torch.ops.aten.add.Tensor(add_83, view_186);  add_83 = view_186 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:119 in forward_custom, code: x1.append(x.permute(1, 0, 2))
        permute_114: "f32[2, 50, 768]" = torch.ops.aten.permute.default(add_86, [1, 0, 2]);  add_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:121 in forward_custom, code: x = self.model.ln_post(x1[-1][:, 0, :])
        select_36: "f32[2, 768]" = torch.ops.aten.select.int(permute_114, 1, 0);  permute_114 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/CLIP/clip/model.py:163 in forward, code: ret = super().forward(x.type(torch.float32))
        clone_50: "f32[2, 768]" = torch.ops.aten.clone.default(select_36, memory_format = torch.contiguous_format);  select_36 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_50, [1], correction = 0, keepdim = True)
        getitem_98: "f32[2, 1]" = var_mean_25[0]
        getitem_99: "f32[2, 1]" = var_mean_25[1];  var_mean_25 = None
        add_87: "f32[2, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_25: "f32[2, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_31: "f32[2, 768]" = torch.ops.aten.sub.Tensor(clone_50, getitem_99);  clone_50 = getitem_99 = None
        mul_79: "f32[2, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_25);  sub_31 = rsqrt_25 = None
        mul_80: "f32[2, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_153);  mul_79 = primals_153 = None
        add_88: "f32[2, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_154);  mul_80 = primals_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:123 in forward_custom, code: x = x @ self.model.proj
        mm: "f32[2, 512]" = torch.ops.aten.mm.default(add_88, primals_155);  add_88 = primals_155 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:134 in __call__, code: x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
        slice_6: "f32[2, 49, 768]" = torch.ops.aten.slice.Tensor(permute_40, 1, 1, 9223372036854775807);  permute_40 = None
        permute_115: "f32[2, 768, 49]" = torch.ops.aten.permute.default(slice_6, [0, 2, 1]);  slice_6 = None
        view_187: "f32[2, 768, 7, 7]" = torch.ops.aten.view.default(permute_115, [2, 768, 7, 7]);  permute_115 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cvmodel.py:135 in __call__, code: x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
        slice_9: "f32[2, 49, 768]" = torch.ops.aten.slice.Tensor(permute_77, 1, 1, 9223372036854775807);  permute_77 = None
        permute_116: "f32[2, 768, 49]" = torch.ops.aten.permute.default(slice_9, [0, 2, 1]);  slice_9 = None
        view_188: "f32[2, 768, 7, 7]" = torch.ops.aten.view.default(permute_116, [2, 768, 7, 7]);  permute_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_189: "f32[256, 6912]" = torch.ops.aten.view.default(primals_156, [256, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_117: "f32[6912, 256]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mul_81: "f32[6912, 256]" = torch.ops.aten.mul.Tensor(permute_117, primals_157);  permute_117 = None
        sum_1: "f32[6912]" = torch.ops.aten.sum.dim_IntList(mul_81, [1]);  mul_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_1: "f32[6912]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 2.0)
        sum_2: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_1, [0], True);  pow_1 = None
        pow_2: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min_4: "f32[1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12);  pow_2 = None
        expand_7: "f32[6912]" = torch.ops.aten.expand.default(clamp_min_4, [6912]);  clamp_min_4 = None
        div_1: "f32[6912]" = torch.ops.aten.div.Tensor(sum_1, expand_7);  sum_1 = expand_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_82: "f32[256, 6912]" = torch.ops.aten.mul.Tensor(view_189, div_1);  view_189 = None
        sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_82, [1]);  mul_82 = None
        pow_3: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 2.0)
        sum_4: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_3, [0], True);  pow_3 = None
        pow_4: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        clamp_min_5: "f32[1]" = torch.ops.aten.clamp_min.default(pow_4, 1e-12);  pow_4 = None
        expand_9: "f32[256]" = torch.ops.aten.expand.default(clamp_min_5, [256]);  clamp_min_5 = None
        div_2: "f32[256]" = torch.ops.aten.div.Tensor(sum_3, expand_9);  expand_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_84: "f32[256]" = torch.ops.aten.mul.Tensor(div_2, sum_3);  sum_3 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(mul_84);  mul_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_3: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(primals_156, sum_6)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        convolution_1: "f32[2, 256, 7, 7]" = torch.ops.aten.convolution.default(view_187, div_3, primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_159 = None
        gt: "b8[2, 256, 7, 7]" = torch.ops.aten.gt.Scalar(convolution_1, 0)
        mul_85: "f32[2, 256, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_1, 0.2)
        where: "f32[2, 256, 7, 7]" = torch.ops.aten.where.self(gt, convolution_1, mul_85);  gt = convolution_1 = mul_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[2, 256, 9, 9]" = torch.ops.aten.constant_pad_nd.default(where, [1, 1, 1, 1], 0.0)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/blurpool.py:53 in forward, code: return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        convolution_2: "f32[2, 256, 6, 6]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 256)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_190: "f32[1, 256]" = torch.ops.aten.view.default(primals_161, [1, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_118: "f32[256, 1]" = torch.ops.aten.permute.default(view_190, [1, 0])
        mul_86: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_118, primals_162);  permute_118 = None
        sum_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_86, [1]);  mul_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_5: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 2.0)
        sum_8: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_5, [0], True);  pow_5 = None
        pow_6: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None
        clamp_min_6: "f32[1]" = torch.ops.aten.clamp_min.default(pow_6, 1e-12);  pow_6 = None
        expand_11: "f32[256]" = torch.ops.aten.expand.default(clamp_min_6, [256])
        div_4: "f32[256]" = torch.ops.aten.div.Tensor(sum_7, expand_11);  sum_7 = expand_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_87: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_190, div_4);  view_190 = None
        sum_9: "f32[1]" = torch.ops.aten.sum.dim_IntList(mul_87, [1]);  mul_87 = None
        pow_7: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 2.0)
        sum_10: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_7, [0], True);  pow_7 = None
        pow_8: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
        clamp_min_7: "f32[1]" = torch.ops.aten.clamp_min.default(pow_8, 1e-12);  pow_8 = None
        expand_13: "f32[1]" = torch.ops.aten.expand.default(clamp_min_7, [1]);  clamp_min_7 = None
        div_5: "f32[1]" = torch.ops.aten.div.Tensor(sum_9, expand_13);  expand_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_89: "f32[1]" = torch.ops.aten.mul.Tensor(div_5, sum_9)
        sum_12: "f32[]" = torch.ops.aten.sum.default(mul_89);  mul_89 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_6: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(primals_161, sum_12);  sum_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        convolution_3: "f32[2, 1, 3, 3]" = torch.ops.aten.convolution.default(convolution_2, div_6, primals_164, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
        squeeze_12: "f32[2, 3, 3]" = torch.ops.aten.squeeze.dim(convolution_3, 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_191: "f32[256, 6912]" = torch.ops.aten.view.default(primals_165, [256, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_119: "f32[6912, 256]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mul_90: "f32[6912, 256]" = torch.ops.aten.mul.Tensor(permute_119, primals_166);  permute_119 = None
        sum_13: "f32[6912]" = torch.ops.aten.sum.dim_IntList(mul_90, [1]);  mul_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_9: "f32[6912]" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 2.0)
        sum_14: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_9, [0], True);  pow_9 = None
        pow_10: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None
        clamp_min_8: "f32[1]" = torch.ops.aten.clamp_min.default(pow_10, 1e-12);  pow_10 = None
        expand_15: "f32[6912]" = torch.ops.aten.expand.default(clamp_min_8, [6912]);  clamp_min_8 = None
        div_7: "f32[6912]" = torch.ops.aten.div.Tensor(sum_13, expand_15);  sum_13 = expand_15 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_91: "f32[256, 6912]" = torch.ops.aten.mul.Tensor(view_191, div_7);  view_191 = None
        sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_91, [1]);  mul_91 = None
        pow_11: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_15, 2.0)
        sum_16: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_11, [0], True);  pow_11 = None
        pow_12: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_16, 0.5);  sum_16 = None
        clamp_min_9: "f32[1]" = torch.ops.aten.clamp_min.default(pow_12, 1e-12);  pow_12 = None
        expand_17: "f32[256]" = torch.ops.aten.expand.default(clamp_min_9, [256]);  clamp_min_9 = None
        div_8: "f32[256]" = torch.ops.aten.div.Tensor(sum_15, expand_17);  expand_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_93: "f32[256]" = torch.ops.aten.mul.Tensor(div_8, sum_15);  sum_15 = None
        sum_18: "f32[]" = torch.ops.aten.sum.default(mul_93);  mul_93 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_9: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(primals_165, sum_18)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        convolution_4: "f32[2, 256, 7, 7]" = torch.ops.aten.convolution.default(view_188, div_9, primals_168, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_168 = None
        gt_1: "b8[2, 256, 7, 7]" = torch.ops.aten.gt.Scalar(convolution_4, 0)
        mul_94: "f32[2, 256, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_4, 0.2)
        where_1: "f32[2, 256, 7, 7]" = torch.ops.aten.where.self(gt_1, convolution_4, mul_94);  gt_1 = convolution_4 = mul_94 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[2, 256, 9, 9]" = torch.ops.aten.constant_pad_nd.default(where_1, [1, 1, 1, 1], 0.0)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/blurpool.py:53 in forward, code: return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        convolution_5: "f32[2, 256, 6, 6]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 256)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_192: "f32[1, 256]" = torch.ops.aten.view.default(primals_170, [1, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_120: "f32[256, 1]" = torch.ops.aten.permute.default(view_192, [1, 0])
        mul_95: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_120, primals_171);  permute_120 = None
        sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_95, [1]);  mul_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_13: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_19, 2.0)
        sum_20: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_13, [0], True);  pow_13 = None
        pow_14: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_20, 0.5);  sum_20 = None
        clamp_min_10: "f32[1]" = torch.ops.aten.clamp_min.default(pow_14, 1e-12);  pow_14 = None
        expand_19: "f32[256]" = torch.ops.aten.expand.default(clamp_min_10, [256])
        div_10: "f32[256]" = torch.ops.aten.div.Tensor(sum_19, expand_19);  sum_19 = expand_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_96: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_192, div_10);  view_192 = None
        sum_21: "f32[1]" = torch.ops.aten.sum.dim_IntList(mul_96, [1]);  mul_96 = None
        pow_15: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_21, 2.0)
        sum_22: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_15, [0], True);  pow_15 = None
        pow_16: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
        clamp_min_11: "f32[1]" = torch.ops.aten.clamp_min.default(pow_16, 1e-12);  pow_16 = None
        expand_21: "f32[1]" = torch.ops.aten.expand.default(clamp_min_11, [1]);  clamp_min_11 = None
        div_11: "f32[1]" = torch.ops.aten.div.Tensor(sum_21, expand_21);  expand_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_98: "f32[1]" = torch.ops.aten.mul.Tensor(div_11, sum_21)
        sum_24: "f32[]" = torch.ops.aten.sum.default(mul_98);  mul_98 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_12: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(primals_170, sum_24);  sum_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        convolution_6: "f32[2, 1, 3, 3]" = torch.ops.aten.convolution.default(convolution_5, div_12, primals_173, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
        squeeze_13: "f32[2, 3, 3]" = torch.ops.aten.squeeze.dim(convolution_6, 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_193: "f32[256, 512]" = torch.ops.aten.view.default(primals_174, [256, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_121: "f32[512, 256]" = torch.ops.aten.permute.default(view_193, [1, 0])
        mul_99: "f32[512, 256]" = torch.ops.aten.mul.Tensor(permute_121, primals_175);  permute_121 = None
        sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_99, [1]);  mul_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_17: "f32[512]" = torch.ops.aten.pow.Tensor_Scalar(sum_25, 2.0)
        sum_26: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_17, [0], True);  pow_17 = None
        pow_18: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_26, 0.5);  sum_26 = None
        clamp_min_12: "f32[1]" = torch.ops.aten.clamp_min.default(pow_18, 1e-12);  pow_18 = None
        expand_23: "f32[512]" = torch.ops.aten.expand.default(clamp_min_12, [512]);  clamp_min_12 = None
        div_13: "f32[512]" = torch.ops.aten.div.Tensor(sum_25, expand_23);  sum_25 = expand_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_100: "f32[256, 512]" = torch.ops.aten.mul.Tensor(view_193, div_13);  view_193 = None
        sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_100, [1]);  mul_100 = None
        pow_19: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_27, 2.0)
        sum_28: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_19, [0], True);  pow_19 = None
        pow_20: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_28, 0.5);  sum_28 = None
        clamp_min_13: "f32[1]" = torch.ops.aten.clamp_min.default(pow_20, 1e-12);  pow_20 = None
        expand_25: "f32[256]" = torch.ops.aten.expand.default(clamp_min_13, [256]);  clamp_min_13 = None
        div_14: "f32[256]" = torch.ops.aten.div.Tensor(sum_27, expand_25);  expand_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(div_14, sum_27);  sum_27 = None
        sum_30: "f32[]" = torch.ops.aten.sum.default(mul_102);  mul_102 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_15: "f32[256, 512]" = torch.ops.aten.div.Tensor(primals_174, sum_30)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:37 in forward, code: h = self.decoder[-1](x[-1].float())
        permute_122: "f32[512, 256]" = torch.ops.aten.permute.default(div_15, [1, 0])
        addmm_48: "f32[2, 256]" = torch.ops.aten.addmm.default(primals_177, mm, permute_122);  primals_177 = permute_122 = None
        gt_2: "b8[2, 256]" = torch.ops.aten.gt.Scalar(addmm_48, 0)
        mul_103: "f32[2, 256]" = torch.ops.aten.mul.Tensor(addmm_48, 0.2)
        where_2: "f32[2, 256]" = torch.ops.aten.where.self(gt_2, addmm_48, mul_103);  gt_2 = addmm_48 = mul_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_194: "f32[1, 256]" = torch.ops.aten.view.default(primals_178, [1, -1])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_123: "f32[256, 1]" = torch.ops.aten.permute.default(view_194, [1, 0])
        mul_104: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_123, primals_179);  permute_123 = None
        sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_104, [1]);  mul_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        pow_21: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sum_31, 2.0)
        sum_32: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_21, [0], True);  pow_21 = None
        pow_22: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_32, 0.5);  sum_32 = None
        clamp_min_14: "f32[1]" = torch.ops.aten.clamp_min.default(pow_22, 1e-12);  pow_22 = None
        expand_27: "f32[256]" = torch.ops.aten.expand.default(clamp_min_14, [256])
        div_16: "f32[256]" = torch.ops.aten.div.Tensor(sum_31, expand_27);  sum_31 = expand_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        mul_105: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_194, div_16);  view_194 = None
        sum_33: "f32[1]" = torch.ops.aten.sum.dim_IntList(mul_105, [1]);  mul_105 = None
        pow_23: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_33, 2.0)
        sum_34: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_23, [0], True);  pow_23 = None
        pow_24: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
        clamp_min_15: "f32[1]" = torch.ops.aten.clamp_min.default(pow_24, 1e-12);  pow_24 = None
        expand_29: "f32[1]" = torch.ops.aten.expand.default(clamp_min_15, [1]);  clamp_min_15 = None
        div_17: "f32[1]" = torch.ops.aten.div.Tensor(sum_33, expand_29);  expand_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_107: "f32[1]" = torch.ops.aten.mul.Tensor(div_17, sum_33)
        sum_36: "f32[]" = torch.ops.aten.sum.default(mul_107);  mul_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_18: "f32[1, 256]" = torch.ops.aten.div.Tensor(primals_178, sum_36);  sum_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:38 in forward, code: out = self.out(h)
        permute_124: "f32[256, 1]" = torch.ops.aten.permute.default(div_18, [1, 0])
        addmm_49: "f32[2, 1]" = torch.ops.aten.addmm.default(primals_181, where_2, permute_124);  primals_181 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:43 in forward, code: loss_ = self.lossfn(each, target_)
        full_default_4: "f32[2, 3, 3]" = torch.ops.aten.full.default([2, 3, 3], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_109: "f32[2, 3, 3]" = torch.ops.aten.mul.Tensor(full_default_4, squeeze_12)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum: "f32[2, 3, 3]" = torch.ops.aten.minimum.default(full_default_5, squeeze_12)
        abs_1: "f32[2, 3, 3]" = torch.ops.aten.abs.default(squeeze_12);  squeeze_12 = None
        neg: "f32[2, 3, 3]" = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        exp: "f32[2, 3, 3]" = torch.ops.aten.exp.default(neg);  neg = None
        log1p: "f32[2, 3, 3]" = torch.ops.aten.log1p.default(exp);  exp = None
        sub_33: "f32[2, 3, 3]" = torch.ops.aten.sub.Tensor(minimum, log1p);  minimum = log1p = None
        sub_34: "f32[2, 3, 3]" = torch.ops.aten.sub.Tensor(mul_109, sub_33);  mul_109 = sub_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:45 in forward, code: loss_ = loss_.mean([1, 2]).reshape(-1, 1)
        mean_2: "f32[2]" = torch.ops.aten.mean.dim(sub_34, [1, 2]);  sub_34 = None
        view_195: "f32[2, 1]" = torch.ops.aten.view.default(mean_2, [-1, 1]);  mean_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:46 in forward, code: loss += loss_
        add_89: "f32[2, 1]" = torch.ops.aten.add.Tensor(view_195, 0);  view_195 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:43 in forward, code: loss_ = self.lossfn(each, target_)
        mul_110: "f32[2, 3, 3]" = torch.ops.aten.mul.Tensor(full_default_4, squeeze_13);  full_default_4 = None
        minimum_1: "f32[2, 3, 3]" = torch.ops.aten.minimum.default(full_default_5, squeeze_13)
        abs_2: "f32[2, 3, 3]" = torch.ops.aten.abs.default(squeeze_13);  squeeze_13 = None
        neg_1: "f32[2, 3, 3]" = torch.ops.aten.neg.default(abs_2);  abs_2 = None
        exp_1: "f32[2, 3, 3]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        log1p_1: "f32[2, 3, 3]" = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
        sub_36: "f32[2, 3, 3]" = torch.ops.aten.sub.Tensor(minimum_1, log1p_1);  minimum_1 = log1p_1 = None
        sub_37: "f32[2, 3, 3]" = torch.ops.aten.sub.Tensor(mul_110, sub_36);  mul_110 = sub_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:45 in forward, code: loss_ = loss_.mean([1, 2]).reshape(-1, 1)
        mean_3: "f32[2]" = torch.ops.aten.mean.dim(sub_37, [1, 2]);  sub_37 = None
        view_196: "f32[2, 1]" = torch.ops.aten.view.default(mean_3, [-1, 1]);  mean_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:46 in forward, code: loss += loss_
        add_90: "f32[2, 1]" = torch.ops.aten.add.Tensor(add_89, view_196);  add_89 = view_196 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:43 in forward, code: loss_ = self.lossfn(each, target_)
        full_default_10: "f32[2, 1]" = torch.ops.aten.full.default([2, 1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_111: "f32[2, 1]" = torch.ops.aten.mul.Tensor(full_default_10, addmm_49);  full_default_10 = None
        minimum_2: "f32[2, 1]" = torch.ops.aten.minimum.default(full_default_5, addmm_49);  full_default_5 = None
        abs_3: "f32[2, 1]" = torch.ops.aten.abs.default(addmm_49)
        neg_2: "f32[2, 1]" = torch.ops.aten.neg.default(abs_3);  abs_3 = None
        exp_2: "f32[2, 1]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        log1p_2: "f32[2, 1]" = torch.ops.aten.log1p.default(exp_2);  exp_2 = None
        sub_39: "f32[2, 1]" = torch.ops.aten.sub.Tensor(minimum_2, log1p_2);  minimum_2 = log1p_2 = None
        sub_40: "f32[2, 1]" = torch.ops.aten.sub.Tensor(mul_111, sub_39);  mul_111 = sub_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:46 in forward, code: loss += loss_
        add_91: "f32[2, 1]" = torch.ops.aten.add.Tensor(add_90, sub_40);  add_90 = sub_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:105 in forward, code: loss += loss_
        add_92: "f32[2, 1]" = torch.ops.aten.add.Tensor(add_91, 0);  add_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:38 in forward, code: out = self.out(h)
        permute_125: "f32[1, 256]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        gt_4: "b8[2, 256, 7, 7]" = torch.ops.aten.gt.Scalar(where_1, 0);  where_1 = None
        gt_5: "b8[2, 256, 7, 7]" = torch.ops.aten.gt.Scalar(where, 0);  where = None
        
        # No stacktrace found for following nodes
        copy_: "f32[256]" = torch.ops.aten.copy_.default(primals_157, div_2);  primals_157 = copy_ = None
        copy__1: "f32[6912]" = torch.ops.aten.copy_.default(primals_158, div_1);  primals_158 = copy__1 = None
        copy__2: "f32[1]" = torch.ops.aten.copy_.default(primals_162, div_5);  div_5 = copy__2 = None
        copy__3: "f32[256]" = torch.ops.aten.copy_.default(primals_163, div_4);  primals_163 = div_4 = copy__3 = None
        copy__4: "f32[256]" = torch.ops.aten.copy_.default(primals_166, div_8);  primals_166 = copy__4 = None
        copy__5: "f32[6912]" = torch.ops.aten.copy_.default(primals_167, div_7);  primals_167 = copy__5 = None
        copy__6: "f32[1]" = torch.ops.aten.copy_.default(primals_171, div_11);  div_11 = copy__6 = None
        copy__7: "f32[256]" = torch.ops.aten.copy_.default(primals_172, div_10);  primals_172 = div_10 = copy__7 = None
        copy__8: "f32[256]" = torch.ops.aten.copy_.default(primals_175, div_14);  primals_175 = copy__8 = None
        copy__9: "f32[512]" = torch.ops.aten.copy_.default(primals_176, div_13);  primals_176 = copy__9 = None
        copy__10: "f32[1]" = torch.ops.aten.copy_.default(primals_179, div_17);  div_17 = copy__10 = None
        copy__11: "f32[256]" = torch.ops.aten.copy_.default(primals_180, div_16);  primals_180 = div_16 = copy__11 = None
        return (add_92, div_3, div_6, div_9, div_12, div_15, div_18, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, mm, view_187, view_188, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, gt_4, gt_5)
        