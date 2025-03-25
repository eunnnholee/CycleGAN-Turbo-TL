
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch._decomp', 'torch._prims', 'torch.distributions', 'torch._refs', 'torch.testing'}
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_level', 'repro_after', 'constant_functions', 'skipfiles_inline_module_allowlist'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu118
# torch cuda version: 11.8
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Sep_21_10:33:58_PDT_2022 
# Cuda compilation tools, release 11.8, V11.8.89 
# Build cuda_11.8.r11.8/compiler.31833905_0 

# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe MIG 3g.40gb : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181):
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(7, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        sub = torch.ops.aten.sub.Tensor(inductor_random_default_2, 0.5);  inductor_random_default_2 = None
        add = torch.ops.aten.add.Tensor(primals_1, sub);  primals_1 = sub = None
        mean = torch.ops.aten.mean.dim(add, [1], True)
        sub_1 = torch.ops.aten.sub.Tensor(add, mean);  add = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        mul = torch.ops.aten.mul.Tensor(inductor_random_default_1, 2)
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, mul);  sub_1 = mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, mean);  mul_1 = mean = None
        mean_1 = torch.ops.aten.mean.dim(add_1, [1, 2, 3], True)
        sub_2 = torch.ops.aten.sub.Tensor(add_1, mean_1);  add_1 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default = torch.ops.prims.inductor_random.default([2, 1, 1, 1], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        add_2 = torch.ops.aten.add.Tensor(inductor_random_default, 0.5)
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, add_2);  sub_2 = add_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, mean_1);  mul_2 = mean_1 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_randint_default_3 = torch.ops.prims.inductor_randint.default(-32, 33, [2, 1, 1], inductor_lookup_seed_default_3);  inductor_lookup_seed_default_3 = None
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_randint_default_2 = torch.ops.prims.inductor_randint.default(-32, 33, [2, 1, 1], inductor_lookup_seed_default_4);  inductor_lookup_seed_default_4 = None
        iota = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_1 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view = torch.ops.aten.view.default(iota, [-1, 1, 1]);  iota = None
        expand = torch.ops.aten.expand.default(view, [2, 256, 256])
        view_1 = torch.ops.aten.view.default(iota_1, [1, -1, 1])
        expand_1 = torch.ops.aten.expand.default(view_1, [2, 256, 256]);  view_1 = None
        view_2 = torch.ops.aten.view.default(iota_1, [1, 1, -1]);  iota_1 = None
        expand_2 = torch.ops.aten.expand.default(view_2, [2, 256, 256]);  view_2 = None
        add_4 = torch.ops.aten.add.Tensor(expand_1, inductor_randint_default_3);  expand_1 = inductor_randint_default_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, 1);  add_4 = None
        clamp_min = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 257);  clamp_min = None
        add_6 = torch.ops.aten.add.Tensor(expand_2, inductor_randint_default_2);  expand_2 = inductor_randint_default_2 = None
        add_7 = torch.ops.aten.add.Tensor(add_6, 1);  add_6 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(add_7, 0);  add_7 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 257);  clamp_min_1 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(add_3, [1, 1, 1, 1, 0, 0, 0, 0], 0.0);  add_3 = None
        permute = torch.ops.aten.permute.default(constant_pad_nd, [0, 2, 3, 1]);  constant_pad_nd = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        index = torch.ops.aten.index.Tensor(clone, [expand, clamp_max, clamp_max_1]);  clone = expand = None
        permute_1 = torch.ops.aten.permute.default(index, [0, 3, 1, 2]);  index = None
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_randint_default_1 = torch.ops.prims.inductor_randint.default(0, 257, [2, 1, 1], inductor_lookup_seed_default_5);  inductor_lookup_seed_default_5 = None
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6);  inductor_seeds_default = None
        inductor_randint_default = torch.ops.prims.inductor_randint.default(0, 257, [2, 1, 1], inductor_lookup_seed_default_6);  inductor_lookup_seed_default_6 = None
        iota_4 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand_3 = torch.ops.aten.expand.default(view, [2, 128, 128])
        view_4 = torch.ops.aten.view.default(iota_4, [1, -1, 1])
        expand_4 = torch.ops.aten.expand.default(view_4, [2, 128, 128]);  view_4 = None
        view_5 = torch.ops.aten.view.default(iota_4, [1, 1, -1]);  iota_4 = None
        expand_5 = torch.ops.aten.expand.default(view_5, [2, 128, 128]);  view_5 = None
        add_8 = torch.ops.aten.add.Tensor(expand_4, inductor_randint_default_1);  expand_4 = inductor_randint_default_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, 64);  add_8 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_3, 0);  sub_3 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
        add_9 = torch.ops.aten.add.Tensor(expand_5, inductor_randint_default);  expand_5 = inductor_randint_default = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, 64);  add_9 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_4, 0);  sub_4 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
        full_default = torch.ops.aten.full.default([2, 256, 256], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default, [expand_3, clamp_max_2, clamp_max_3], full_default_1);  full_default = expand_3 = clamp_max_2 = clamp_max_3 = full_default_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(index_put, 1)
        mul_3 = torch.ops.aten.mul.Tensor(permute_1, unsqueeze_1);  permute_1 = unsqueeze_1 = None
        clone_1 = torch.ops.aten.clone.default(mul_3, memory_format = torch.contiguous_format);  mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(clone_1, 0.5);  clone_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_4, 0.5);  mul_4 = None
        _adaptive_avg_pool2d = torch.ops.aten._adaptive_avg_pool2d.default(add_10, [224, 224])
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        device_put = torch.ops.prims.device_put.default(unsqueeze_3, device(type='cuda', index=0));  unsqueeze_3 = None
        sub_5 = torch.ops.aten.sub.Tensor(_adaptive_avg_pool2d, device_put);  _adaptive_avg_pool2d = device_put = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_3, 1);  primals_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        device_put_1 = torch.ops.prims.device_put.default(unsqueeze_5, device(type='cuda', index=0));  unsqueeze_5 = None
        div = torch.ops.aten.div.Tensor(sub_5, device_put_1);  sub_5 = None
        convolution = torch.ops.aten.convolution.default(div, primals_4, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1)
        view_6 = torch.ops.aten.view.default(convolution, [2, 768, -1]);  convolution = None
        permute_2 = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        full_default_2 = torch.ops.aten.full.default([2, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add_11 = torch.ops.aten.add.Tensor(primals_5, full_default_2);  primals_5 = full_default_2 = None
        cat = torch.ops.aten.cat.default([add_11, permute_2], 1);  add_11 = permute_2 = None
        add_12 = torch.ops.aten.add.Tensor(cat, primals_6)
        var_mean = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_13 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_12, getitem_1);  add_12 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, primals_7);  mul_5 = None
        add_14 = torch.ops.aten.add.Tensor(mul_6, primals_8);  mul_6 = None
        permute_3 = torch.ops.aten.permute.default(add_14, [1, 0, 2]);  add_14 = None
        clone_2 = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_7 = torch.ops.aten.sub.Tensor(clone_2, getitem_3);  clone_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, primals_9);  mul_7 = None
        add_16 = torch.ops.aten.add.Tensor(mul_8, primals_10);  mul_8 = primals_10 = None
        view_7 = torch.ops.aten.view.default(add_16, [100, 768]);  add_16 = None
        permute_4 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        addmm = torch.ops.aten.addmm.default(primals_11, view_7, permute_4);  primals_11 = view_7 = None
        view_8 = torch.ops.aten.view.default(addmm, [50, 2, 2304]);  addmm = None
        view_9 = torch.ops.aten.view.default(view_8, [50, 2, 3, 768]);  view_8 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(view_9, 0);  view_9 = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze = torch.ops.aten.squeeze.dim(permute_5, -2);  permute_5 = None
        clone_3 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_3, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_3, 0, 1)
        select_2 = torch.ops.aten.select.int(clone_3, 0, 2);  clone_3 = None
        view_10 = torch.ops.aten.view.default(select, [50, 24, 64]);  select = None
        permute_6 = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        view_11 = torch.ops.aten.view.default(select_1, [50, 24, 64]);  select_1 = None
        permute_7 = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        view_12 = torch.ops.aten.view.default(select_2, [50, 24, 64]);  select_2 = None
        permute_8 = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        view_13 = torch.ops.aten.view.default(permute_6, [2, 12, 50, 64]);  permute_6 = None
        view_14 = torch.ops.aten.view.default(permute_7, [2, 12, 50, 64]);  permute_7 = None
        view_15 = torch.ops.aten.view.default(permute_8, [2, 12, 50, 64]);  permute_8 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_13, view_14, view_15, None, True)
        getitem_4 = _scaled_dot_product_efficient_attention[0]
        getitem_5 = _scaled_dot_product_efficient_attention[1]
        getitem_6 = _scaled_dot_product_efficient_attention[2]
        getitem_7 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_9 = torch.ops.aten.permute.default(getitem_4, [2, 0, 1, 3])
        clone_4 = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_16 = torch.ops.aten.view.default(clone_4, [100, 768]);  clone_4 = None
        permute_10 = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_14, view_16, permute_10);  primals_14 = view_16 = None
        view_17 = torch.ops.aten.view.default(addmm_1, [50, 2, 768]);  addmm_1 = None
        add_17 = torch.ops.aten.add.Tensor(permute_3, view_17);  permute_3 = view_17 = None
        clone_5 = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
        var_mean_2 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8 = torch.ops.aten.sub.Tensor(clone_5, getitem_9);  clone_5 = getitem_9 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_2);  sub_8 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_15)
        add_19 = torch.ops.aten.add.Tensor(mul_10, primals_16);  mul_10 = primals_16 = None
        view_18 = torch.ops.aten.view.default(add_19, [100, 768]);  add_19 = None
        permute_11 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_18, view_18, permute_11);  primals_18 = view_18 = None
        view_19 = torch.ops.aten.view.default(addmm_2, [50, 2, 3072])
        mul_11 = torch.ops.aten.mul.Tensor(view_19, 1.702)
        sigmoid = torch.ops.aten.sigmoid.default(mul_11);  mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_19, sigmoid);  view_19 = sigmoid = None
        view_20 = torch.ops.aten.view.default(mul_12, [100, 3072]);  mul_12 = None
        permute_12 = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_20, view_20, permute_12);  primals_20 = view_20 = None
        view_21 = torch.ops.aten.view.default(addmm_3, [50, 2, 768]);  addmm_3 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, view_21);  add_17 = view_21 = None
        clone_6 = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
        var_mean_3 = torch.ops.aten.var_mean.correction(clone_6, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_9 = torch.ops.aten.sub.Tensor(clone_6, getitem_11);  clone_6 = getitem_11 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_3);  sub_9 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, primals_21)
        add_22 = torch.ops.aten.add.Tensor(mul_14, primals_22);  mul_14 = primals_22 = None
        view_22 = torch.ops.aten.view.default(add_22, [100, 768]);  add_22 = None
        permute_13 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_4 = torch.ops.aten.addmm.default(primals_23, view_22, permute_13);  primals_23 = view_22 = None
        view_23 = torch.ops.aten.view.default(addmm_4, [50, 2, 2304]);  addmm_4 = None
        view_24 = torch.ops.aten.view.default(view_23, [50, 2, 3, 768]);  view_23 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_24, 0);  view_24 = None
        permute_14 = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_14, -2);  permute_14 = None
        clone_7 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3 = torch.ops.aten.select.int(clone_7, 0, 0)
        select_4 = torch.ops.aten.select.int(clone_7, 0, 1)
        select_5 = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_25 = torch.ops.aten.view.default(select_3, [50, 24, 64]);  select_3 = None
        permute_15 = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        view_26 = torch.ops.aten.view.default(select_4, [50, 24, 64]);  select_4 = None
        permute_16 = torch.ops.aten.permute.default(view_26, [1, 0, 2]);  view_26 = None
        view_27 = torch.ops.aten.view.default(select_5, [50, 24, 64]);  select_5 = None
        permute_17 = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        view_28 = torch.ops.aten.view.default(permute_15, [2, 12, 50, 64]);  permute_15 = None
        view_29 = torch.ops.aten.view.default(permute_16, [2, 12, 50, 64]);  permute_16 = None
        view_30 = torch.ops.aten.view.default(permute_17, [2, 12, 50, 64]);  permute_17 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_28, view_29, view_30, None, True)
        getitem_12 = _scaled_dot_product_efficient_attention_1[0]
        getitem_13 = _scaled_dot_product_efficient_attention_1[1]
        getitem_14 = _scaled_dot_product_efficient_attention_1[2]
        getitem_15 = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        permute_18 = torch.ops.aten.permute.default(getitem_12, [2, 0, 1, 3])
        clone_8 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_31 = torch.ops.aten.view.default(clone_8, [100, 768]);  clone_8 = None
        permute_19 = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_26, view_31, permute_19);  primals_26 = view_31 = None
        view_32 = torch.ops.aten.view.default(addmm_5, [50, 2, 768]);  addmm_5 = None
        add_23 = torch.ops.aten.add.Tensor(add_20, view_32);  add_20 = view_32 = None
        clone_9 = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
        var_mean_4 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_10 = torch.ops.aten.sub.Tensor(clone_9, getitem_17);  clone_9 = getitem_17 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, primals_27)
        add_25 = torch.ops.aten.add.Tensor(mul_16, primals_28);  mul_16 = primals_28 = None
        view_33 = torch.ops.aten.view.default(add_25, [100, 768]);  add_25 = None
        permute_20 = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_30, view_33, permute_20);  primals_30 = view_33 = None
        view_34 = torch.ops.aten.view.default(addmm_6, [50, 2, 3072])
        mul_17 = torch.ops.aten.mul.Tensor(view_34, 1.702)
        sigmoid_1 = torch.ops.aten.sigmoid.default(mul_17);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_34, sigmoid_1);  view_34 = sigmoid_1 = None
        view_35 = torch.ops.aten.view.default(mul_18, [100, 3072]);  mul_18 = None
        permute_21 = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_32, view_35, permute_21);  primals_32 = view_35 = None
        view_36 = torch.ops.aten.view.default(addmm_7, [50, 2, 768]);  addmm_7 = None
        add_26 = torch.ops.aten.add.Tensor(add_23, view_36);  add_23 = view_36 = None
        clone_10 = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        var_mean_5 = torch.ops.aten.var_mean.correction(clone_10, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_5[0]
        getitem_19 = var_mean_5[1];  var_mean_5 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_11 = torch.ops.aten.sub.Tensor(clone_10, getitem_19);  clone_10 = getitem_19 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, primals_33)
        add_28 = torch.ops.aten.add.Tensor(mul_20, primals_34);  mul_20 = primals_34 = None
        view_37 = torch.ops.aten.view.default(add_28, [100, 768]);  add_28 = None
        permute_22 = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_35, view_37, permute_22);  primals_35 = view_37 = None
        view_38 = torch.ops.aten.view.default(addmm_8, [50, 2, 2304]);  addmm_8 = None
        view_39 = torch.ops.aten.view.default(view_38, [50, 2, 3, 768]);  view_38 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(view_39, 0);  view_39 = None
        permute_23 = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(permute_23, -2);  permute_23 = None
        clone_11 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6 = torch.ops.aten.select.int(clone_11, 0, 0)
        select_7 = torch.ops.aten.select.int(clone_11, 0, 1)
        select_8 = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_40 = torch.ops.aten.view.default(select_6, [50, 24, 64]);  select_6 = None
        permute_24 = torch.ops.aten.permute.default(view_40, [1, 0, 2]);  view_40 = None
        view_41 = torch.ops.aten.view.default(select_7, [50, 24, 64]);  select_7 = None
        permute_25 = torch.ops.aten.permute.default(view_41, [1, 0, 2]);  view_41 = None
        view_42 = torch.ops.aten.view.default(select_8, [50, 24, 64]);  select_8 = None
        permute_26 = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        view_43 = torch.ops.aten.view.default(permute_24, [2, 12, 50, 64]);  permute_24 = None
        view_44 = torch.ops.aten.view.default(permute_25, [2, 12, 50, 64]);  permute_25 = None
        view_45 = torch.ops.aten.view.default(permute_26, [2, 12, 50, 64]);  permute_26 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_43, view_44, view_45, None, True)
        getitem_20 = _scaled_dot_product_efficient_attention_2[0]
        getitem_21 = _scaled_dot_product_efficient_attention_2[1]
        getitem_22 = _scaled_dot_product_efficient_attention_2[2]
        getitem_23 = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        permute_27 = torch.ops.aten.permute.default(getitem_20, [2, 0, 1, 3])
        clone_12 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_46 = torch.ops.aten.view.default(clone_12, [100, 768]);  clone_12 = None
        permute_28 = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_38, view_46, permute_28);  primals_38 = view_46 = None
        view_47 = torch.ops.aten.view.default(addmm_9, [50, 2, 768]);  addmm_9 = None
        add_29 = torch.ops.aten.add.Tensor(add_26, view_47);  add_26 = view_47 = None
        clone_13 = torch.ops.aten.clone.default(add_29, memory_format = torch.contiguous_format)
        var_mean_6 = torch.ops.aten.var_mean.correction(clone_13, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(clone_13, getitem_25);  clone_13 = getitem_25 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, primals_39)
        add_31 = torch.ops.aten.add.Tensor(mul_22, primals_40);  mul_22 = primals_40 = None
        view_48 = torch.ops.aten.view.default(add_31, [100, 768]);  add_31 = None
        permute_29 = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_42, view_48, permute_29);  primals_42 = view_48 = None
        view_49 = torch.ops.aten.view.default(addmm_10, [50, 2, 3072])
        mul_23 = torch.ops.aten.mul.Tensor(view_49, 1.702)
        sigmoid_2 = torch.ops.aten.sigmoid.default(mul_23);  mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(view_49, sigmoid_2);  view_49 = sigmoid_2 = None
        view_50 = torch.ops.aten.view.default(mul_24, [100, 3072]);  mul_24 = None
        permute_30 = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_44, view_50, permute_30);  primals_44 = view_50 = None
        view_51 = torch.ops.aten.view.default(addmm_11, [50, 2, 768]);  addmm_11 = None
        add_32 = torch.ops.aten.add.Tensor(add_29, view_51);  add_29 = view_51 = None
        clone_14 = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format)
        var_mean_7 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_7[0]
        getitem_27 = var_mean_7[1];  var_mean_7 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13 = torch.ops.aten.sub.Tensor(clone_14, getitem_27);  clone_14 = getitem_27 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, primals_45)
        add_34 = torch.ops.aten.add.Tensor(mul_26, primals_46);  mul_26 = primals_46 = None
        view_52 = torch.ops.aten.view.default(add_34, [100, 768]);  add_34 = None
        permute_31 = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_47, view_52, permute_31);  primals_47 = view_52 = None
        view_53 = torch.ops.aten.view.default(addmm_12, [50, 2, 2304]);  addmm_12 = None
        view_54 = torch.ops.aten.view.default(view_53, [50, 2, 3, 768]);  view_53 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_54, 0);  view_54 = None
        permute_32 = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(permute_32, -2);  permute_32 = None
        clone_15 = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9 = torch.ops.aten.select.int(clone_15, 0, 0)
        select_10 = torch.ops.aten.select.int(clone_15, 0, 1)
        select_11 = torch.ops.aten.select.int(clone_15, 0, 2);  clone_15 = None
        view_55 = torch.ops.aten.view.default(select_9, [50, 24, 64]);  select_9 = None
        permute_33 = torch.ops.aten.permute.default(view_55, [1, 0, 2]);  view_55 = None
        view_56 = torch.ops.aten.view.default(select_10, [50, 24, 64]);  select_10 = None
        permute_34 = torch.ops.aten.permute.default(view_56, [1, 0, 2]);  view_56 = None
        view_57 = torch.ops.aten.view.default(select_11, [50, 24, 64]);  select_11 = None
        permute_35 = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        view_58 = torch.ops.aten.view.default(permute_33, [2, 12, 50, 64]);  permute_33 = None
        view_59 = torch.ops.aten.view.default(permute_34, [2, 12, 50, 64]);  permute_34 = None
        view_60 = torch.ops.aten.view.default(permute_35, [2, 12, 50, 64]);  permute_35 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_58, view_59, view_60, None, True)
        getitem_28 = _scaled_dot_product_efficient_attention_3[0]
        getitem_29 = _scaled_dot_product_efficient_attention_3[1]
        getitem_30 = _scaled_dot_product_efficient_attention_3[2]
        getitem_31 = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3])
        clone_16 = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_61 = torch.ops.aten.view.default(clone_16, [100, 768]);  clone_16 = None
        permute_37 = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_50, view_61, permute_37);  primals_50 = view_61 = None
        view_62 = torch.ops.aten.view.default(addmm_13, [50, 2, 768]);  addmm_13 = None
        add_35 = torch.ops.aten.add.Tensor(add_32, view_62);  add_32 = view_62 = None
        clone_17 = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
        var_mean_8 = torch.ops.aten.var_mean.correction(clone_17, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14 = torch.ops.aten.sub.Tensor(clone_17, getitem_33);  clone_17 = getitem_33 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, primals_51)
        add_37 = torch.ops.aten.add.Tensor(mul_28, primals_52);  mul_28 = primals_52 = None
        view_63 = torch.ops.aten.view.default(add_37, [100, 768]);  add_37 = None
        permute_38 = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_54, view_63, permute_38);  primals_54 = view_63 = None
        view_64 = torch.ops.aten.view.default(addmm_14, [50, 2, 3072])
        mul_29 = torch.ops.aten.mul.Tensor(view_64, 1.702)
        sigmoid_3 = torch.ops.aten.sigmoid.default(mul_29);  mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_64, sigmoid_3);  view_64 = sigmoid_3 = None
        view_65 = torch.ops.aten.view.default(mul_30, [100, 3072]);  mul_30 = None
        permute_39 = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_56, view_65, permute_39);  primals_56 = view_65 = None
        view_66 = torch.ops.aten.view.default(addmm_15, [50, 2, 768]);  addmm_15 = None
        add_38 = torch.ops.aten.add.Tensor(add_35, view_66);  add_35 = view_66 = None
        permute_40 = torch.ops.aten.permute.default(add_38, [1, 0, 2])
        clone_18 = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
        var_mean_9 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_9[0]
        getitem_35 = var_mean_9[1];  var_mean_9 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_15 = torch.ops.aten.sub.Tensor(clone_18, getitem_35);  clone_18 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, primals_57);  mul_31 = None
        add_40 = torch.ops.aten.add.Tensor(mul_32, primals_58);  mul_32 = primals_58 = None
        view_67 = torch.ops.aten.view.default(add_40, [100, 768]);  add_40 = None
        permute_41 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_59, view_67, permute_41);  primals_59 = view_67 = None
        view_68 = torch.ops.aten.view.default(addmm_16, [50, 2, 2304]);  addmm_16 = None
        view_69 = torch.ops.aten.view.default(view_68, [50, 2, 3, 768]);  view_68 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(view_69, 0);  view_69 = None
        permute_42 = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(permute_42, -2);  permute_42 = None
        clone_19 = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12 = torch.ops.aten.select.int(clone_19, 0, 0)
        select_13 = torch.ops.aten.select.int(clone_19, 0, 1)
        select_14 = torch.ops.aten.select.int(clone_19, 0, 2);  clone_19 = None
        view_70 = torch.ops.aten.view.default(select_12, [50, 24, 64]);  select_12 = None
        permute_43 = torch.ops.aten.permute.default(view_70, [1, 0, 2]);  view_70 = None
        view_71 = torch.ops.aten.view.default(select_13, [50, 24, 64]);  select_13 = None
        permute_44 = torch.ops.aten.permute.default(view_71, [1, 0, 2]);  view_71 = None
        view_72 = torch.ops.aten.view.default(select_14, [50, 24, 64]);  select_14 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        view_73 = torch.ops.aten.view.default(permute_43, [2, 12, 50, 64]);  permute_43 = None
        view_74 = torch.ops.aten.view.default(permute_44, [2, 12, 50, 64]);  permute_44 = None
        view_75 = torch.ops.aten.view.default(permute_45, [2, 12, 50, 64]);  permute_45 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_73, view_74, view_75, None, True)
        getitem_36 = _scaled_dot_product_efficient_attention_4[0]
        getitem_37 = _scaled_dot_product_efficient_attention_4[1]
        getitem_38 = _scaled_dot_product_efficient_attention_4[2]
        getitem_39 = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_36, [2, 0, 1, 3])
        clone_20 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_76 = torch.ops.aten.view.default(clone_20, [100, 768]);  clone_20 = None
        permute_47 = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_62, view_76, permute_47);  primals_62 = view_76 = None
        view_77 = torch.ops.aten.view.default(addmm_17, [50, 2, 768]);  addmm_17 = None
        add_41 = torch.ops.aten.add.Tensor(add_38, view_77);  view_77 = None
        clone_21 = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
        var_mean_10 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16 = torch.ops.aten.sub.Tensor(clone_21, getitem_41);  clone_21 = getitem_41 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, primals_63)
        add_43 = torch.ops.aten.add.Tensor(mul_34, primals_64);  mul_34 = primals_64 = None
        view_78 = torch.ops.aten.view.default(add_43, [100, 768]);  add_43 = None
        permute_48 = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_66, view_78, permute_48);  primals_66 = view_78 = None
        view_79 = torch.ops.aten.view.default(addmm_18, [50, 2, 3072])
        mul_35 = torch.ops.aten.mul.Tensor(view_79, 1.702)
        sigmoid_4 = torch.ops.aten.sigmoid.default(mul_35);  mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_79, sigmoid_4);  view_79 = sigmoid_4 = None
        view_80 = torch.ops.aten.view.default(mul_36, [100, 3072]);  mul_36 = None
        permute_49 = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_68, view_80, permute_49);  primals_68 = view_80 = None
        view_81 = torch.ops.aten.view.default(addmm_19, [50, 2, 768]);  addmm_19 = None
        add_44 = torch.ops.aten.add.Tensor(add_41, view_81);  add_41 = view_81 = None
        clone_22 = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
        var_mean_11 = torch.ops.aten.var_mean.correction(clone_22, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_11[0]
        getitem_43 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_17 = torch.ops.aten.sub.Tensor(clone_22, getitem_43);  clone_22 = getitem_43 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, primals_69)
        add_46 = torch.ops.aten.add.Tensor(mul_38, primals_70);  mul_38 = primals_70 = None
        view_82 = torch.ops.aten.view.default(add_46, [100, 768]);  add_46 = None
        permute_50 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_71, view_82, permute_50);  primals_71 = view_82 = None
        view_83 = torch.ops.aten.view.default(addmm_20, [50, 2, 2304]);  addmm_20 = None
        view_84 = torch.ops.aten.view.default(view_83, [50, 2, 3, 768]);  view_83 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(view_84, 0);  view_84 = None
        permute_51 = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(permute_51, -2);  permute_51 = None
        clone_23 = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15 = torch.ops.aten.select.int(clone_23, 0, 0)
        select_16 = torch.ops.aten.select.int(clone_23, 0, 1)
        select_17 = torch.ops.aten.select.int(clone_23, 0, 2);  clone_23 = None
        view_85 = torch.ops.aten.view.default(select_15, [50, 24, 64]);  select_15 = None
        permute_52 = torch.ops.aten.permute.default(view_85, [1, 0, 2]);  view_85 = None
        view_86 = torch.ops.aten.view.default(select_16, [50, 24, 64]);  select_16 = None
        permute_53 = torch.ops.aten.permute.default(view_86, [1, 0, 2]);  view_86 = None
        view_87 = torch.ops.aten.view.default(select_17, [50, 24, 64]);  select_17 = None
        permute_54 = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        view_88 = torch.ops.aten.view.default(permute_52, [2, 12, 50, 64]);  permute_52 = None
        view_89 = torch.ops.aten.view.default(permute_53, [2, 12, 50, 64]);  permute_53 = None
        view_90 = torch.ops.aten.view.default(permute_54, [2, 12, 50, 64]);  permute_54 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_88, view_89, view_90, None, True)
        getitem_44 = _scaled_dot_product_efficient_attention_5[0]
        getitem_45 = _scaled_dot_product_efficient_attention_5[1]
        getitem_46 = _scaled_dot_product_efficient_attention_5[2]
        getitem_47 = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        permute_55 = torch.ops.aten.permute.default(getitem_44, [2, 0, 1, 3])
        clone_24 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_91 = torch.ops.aten.view.default(clone_24, [100, 768]);  clone_24 = None
        permute_56 = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_74, view_91, permute_56);  primals_74 = view_91 = None
        view_92 = torch.ops.aten.view.default(addmm_21, [50, 2, 768]);  addmm_21 = None
        add_47 = torch.ops.aten.add.Tensor(add_44, view_92);  add_44 = view_92 = None
        clone_25 = torch.ops.aten.clone.default(add_47, memory_format = torch.contiguous_format)
        var_mean_12 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_18 = torch.ops.aten.sub.Tensor(clone_25, getitem_49);  clone_25 = getitem_49 = None
        mul_39 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_39, primals_75)
        add_49 = torch.ops.aten.add.Tensor(mul_40, primals_76);  mul_40 = primals_76 = None
        view_93 = torch.ops.aten.view.default(add_49, [100, 768]);  add_49 = None
        permute_57 = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_78, view_93, permute_57);  primals_78 = view_93 = None
        view_94 = torch.ops.aten.view.default(addmm_22, [50, 2, 3072])
        mul_41 = torch.ops.aten.mul.Tensor(view_94, 1.702)
        sigmoid_5 = torch.ops.aten.sigmoid.default(mul_41);  mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_94, sigmoid_5);  view_94 = sigmoid_5 = None
        view_95 = torch.ops.aten.view.default(mul_42, [100, 3072]);  mul_42 = None
        permute_58 = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_80, view_95, permute_58);  primals_80 = view_95 = None
        view_96 = torch.ops.aten.view.default(addmm_23, [50, 2, 768]);  addmm_23 = None
        add_50 = torch.ops.aten.add.Tensor(add_47, view_96);  add_47 = view_96 = None
        clone_26 = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
        var_mean_13 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_13[0]
        getitem_51 = var_mean_13[1];  var_mean_13 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_19 = torch.ops.aten.sub.Tensor(clone_26, getitem_51);  clone_26 = getitem_51 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, primals_81)
        add_52 = torch.ops.aten.add.Tensor(mul_44, primals_82);  mul_44 = primals_82 = None
        view_97 = torch.ops.aten.view.default(add_52, [100, 768]);  add_52 = None
        permute_59 = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        addmm_24 = torch.ops.aten.addmm.default(primals_83, view_97, permute_59);  primals_83 = view_97 = None
        view_98 = torch.ops.aten.view.default(addmm_24, [50, 2, 2304]);  addmm_24 = None
        view_99 = torch.ops.aten.view.default(view_98, [50, 2, 3, 768]);  view_98 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(view_99, 0);  view_99 = None
        permute_60 = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(permute_60, -2);  permute_60 = None
        clone_27 = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18 = torch.ops.aten.select.int(clone_27, 0, 0)
        select_19 = torch.ops.aten.select.int(clone_27, 0, 1)
        select_20 = torch.ops.aten.select.int(clone_27, 0, 2);  clone_27 = None
        view_100 = torch.ops.aten.view.default(select_18, [50, 24, 64]);  select_18 = None
        permute_61 = torch.ops.aten.permute.default(view_100, [1, 0, 2]);  view_100 = None
        view_101 = torch.ops.aten.view.default(select_19, [50, 24, 64]);  select_19 = None
        permute_62 = torch.ops.aten.permute.default(view_101, [1, 0, 2]);  view_101 = None
        view_102 = torch.ops.aten.view.default(select_20, [50, 24, 64]);  select_20 = None
        permute_63 = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        view_103 = torch.ops.aten.view.default(permute_61, [2, 12, 50, 64]);  permute_61 = None
        view_104 = torch.ops.aten.view.default(permute_62, [2, 12, 50, 64]);  permute_62 = None
        view_105 = torch.ops.aten.view.default(permute_63, [2, 12, 50, 64]);  permute_63 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_103, view_104, view_105, None, True)
        getitem_52 = _scaled_dot_product_efficient_attention_6[0]
        getitem_53 = _scaled_dot_product_efficient_attention_6[1]
        getitem_54 = _scaled_dot_product_efficient_attention_6[2]
        getitem_55 = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
        permute_64 = torch.ops.aten.permute.default(getitem_52, [2, 0, 1, 3])
        clone_28 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106 = torch.ops.aten.view.default(clone_28, [100, 768]);  clone_28 = None
        permute_65 = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        addmm_25 = torch.ops.aten.addmm.default(primals_86, view_106, permute_65);  primals_86 = view_106 = None
        view_107 = torch.ops.aten.view.default(addmm_25, [50, 2, 768]);  addmm_25 = None
        add_53 = torch.ops.aten.add.Tensor(add_50, view_107);  add_50 = view_107 = None
        clone_29 = torch.ops.aten.clone.default(add_53, memory_format = torch.contiguous_format)
        var_mean_14 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_20 = torch.ops.aten.sub.Tensor(clone_29, getitem_57);  clone_29 = getitem_57 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, primals_87)
        add_55 = torch.ops.aten.add.Tensor(mul_46, primals_88);  mul_46 = primals_88 = None
        view_108 = torch.ops.aten.view.default(add_55, [100, 768]);  add_55 = None
        permute_66 = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        addmm_26 = torch.ops.aten.addmm.default(primals_90, view_108, permute_66);  primals_90 = view_108 = None
        view_109 = torch.ops.aten.view.default(addmm_26, [50, 2, 3072])
        mul_47 = torch.ops.aten.mul.Tensor(view_109, 1.702)
        sigmoid_6 = torch.ops.aten.sigmoid.default(mul_47);  mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(view_109, sigmoid_6);  view_109 = sigmoid_6 = None
        view_110 = torch.ops.aten.view.default(mul_48, [100, 3072]);  mul_48 = None
        permute_67 = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        addmm_27 = torch.ops.aten.addmm.default(primals_92, view_110, permute_67);  primals_92 = view_110 = None
        view_111 = torch.ops.aten.view.default(addmm_27, [50, 2, 768]);  addmm_27 = None
        add_56 = torch.ops.aten.add.Tensor(add_53, view_111);  add_53 = view_111 = None
        clone_30 = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
        var_mean_15 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_15[0]
        getitem_59 = var_mean_15[1];  var_mean_15 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_21 = torch.ops.aten.sub.Tensor(clone_30, getitem_59);  clone_30 = getitem_59 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_93)
        add_58 = torch.ops.aten.add.Tensor(mul_50, primals_94);  mul_50 = primals_94 = None
        view_112 = torch.ops.aten.view.default(add_58, [100, 768]);  add_58 = None
        permute_68 = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        addmm_28 = torch.ops.aten.addmm.default(primals_95, view_112, permute_68);  primals_95 = view_112 = None
        view_113 = torch.ops.aten.view.default(addmm_28, [50, 2, 2304]);  addmm_28 = None
        view_114 = torch.ops.aten.view.default(view_113, [50, 2, 3, 768]);  view_113 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_114, 0);  view_114 = None
        permute_69 = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(permute_69, -2);  permute_69 = None
        clone_31 = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21 = torch.ops.aten.select.int(clone_31, 0, 0)
        select_22 = torch.ops.aten.select.int(clone_31, 0, 1)
        select_23 = torch.ops.aten.select.int(clone_31, 0, 2);  clone_31 = None
        view_115 = torch.ops.aten.view.default(select_21, [50, 24, 64]);  select_21 = None
        permute_70 = torch.ops.aten.permute.default(view_115, [1, 0, 2]);  view_115 = None
        view_116 = torch.ops.aten.view.default(select_22, [50, 24, 64]);  select_22 = None
        permute_71 = torch.ops.aten.permute.default(view_116, [1, 0, 2]);  view_116 = None
        view_117 = torch.ops.aten.view.default(select_23, [50, 24, 64]);  select_23 = None
        permute_72 = torch.ops.aten.permute.default(view_117, [1, 0, 2]);  view_117 = None
        view_118 = torch.ops.aten.view.default(permute_70, [2, 12, 50, 64]);  permute_70 = None
        view_119 = torch.ops.aten.view.default(permute_71, [2, 12, 50, 64]);  permute_71 = None
        view_120 = torch.ops.aten.view.default(permute_72, [2, 12, 50, 64]);  permute_72 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_118, view_119, view_120, None, True)
        getitem_60 = _scaled_dot_product_efficient_attention_7[0]
        getitem_61 = _scaled_dot_product_efficient_attention_7[1]
        getitem_62 = _scaled_dot_product_efficient_attention_7[2]
        getitem_63 = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
        permute_73 = torch.ops.aten.permute.default(getitem_60, [2, 0, 1, 3])
        clone_32 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_121 = torch.ops.aten.view.default(clone_32, [100, 768]);  clone_32 = None
        permute_74 = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        addmm_29 = torch.ops.aten.addmm.default(primals_98, view_121, permute_74);  primals_98 = view_121 = None
        view_122 = torch.ops.aten.view.default(addmm_29, [50, 2, 768]);  addmm_29 = None
        add_59 = torch.ops.aten.add.Tensor(add_56, view_122);  add_56 = view_122 = None
        clone_33 = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
        var_mean_16 = torch.ops.aten.var_mean.correction(clone_33, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_22 = torch.ops.aten.sub.Tensor(clone_33, getitem_65);  clone_33 = getitem_65 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, primals_99)
        add_61 = torch.ops.aten.add.Tensor(mul_52, primals_100);  mul_52 = primals_100 = None
        view_123 = torch.ops.aten.view.default(add_61, [100, 768]);  add_61 = None
        permute_75 = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        addmm_30 = torch.ops.aten.addmm.default(primals_102, view_123, permute_75);  primals_102 = view_123 = None
        view_124 = torch.ops.aten.view.default(addmm_30, [50, 2, 3072])
        mul_53 = torch.ops.aten.mul.Tensor(view_124, 1.702)
        sigmoid_7 = torch.ops.aten.sigmoid.default(mul_53);  mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_124, sigmoid_7);  view_124 = sigmoid_7 = None
        view_125 = torch.ops.aten.view.default(mul_54, [100, 3072]);  mul_54 = None
        permute_76 = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        addmm_31 = torch.ops.aten.addmm.default(primals_104, view_125, permute_76);  primals_104 = view_125 = None
        view_126 = torch.ops.aten.view.default(addmm_31, [50, 2, 768]);  addmm_31 = None
        add_62 = torch.ops.aten.add.Tensor(add_59, view_126);  add_59 = view_126 = None
        permute_77 = torch.ops.aten.permute.default(add_62, [1, 0, 2])
        clone_34 = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
        var_mean_17 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_17[0]
        getitem_67 = var_mean_17[1];  var_mean_17 = None
        add_63 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_23 = torch.ops.aten.sub.Tensor(clone_34, getitem_67);  clone_34 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, primals_105);  mul_55 = None
        add_64 = torch.ops.aten.add.Tensor(mul_56, primals_106);  mul_56 = primals_106 = None
        view_127 = torch.ops.aten.view.default(add_64, [100, 768]);  add_64 = None
        permute_78 = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        addmm_32 = torch.ops.aten.addmm.default(primals_107, view_127, permute_78);  primals_107 = view_127 = None
        view_128 = torch.ops.aten.view.default(addmm_32, [50, 2, 2304]);  addmm_32 = None
        view_129 = torch.ops.aten.view.default(view_128, [50, 2, 3, 768]);  view_128 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(view_129, 0);  view_129 = None
        permute_79 = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(permute_79, -2);  permute_79 = None
        clone_35 = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24 = torch.ops.aten.select.int(clone_35, 0, 0)
        select_25 = torch.ops.aten.select.int(clone_35, 0, 1)
        select_26 = torch.ops.aten.select.int(clone_35, 0, 2);  clone_35 = None
        view_130 = torch.ops.aten.view.default(select_24, [50, 24, 64]);  select_24 = None
        permute_80 = torch.ops.aten.permute.default(view_130, [1, 0, 2]);  view_130 = None
        view_131 = torch.ops.aten.view.default(select_25, [50, 24, 64]);  select_25 = None
        permute_81 = torch.ops.aten.permute.default(view_131, [1, 0, 2]);  view_131 = None
        view_132 = torch.ops.aten.view.default(select_26, [50, 24, 64]);  select_26 = None
        permute_82 = torch.ops.aten.permute.default(view_132, [1, 0, 2]);  view_132 = None
        view_133 = torch.ops.aten.view.default(permute_80, [2, 12, 50, 64]);  permute_80 = None
        view_134 = torch.ops.aten.view.default(permute_81, [2, 12, 50, 64]);  permute_81 = None
        view_135 = torch.ops.aten.view.default(permute_82, [2, 12, 50, 64]);  permute_82 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_133, view_134, view_135, None, True)
        getitem_68 = _scaled_dot_product_efficient_attention_8[0]
        getitem_69 = _scaled_dot_product_efficient_attention_8[1]
        getitem_70 = _scaled_dot_product_efficient_attention_8[2]
        getitem_71 = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
        permute_83 = torch.ops.aten.permute.default(getitem_68, [2, 0, 1, 3])
        clone_36 = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_136 = torch.ops.aten.view.default(clone_36, [100, 768]);  clone_36 = None
        permute_84 = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        addmm_33 = torch.ops.aten.addmm.default(primals_110, view_136, permute_84);  primals_110 = view_136 = None
        view_137 = torch.ops.aten.view.default(addmm_33, [50, 2, 768]);  addmm_33 = None
        add_65 = torch.ops.aten.add.Tensor(add_62, view_137);  view_137 = None
        clone_37 = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
        var_mean_18 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_24 = torch.ops.aten.sub.Tensor(clone_37, getitem_73);  clone_37 = getitem_73 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_18);  sub_24 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, primals_111)
        add_67 = torch.ops.aten.add.Tensor(mul_58, primals_112);  mul_58 = primals_112 = None
        view_138 = torch.ops.aten.view.default(add_67, [100, 768]);  add_67 = None
        permute_85 = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        addmm_34 = torch.ops.aten.addmm.default(primals_114, view_138, permute_85);  primals_114 = view_138 = None
        view_139 = torch.ops.aten.view.default(addmm_34, [50, 2, 3072])
        mul_59 = torch.ops.aten.mul.Tensor(view_139, 1.702)
        sigmoid_8 = torch.ops.aten.sigmoid.default(mul_59);  mul_59 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_139, sigmoid_8);  view_139 = sigmoid_8 = None
        view_140 = torch.ops.aten.view.default(mul_60, [100, 3072]);  mul_60 = None
        permute_86 = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        addmm_35 = torch.ops.aten.addmm.default(primals_116, view_140, permute_86);  primals_116 = view_140 = None
        view_141 = torch.ops.aten.view.default(addmm_35, [50, 2, 768]);  addmm_35 = None
        add_68 = torch.ops.aten.add.Tensor(add_65, view_141);  add_65 = view_141 = None
        clone_38 = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
        var_mean_19 = torch.ops.aten.var_mean.correction(clone_38, [2], correction = 0, keepdim = True)
        getitem_74 = var_mean_19[0]
        getitem_75 = var_mean_19[1];  var_mean_19 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_25 = torch.ops.aten.sub.Tensor(clone_38, getitem_75);  clone_38 = getitem_75 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_19);  sub_25 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, primals_117)
        add_70 = torch.ops.aten.add.Tensor(mul_62, primals_118);  mul_62 = primals_118 = None
        view_142 = torch.ops.aten.view.default(add_70, [100, 768]);  add_70 = None
        permute_87 = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        addmm_36 = torch.ops.aten.addmm.default(primals_119, view_142, permute_87);  primals_119 = view_142 = None
        view_143 = torch.ops.aten.view.default(addmm_36, [50, 2, 2304]);  addmm_36 = None
        view_144 = torch.ops.aten.view.default(view_143, [50, 2, 3, 768]);  view_143 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_144, 0);  view_144 = None
        permute_88 = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(permute_88, -2);  permute_88 = None
        clone_39 = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27 = torch.ops.aten.select.int(clone_39, 0, 0)
        select_28 = torch.ops.aten.select.int(clone_39, 0, 1)
        select_29 = torch.ops.aten.select.int(clone_39, 0, 2);  clone_39 = None
        view_145 = torch.ops.aten.view.default(select_27, [50, 24, 64]);  select_27 = None
        permute_89 = torch.ops.aten.permute.default(view_145, [1, 0, 2]);  view_145 = None
        view_146 = torch.ops.aten.view.default(select_28, [50, 24, 64]);  select_28 = None
        permute_90 = torch.ops.aten.permute.default(view_146, [1, 0, 2]);  view_146 = None
        view_147 = torch.ops.aten.view.default(select_29, [50, 24, 64]);  select_29 = None
        permute_91 = torch.ops.aten.permute.default(view_147, [1, 0, 2]);  view_147 = None
        view_148 = torch.ops.aten.view.default(permute_89, [2, 12, 50, 64]);  permute_89 = None
        view_149 = torch.ops.aten.view.default(permute_90, [2, 12, 50, 64]);  permute_90 = None
        view_150 = torch.ops.aten.view.default(permute_91, [2, 12, 50, 64]);  permute_91 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_148, view_149, view_150, None, True)
        getitem_76 = _scaled_dot_product_efficient_attention_9[0]
        getitem_77 = _scaled_dot_product_efficient_attention_9[1]
        getitem_78 = _scaled_dot_product_efficient_attention_9[2]
        getitem_79 = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
        permute_92 = torch.ops.aten.permute.default(getitem_76, [2, 0, 1, 3])
        clone_40 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_151 = torch.ops.aten.view.default(clone_40, [100, 768]);  clone_40 = None
        permute_93 = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        addmm_37 = torch.ops.aten.addmm.default(primals_122, view_151, permute_93);  primals_122 = view_151 = None
        view_152 = torch.ops.aten.view.default(addmm_37, [50, 2, 768]);  addmm_37 = None
        add_71 = torch.ops.aten.add.Tensor(add_68, view_152);  add_68 = view_152 = None
        clone_41 = torch.ops.aten.clone.default(add_71, memory_format = torch.contiguous_format)
        var_mean_20 = torch.ops.aten.var_mean.correction(clone_41, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_26 = torch.ops.aten.sub.Tensor(clone_41, getitem_81);  clone_41 = getitem_81 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_20);  sub_26 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, primals_123)
        add_73 = torch.ops.aten.add.Tensor(mul_64, primals_124);  mul_64 = primals_124 = None
        view_153 = torch.ops.aten.view.default(add_73, [100, 768]);  add_73 = None
        permute_94 = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        addmm_38 = torch.ops.aten.addmm.default(primals_126, view_153, permute_94);  primals_126 = view_153 = None
        view_154 = torch.ops.aten.view.default(addmm_38, [50, 2, 3072])
        mul_65 = torch.ops.aten.mul.Tensor(view_154, 1.702)
        sigmoid_9 = torch.ops.aten.sigmoid.default(mul_65);  mul_65 = None
        mul_66 = torch.ops.aten.mul.Tensor(view_154, sigmoid_9);  view_154 = sigmoid_9 = None
        view_155 = torch.ops.aten.view.default(mul_66, [100, 3072]);  mul_66 = None
        permute_95 = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        addmm_39 = torch.ops.aten.addmm.default(primals_128, view_155, permute_95);  primals_128 = view_155 = None
        view_156 = torch.ops.aten.view.default(addmm_39, [50, 2, 768]);  addmm_39 = None
        add_74 = torch.ops.aten.add.Tensor(add_71, view_156);  add_71 = view_156 = None
        clone_42 = torch.ops.aten.clone.default(add_74, memory_format = torch.contiguous_format)
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
        getitem_82 = var_mean_21[0]
        getitem_83 = var_mean_21[1];  var_mean_21 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_27 = torch.ops.aten.sub.Tensor(clone_42, getitem_83);  clone_42 = getitem_83 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_21);  sub_27 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, primals_129)
        add_76 = torch.ops.aten.add.Tensor(mul_68, primals_130);  mul_68 = primals_130 = None
        view_157 = torch.ops.aten.view.default(add_76, [100, 768]);  add_76 = None
        permute_96 = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        addmm_40 = torch.ops.aten.addmm.default(primals_131, view_157, permute_96);  primals_131 = view_157 = None
        view_158 = torch.ops.aten.view.default(addmm_40, [50, 2, 2304]);  addmm_40 = None
        view_159 = torch.ops.aten.view.default(view_158, [50, 2, 3, 768]);  view_158 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(view_159, 0);  view_159 = None
        permute_97 = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(permute_97, -2);  permute_97 = None
        clone_43 = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30 = torch.ops.aten.select.int(clone_43, 0, 0)
        select_31 = torch.ops.aten.select.int(clone_43, 0, 1)
        select_32 = torch.ops.aten.select.int(clone_43, 0, 2);  clone_43 = None
        view_160 = torch.ops.aten.view.default(select_30, [50, 24, 64]);  select_30 = None
        permute_98 = torch.ops.aten.permute.default(view_160, [1, 0, 2]);  view_160 = None
        view_161 = torch.ops.aten.view.default(select_31, [50, 24, 64]);  select_31 = None
        permute_99 = torch.ops.aten.permute.default(view_161, [1, 0, 2]);  view_161 = None
        view_162 = torch.ops.aten.view.default(select_32, [50, 24, 64]);  select_32 = None
        permute_100 = torch.ops.aten.permute.default(view_162, [1, 0, 2]);  view_162 = None
        view_163 = torch.ops.aten.view.default(permute_98, [2, 12, 50, 64]);  permute_98 = None
        view_164 = torch.ops.aten.view.default(permute_99, [2, 12, 50, 64]);  permute_99 = None
        view_165 = torch.ops.aten.view.default(permute_100, [2, 12, 50, 64]);  permute_100 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_163, view_164, view_165, None, True)
        getitem_84 = _scaled_dot_product_efficient_attention_10[0]
        getitem_85 = _scaled_dot_product_efficient_attention_10[1]
        getitem_86 = _scaled_dot_product_efficient_attention_10[2]
        getitem_87 = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
        permute_101 = torch.ops.aten.permute.default(getitem_84, [2, 0, 1, 3])
        clone_44 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_166 = torch.ops.aten.view.default(clone_44, [100, 768]);  clone_44 = None
        permute_102 = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        addmm_41 = torch.ops.aten.addmm.default(primals_134, view_166, permute_102);  primals_134 = view_166 = None
        view_167 = torch.ops.aten.view.default(addmm_41, [50, 2, 768]);  addmm_41 = None
        add_77 = torch.ops.aten.add.Tensor(add_74, view_167);  add_74 = view_167 = None
        clone_45 = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
        var_mean_22 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_28 = torch.ops.aten.sub.Tensor(clone_45, getitem_89);  clone_45 = getitem_89 = None
        mul_69 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_69, primals_135)
        add_79 = torch.ops.aten.add.Tensor(mul_70, primals_136);  mul_70 = primals_136 = None
        view_168 = torch.ops.aten.view.default(add_79, [100, 768]);  add_79 = None
        permute_103 = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        addmm_42 = torch.ops.aten.addmm.default(primals_138, view_168, permute_103);  primals_138 = view_168 = None
        view_169 = torch.ops.aten.view.default(addmm_42, [50, 2, 3072])
        mul_71 = torch.ops.aten.mul.Tensor(view_169, 1.702)
        sigmoid_10 = torch.ops.aten.sigmoid.default(mul_71);  mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(view_169, sigmoid_10);  view_169 = sigmoid_10 = None
        view_170 = torch.ops.aten.view.default(mul_72, [100, 3072]);  mul_72 = None
        permute_104 = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        addmm_43 = torch.ops.aten.addmm.default(primals_140, view_170, permute_104);  primals_140 = view_170 = None
        view_171 = torch.ops.aten.view.default(addmm_43, [50, 2, 768]);  addmm_43 = None
        add_80 = torch.ops.aten.add.Tensor(add_77, view_171);  add_77 = view_171 = None
        clone_46 = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
        var_mean_23 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_23[0]
        getitem_91 = var_mean_23[1];  var_mean_23 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_29 = torch.ops.aten.sub.Tensor(clone_46, getitem_91);  clone_46 = getitem_91 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, primals_141)
        add_82 = torch.ops.aten.add.Tensor(mul_74, primals_142);  mul_74 = primals_142 = None
        view_172 = torch.ops.aten.view.default(add_82, [100, 768]);  add_82 = None
        permute_105 = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        addmm_44 = torch.ops.aten.addmm.default(primals_143, view_172, permute_105);  primals_143 = view_172 = None
        view_173 = torch.ops.aten.view.default(addmm_44, [50, 2, 2304]);  addmm_44 = None
        view_174 = torch.ops.aten.view.default(view_173, [50, 2, 3, 768]);  view_173 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_174, 0);  view_174 = None
        permute_106 = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(permute_106, -2);  permute_106 = None
        clone_47 = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33 = torch.ops.aten.select.int(clone_47, 0, 0)
        select_34 = torch.ops.aten.select.int(clone_47, 0, 1)
        select_35 = torch.ops.aten.select.int(clone_47, 0, 2);  clone_47 = None
        view_175 = torch.ops.aten.view.default(select_33, [50, 24, 64]);  select_33 = None
        permute_107 = torch.ops.aten.permute.default(view_175, [1, 0, 2]);  view_175 = None
        view_176 = torch.ops.aten.view.default(select_34, [50, 24, 64]);  select_34 = None
        permute_108 = torch.ops.aten.permute.default(view_176, [1, 0, 2]);  view_176 = None
        view_177 = torch.ops.aten.view.default(select_35, [50, 24, 64]);  select_35 = None
        permute_109 = torch.ops.aten.permute.default(view_177, [1, 0, 2]);  view_177 = None
        view_178 = torch.ops.aten.view.default(permute_107, [2, 12, 50, 64]);  permute_107 = None
        view_179 = torch.ops.aten.view.default(permute_108, [2, 12, 50, 64]);  permute_108 = None
        view_180 = torch.ops.aten.view.default(permute_109, [2, 12, 50, 64]);  permute_109 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_178, view_179, view_180, None, True)
        getitem_92 = _scaled_dot_product_efficient_attention_11[0]
        getitem_93 = _scaled_dot_product_efficient_attention_11[1]
        getitem_94 = _scaled_dot_product_efficient_attention_11[2]
        getitem_95 = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
        permute_110 = torch.ops.aten.permute.default(getitem_92, [2, 0, 1, 3])
        clone_48 = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_181 = torch.ops.aten.view.default(clone_48, [100, 768]);  clone_48 = None
        permute_111 = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        addmm_45 = torch.ops.aten.addmm.default(primals_146, view_181, permute_111);  primals_146 = view_181 = None
        view_182 = torch.ops.aten.view.default(addmm_45, [50, 2, 768]);  addmm_45 = None
        add_83 = torch.ops.aten.add.Tensor(add_80, view_182);  add_80 = view_182 = None
        clone_49 = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
        var_mean_24 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_30 = torch.ops.aten.sub.Tensor(clone_49, getitem_97);  clone_49 = getitem_97 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_24);  sub_30 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, primals_147)
        add_85 = torch.ops.aten.add.Tensor(mul_76, primals_148);  mul_76 = primals_148 = None
        view_183 = torch.ops.aten.view.default(add_85, [100, 768]);  add_85 = None
        permute_112 = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        addmm_46 = torch.ops.aten.addmm.default(primals_150, view_183, permute_112);  primals_150 = view_183 = None
        view_184 = torch.ops.aten.view.default(addmm_46, [50, 2, 3072])
        mul_77 = torch.ops.aten.mul.Tensor(view_184, 1.702)
        sigmoid_11 = torch.ops.aten.sigmoid.default(mul_77);  mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(view_184, sigmoid_11);  view_184 = sigmoid_11 = None
        view_185 = torch.ops.aten.view.default(mul_78, [100, 3072]);  mul_78 = None
        permute_113 = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        addmm_47 = torch.ops.aten.addmm.default(primals_152, view_185, permute_113);  primals_152 = view_185 = None
        view_186 = torch.ops.aten.view.default(addmm_47, [50, 2, 768]);  addmm_47 = None
        add_86 = torch.ops.aten.add.Tensor(add_83, view_186);  add_83 = view_186 = None
        permute_114 = torch.ops.aten.permute.default(add_86, [1, 0, 2]);  add_86 = None
        select_36 = torch.ops.aten.select.int(permute_114, 1, 0);  permute_114 = None
        clone_50 = torch.ops.aten.clone.default(select_36, memory_format = torch.contiguous_format);  select_36 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_50, [1], correction = 0, keepdim = True)
        getitem_98 = var_mean_25[0]
        getitem_99 = var_mean_25[1];  var_mean_25 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_31 = torch.ops.aten.sub.Tensor(clone_50, getitem_99);  clone_50 = getitem_99 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_25);  sub_31 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, primals_153)
        add_88 = torch.ops.aten.add.Tensor(mul_80, primals_154);  mul_80 = primals_154 = None
        mm = torch.ops.aten.mm.default(add_88, primals_155);  add_88 = None
        slice_6 = torch.ops.aten.slice.Tensor(permute_40, 1, 1, 9223372036854775807);  permute_40 = None
        permute_115 = torch.ops.aten.permute.default(slice_6, [0, 2, 1]);  slice_6 = None
        view_187 = torch.ops.aten.view.default(permute_115, [2, 768, 7, 7]);  permute_115 = None
        slice_9 = torch.ops.aten.slice.Tensor(permute_77, 1, 1, 9223372036854775807);  permute_77 = None
        permute_116 = torch.ops.aten.permute.default(slice_9, [0, 2, 1]);  slice_9 = None
        view_188 = torch.ops.aten.view.default(permute_116, [2, 768, 7, 7]);  permute_116 = None
        view_189 = torch.ops.aten.view.default(primals_156, [256, -1])
        permute_117 = torch.ops.aten.permute.default(view_189, [1, 0])
        mul_81 = torch.ops.aten.mul.Tensor(permute_117, primals_157);  permute_117 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_81, [1]);  mul_81 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 2.0)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [0], True);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min_4 = torch.ops.aten.clamp_min.default(pow_2, 1e-12);  pow_2 = None
        expand_7 = torch.ops.aten.expand.default(clamp_min_4, [6912]);  clamp_min_4 = None
        div_1 = torch.ops.aten.div.Tensor(sum_1, expand_7);  sum_1 = expand_7 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_189, div_1);  view_189 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_82, [1]);  mul_82 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sum_3, 2.0)
        sum_4 = torch.ops.aten.sum.dim_IntList(pow_3, [0], True);  pow_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        clamp_min_5 = torch.ops.aten.clamp_min.default(pow_4, 1e-12);  pow_4 = None
        expand_9 = torch.ops.aten.expand.default(clamp_min_5, [256]);  clamp_min_5 = None
        div_2 = torch.ops.aten.div.Tensor(sum_3, expand_9);  expand_9 = None
        mul_84 = torch.ops.aten.mul.Tensor(div_2, sum_3);  sum_3 = None
        sum_6 = torch.ops.aten.sum.default(mul_84);  mul_84 = None
        div_3 = torch.ops.aten.div.Tensor(primals_156, sum_6)
        convolution_1 = torch.ops.aten.convolution.default(view_187, div_3, primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  view_187 = primals_159 = None
        gt = torch.ops.aten.gt.Scalar(convolution_1, 0)
        mul_85 = torch.ops.aten.mul.Tensor(convolution_1, 0.2)
        where = torch.ops.aten.where.self(gt, convolution_1, mul_85);  gt = convolution_1 = mul_85 = None
        constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(where, [1, 1, 1, 1], 0.0)
        convolution_2 = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 256)
        view_190 = torch.ops.aten.view.default(primals_161, [1, -1])
        permute_118 = torch.ops.aten.permute.default(view_190, [1, 0])
        mul_86 = torch.ops.aten.mul.Tensor(permute_118, primals_162);  permute_118 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_86, [1]);  mul_86 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sum_7, 2.0)
        sum_8 = torch.ops.aten.sum.dim_IntList(pow_5, [0], True);  pow_5 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None
        clamp_min_6 = torch.ops.aten.clamp_min.default(pow_6, 1e-12);  pow_6 = None
        expand_11 = torch.ops.aten.expand.default(clamp_min_6, [256])
        div_4 = torch.ops.aten.div.Tensor(sum_7, expand_11);  sum_7 = expand_11 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_190, div_4);  view_190 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_87, [1]);  mul_87 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sum_9, 2.0)
        sum_10 = torch.ops.aten.sum.dim_IntList(pow_7, [0], True);  pow_7 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(pow_8, 1e-12);  pow_8 = None
        expand_13 = torch.ops.aten.expand.default(clamp_min_7, [1]);  clamp_min_7 = None
        div_5 = torch.ops.aten.div.Tensor(sum_9, expand_13);  expand_13 = None
        mul_89 = torch.ops.aten.mul.Tensor(div_5, sum_9)
        sum_12 = torch.ops.aten.sum.default(mul_89);  mul_89 = None
        div_6 = torch.ops.aten.div.Tensor(primals_161, sum_12);  sum_12 = None
        convolution_3 = torch.ops.aten.convolution.default(convolution_2, div_6, primals_164, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(convolution_3, 1)
        view_191 = torch.ops.aten.view.default(primals_165, [256, -1])
        permute_119 = torch.ops.aten.permute.default(view_191, [1, 0])
        mul_90 = torch.ops.aten.mul.Tensor(permute_119, primals_166);  permute_119 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_90, [1]);  mul_90 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sum_13, 2.0)
        sum_14 = torch.ops.aten.sum.dim_IntList(pow_9, [0], True);  pow_9 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None
        clamp_min_8 = torch.ops.aten.clamp_min.default(pow_10, 1e-12);  pow_10 = None
        expand_15 = torch.ops.aten.expand.default(clamp_min_8, [6912]);  clamp_min_8 = None
        div_7 = torch.ops.aten.div.Tensor(sum_13, expand_15);  sum_13 = expand_15 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_191, div_7);  view_191 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_91, [1]);  mul_91 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(sum_15, 2.0)
        sum_16 = torch.ops.aten.sum.dim_IntList(pow_11, [0], True);  pow_11 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(sum_16, 0.5);  sum_16 = None
        clamp_min_9 = torch.ops.aten.clamp_min.default(pow_12, 1e-12);  pow_12 = None
        expand_17 = torch.ops.aten.expand.default(clamp_min_9, [256]);  clamp_min_9 = None
        div_8 = torch.ops.aten.div.Tensor(sum_15, expand_17);  expand_17 = None
        mul_93 = torch.ops.aten.mul.Tensor(div_8, sum_15);  sum_15 = None
        sum_18 = torch.ops.aten.sum.default(mul_93);  mul_93 = None
        div_9 = torch.ops.aten.div.Tensor(primals_165, sum_18)
        convolution_4 = torch.ops.aten.convolution.default(view_188, div_9, primals_168, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  view_188 = primals_168 = None
        gt_1 = torch.ops.aten.gt.Scalar(convolution_4, 0)
        mul_94 = torch.ops.aten.mul.Tensor(convolution_4, 0.2)
        where_1 = torch.ops.aten.where.self(gt_1, convolution_4, mul_94);  gt_1 = convolution_4 = mul_94 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(where_1, [1, 1, 1, 1], 0.0)
        convolution_5 = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 256)
        view_192 = torch.ops.aten.view.default(primals_170, [1, -1])
        permute_120 = torch.ops.aten.permute.default(view_192, [1, 0])
        mul_95 = torch.ops.aten.mul.Tensor(permute_120, primals_171);  permute_120 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_95, [1]);  mul_95 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(sum_19, 2.0)
        sum_20 = torch.ops.aten.sum.dim_IntList(pow_13, [0], True);  pow_13 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(sum_20, 0.5);  sum_20 = None
        clamp_min_10 = torch.ops.aten.clamp_min.default(pow_14, 1e-12);  pow_14 = None
        expand_19 = torch.ops.aten.expand.default(clamp_min_10, [256])
        div_10 = torch.ops.aten.div.Tensor(sum_19, expand_19);  sum_19 = expand_19 = None
        mul_96 = torch.ops.aten.mul.Tensor(view_192, div_10);  view_192 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_96, [1]);  mul_96 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(sum_21, 2.0)
        sum_22 = torch.ops.aten.sum.dim_IntList(pow_15, [0], True);  pow_15 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
        clamp_min_11 = torch.ops.aten.clamp_min.default(pow_16, 1e-12);  pow_16 = None
        expand_21 = torch.ops.aten.expand.default(clamp_min_11, [1]);  clamp_min_11 = None
        div_11 = torch.ops.aten.div.Tensor(sum_21, expand_21);  expand_21 = None
        mul_98 = torch.ops.aten.mul.Tensor(div_11, sum_21)
        sum_24 = torch.ops.aten.sum.default(mul_98);  mul_98 = None
        div_12 = torch.ops.aten.div.Tensor(primals_170, sum_24);  sum_24 = None
        convolution_6 = torch.ops.aten.convolution.default(convolution_5, div_12, primals_173, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(convolution_6, 1)
        view_193 = torch.ops.aten.view.default(primals_174, [256, -1])
        permute_121 = torch.ops.aten.permute.default(view_193, [1, 0])
        mul_99 = torch.ops.aten.mul.Tensor(permute_121, primals_175);  permute_121 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_99, [1]);  mul_99 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(sum_25, 2.0)
        sum_26 = torch.ops.aten.sum.dim_IntList(pow_17, [0], True);  pow_17 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(sum_26, 0.5);  sum_26 = None
        clamp_min_12 = torch.ops.aten.clamp_min.default(pow_18, 1e-12);  pow_18 = None
        expand_23 = torch.ops.aten.expand.default(clamp_min_12, [512]);  clamp_min_12 = None
        div_13 = torch.ops.aten.div.Tensor(sum_25, expand_23);  sum_25 = expand_23 = None
        mul_100 = torch.ops.aten.mul.Tensor(view_193, div_13);  view_193 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_100, [1]);  mul_100 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(sum_27, 2.0)
        sum_28 = torch.ops.aten.sum.dim_IntList(pow_19, [0], True);  pow_19 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(sum_28, 0.5);  sum_28 = None
        clamp_min_13 = torch.ops.aten.clamp_min.default(pow_20, 1e-12);  pow_20 = None
        expand_25 = torch.ops.aten.expand.default(clamp_min_13, [256]);  clamp_min_13 = None
        div_14 = torch.ops.aten.div.Tensor(sum_27, expand_25);  expand_25 = None
        mul_102 = torch.ops.aten.mul.Tensor(div_14, sum_27);  sum_27 = None
        sum_30 = torch.ops.aten.sum.default(mul_102);  mul_102 = None
        div_15 = torch.ops.aten.div.Tensor(primals_174, sum_30)
        permute_122 = torch.ops.aten.permute.default(div_15, [1, 0])
        addmm_48 = torch.ops.aten.addmm.default(primals_177, mm, permute_122);  primals_177 = None
        gt_2 = torch.ops.aten.gt.Scalar(addmm_48, 0)
        mul_103 = torch.ops.aten.mul.Tensor(addmm_48, 0.2)
        where_2 = torch.ops.aten.where.self(gt_2, addmm_48, mul_103);  gt_2 = addmm_48 = mul_103 = None
        view_194 = torch.ops.aten.view.default(primals_178, [1, -1])
        permute_123 = torch.ops.aten.permute.default(view_194, [1, 0])
        mul_104 = torch.ops.aten.mul.Tensor(permute_123, primals_179);  permute_123 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_104, [1]);  mul_104 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(sum_31, 2.0)
        sum_32 = torch.ops.aten.sum.dim_IntList(pow_21, [0], True);  pow_21 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(sum_32, 0.5);  sum_32 = None
        clamp_min_14 = torch.ops.aten.clamp_min.default(pow_22, 1e-12);  pow_22 = None
        expand_27 = torch.ops.aten.expand.default(clamp_min_14, [256])
        div_16 = torch.ops.aten.div.Tensor(sum_31, expand_27);  sum_31 = expand_27 = None
        mul_105 = torch.ops.aten.mul.Tensor(view_194, div_16);  view_194 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_105, [1]);  mul_105 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(sum_33, 2.0)
        sum_34 = torch.ops.aten.sum.dim_IntList(pow_23, [0], True);  pow_23 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
        clamp_min_15 = torch.ops.aten.clamp_min.default(pow_24, 1e-12);  pow_24 = None
        expand_29 = torch.ops.aten.expand.default(clamp_min_15, [1]);  clamp_min_15 = None
        div_17 = torch.ops.aten.div.Tensor(sum_33, expand_29);  expand_29 = None
        mul_107 = torch.ops.aten.mul.Tensor(div_17, sum_33)
        sum_36 = torch.ops.aten.sum.default(mul_107);  mul_107 = None
        div_18 = torch.ops.aten.div.Tensor(primals_178, sum_36);  sum_36 = None
        permute_124 = torch.ops.aten.permute.default(div_18, [1, 0])
        addmm_49 = torch.ops.aten.addmm.default(primals_181, where_2, permute_124);  primals_181 = None
        full_default_4 = torch.ops.aten.full.default([2, 3, 3], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_109 = torch.ops.aten.mul.Tensor(full_default_4, squeeze_12)
        full_default_5 = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum = torch.ops.aten.minimum.default(full_default_5, squeeze_12)
        abs_1 = torch.ops.aten.abs.default(squeeze_12);  squeeze_12 = None
        neg = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        exp = torch.ops.aten.exp.default(neg);  neg = None
        log1p = torch.ops.aten.log1p.default(exp);  exp = None
        sub_33 = torch.ops.aten.sub.Tensor(minimum, log1p);  minimum = log1p = None
        sub_34 = torch.ops.aten.sub.Tensor(mul_109, sub_33);  mul_109 = sub_33 = None
        mean_2 = torch.ops.aten.mean.dim(sub_34, [1, 2]);  sub_34 = None
        view_195 = torch.ops.aten.view.default(mean_2, [-1, 1]);  mean_2 = None
        add_89 = torch.ops.aten.add.Tensor(view_195, 0);  view_195 = None
        mul_110 = torch.ops.aten.mul.Tensor(full_default_4, squeeze_13);  full_default_4 = None
        minimum_1 = torch.ops.aten.minimum.default(full_default_5, squeeze_13)
        abs_2 = torch.ops.aten.abs.default(squeeze_13);  squeeze_13 = None
        neg_1 = torch.ops.aten.neg.default(abs_2);  abs_2 = None
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        log1p_1 = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
        sub_36 = torch.ops.aten.sub.Tensor(minimum_1, log1p_1);  minimum_1 = log1p_1 = None
        sub_37 = torch.ops.aten.sub.Tensor(mul_110, sub_36);  mul_110 = sub_36 = None
        mean_3 = torch.ops.aten.mean.dim(sub_37, [1, 2]);  sub_37 = None
        view_196 = torch.ops.aten.view.default(mean_3, [-1, 1]);  mean_3 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, view_196);  add_89 = view_196 = None
        full_default_10 = torch.ops.aten.full.default([2, 1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_111 = torch.ops.aten.mul.Tensor(full_default_10, addmm_49);  full_default_10 = None
        minimum_2 = torch.ops.aten.minimum.default(full_default_5, addmm_49);  full_default_5 = None
        abs_3 = torch.ops.aten.abs.default(addmm_49)
        neg_2 = torch.ops.aten.neg.default(abs_3);  abs_3 = None
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        log1p_2 = torch.ops.aten.log1p.default(exp_2);  exp_2 = None
        sub_39 = torch.ops.aten.sub.Tensor(minimum_2, log1p_2);  minimum_2 = log1p_2 = None
        sub_40 = torch.ops.aten.sub.Tensor(mul_111, sub_39);  mul_111 = sub_39 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, sub_40);  add_90 = sub_40 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, 0);  add_91 = None
        permute_125 = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_129 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        gt_4 = torch.ops.aten.gt.Scalar(where_1, 0);  where_1 = None
        gt_5 = torch.ops.aten.gt.Scalar(where, 0);  where = None
        permute_135 = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        div_39 = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        permute_137 = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        permute_138 = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        div_40 = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_139 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_145 = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        div_41 = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_146 = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        permute_147 = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        div_42 = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_148 = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        permute_154 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        div_43 = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_155 = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        permute_156 = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        div_44 = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_157 = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        permute_163 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        div_45 = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_164 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_165 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        div_46 = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_166 = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        permute_172 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_174 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_175 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_48 = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_176 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_182 = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        div_49 = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_183 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_184 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_50 = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_185 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_191 = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        div_51 = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_192 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_193 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        div_52 = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_194 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_200 = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        div_53 = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_201 = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_202 = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        div_54 = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_203 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_209 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_211 = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_212 = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        div_56 = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_213 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_219 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_57 = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_220 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_221 = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        div_58 = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_222 = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        permute_228 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_59 = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_229 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_230 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_60 = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_231 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_237 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        div_61 = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_238 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_239 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_62 = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_240 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_246 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        copy_ = torch.ops.aten.copy_.default(primals_157, div_2);  primals_157 = copy_ = None
        copy__1 = torch.ops.aten.copy_.default(primals_158, div_1);  primals_158 = copy__1 = None
        copy__2 = torch.ops.aten.copy_.default(primals_162, div_5);  div_5 = copy__2 = None
        copy__3 = torch.ops.aten.copy_.default(primals_163, div_4);  primals_163 = div_4 = copy__3 = None
        copy__4 = torch.ops.aten.copy_.default(primals_166, div_8);  primals_166 = copy__4 = None
        copy__5 = torch.ops.aten.copy_.default(primals_167, div_7);  primals_167 = copy__5 = None
        copy__6 = torch.ops.aten.copy_.default(primals_171, div_11);  div_11 = copy__6 = None
        copy__7 = torch.ops.aten.copy_.default(primals_172, div_10);  primals_172 = div_10 = copy__7 = None
        copy__8 = torch.ops.aten.copy_.default(primals_175, div_14);  primals_175 = copy__8 = None
        copy__9 = torch.ops.aten.copy_.default(primals_176, div_13);  primals_176 = copy__9 = None
        copy__10 = torch.ops.aten.copy_.default(primals_179, div_17);  div_17 = copy__10 = None
        copy__11 = torch.ops.aten.copy_.default(primals_180, div_16);  primals_180 = div_16 = copy__11 = None
        return (add_92, div_3, div_6, div_9, div_12, div_15, div_18, primals_4, primals_6, primals_7, primals_8, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, inductor_random_default_1, inductor_random_default, view, clamp_max, clamp_max_1, index_put, add_10, device_put_1, div, cat, getitem_1, rsqrt, getitem_3, rsqrt_1, view_13, view_14, view_15, getitem_4, getitem_5, getitem_6, getitem_7, mul_9, addmm_2, mul_13, view_28, view_29, view_30, getitem_12, getitem_13, getitem_14, getitem_15, mul_15, addmm_6, mul_19, view_43, view_44, view_45, getitem_20, getitem_21, getitem_22, getitem_23, mul_21, addmm_10, mul_25, view_58, view_59, view_60, getitem_28, getitem_29, getitem_30, getitem_31, mul_27, addmm_14, add_38, getitem_35, rsqrt_9, view_73, view_74, view_75, getitem_36, getitem_37, getitem_38, getitem_39, mul_33, addmm_18, mul_37, view_88, view_89, view_90, getitem_44, getitem_45, getitem_46, getitem_47, mul_39, addmm_22, mul_43, view_103, view_104, view_105, getitem_52, getitem_53, getitem_54, getitem_55, mul_45, addmm_26, mul_49, view_118, view_119, view_120, getitem_60, getitem_61, getitem_62, getitem_63, mul_51, addmm_30, add_62, getitem_67, rsqrt_17, view_133, view_134, view_135, getitem_68, getitem_69, getitem_70, getitem_71, mul_57, addmm_34, mul_61, view_148, view_149, view_150, getitem_76, getitem_77, getitem_78, getitem_79, mul_63, addmm_38, mul_67, view_163, view_164, view_165, getitem_84, getitem_85, getitem_86, getitem_87, mul_69, addmm_42, mul_73, view_178, view_179, view_180, getitem_92, getitem_93, getitem_94, getitem_95, mul_75, addmm_46, mul_79, mm, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, permute_129, gt_4, gt_5, permute_135, div_39, permute_137, permute_138, div_40, permute_139, permute_145, div_41, permute_146, permute_147, div_42, permute_148, permute_154, div_43, permute_155, permute_156, div_44, permute_157, permute_163, div_45, permute_164, permute_165, div_46, permute_166, permute_172, permute_174, permute_175, div_48, permute_176, permute_182, div_49, permute_183, permute_184, div_50, permute_185, permute_191, div_51, permute_192, permute_193, div_52, permute_194, permute_200, div_53, permute_201, permute_202, div_54, permute_203, permute_209, permute_211, permute_212, div_56, permute_213, permute_219, div_57, permute_220, permute_221, div_58, permute_222, permute_228, div_59, permute_229, permute_230, div_60, permute_231, permute_237, div_61, permute_238, permute_239, div_62, permute_240, permute_246)
        
def load_args(reader):
    buf0 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2, 3, 256, 256), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 12)
    reader.tensor(buf1, (3,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 12)
    reader.tensor(buf2, (3,), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768, 3, 32, 32), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf5, (50, 768), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf10, (2304,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2304, 768), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 768), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072, 768), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf17, (3072,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 3072), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf22, (2304,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf23, (2304, 768), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 768), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (3072, 768), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf29, (3072,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 3072), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf34, (2304,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf35, (2304, 768), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 768), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf40, (3072, 768), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf41, (3072,), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 3072), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2304,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf47, (2304, 768), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 768), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3072, 768), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (3072,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 3072), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf58, (2304,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf59, (2304, 768), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 768), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3072, 768), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf65, (3072,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 3072), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf70, (2304,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2304, 768), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 768), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (3072, 768), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf77, (3072,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 3072), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf82, (2304,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf83, (2304, 768), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768, 768), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf88, (3072, 768), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf89, (3072,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 3072), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2304,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf95, (2304, 768), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 768), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (3072, 768), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (3072,), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 3072), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf106, (2304,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf107, (2304, 768), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 768), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf112, (3072, 768), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf113, (3072,), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 3072), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf118, (2304,), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf119, (2304, 768), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 768), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf124, (3072, 768), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf125, (3072,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 3072), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf130, (2304,), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf131, (2304, 768), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 768), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf136, (3072, 768), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf137, (3072,), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 3072), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf142, (2304,), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf143, (2304, 768), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768, 768), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf148, (3072, 768), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf149, (3072,), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768, 3072), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768,), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 512), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf155, (256, 768, 3, 3), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf157, (6912,), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256,), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf159, (256, 1, 4, 4), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf160, (1, 256, 1, 1), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1,), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf162, (256,), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1,), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf164, (256, 768, 3, 3), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf165, (256,), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf166, (6912,), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf167, (256,), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf168, (256, 1, 4, 4), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1, 256, 1, 1), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1,), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256,), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1,), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256, 512), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf174, (256,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256,), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1, 256), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1,), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf179, (256,), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1,), is_leaf=True)  # primals_181
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)