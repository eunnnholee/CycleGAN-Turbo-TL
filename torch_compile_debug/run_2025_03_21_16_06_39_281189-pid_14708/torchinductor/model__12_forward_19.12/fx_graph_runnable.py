
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
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch.distributions', 'torch._refs', 'torch.testing', 'torch._prims', 'torch._decomp'}
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'skipfiles_inline_module_allowlist', 'constant_functions', 'repro_after', 'repro_level'}
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238):
        cat = torch.ops.aten.cat.default([primals_4, primals_3, primals_2, primals_1]);  primals_4 = primals_3 = primals_2 = primals_1 = None
        view = torch.ops.aten.view.default(cat, [4, 4, 32, 32]);  cat = None
        div = torch.ops.aten.div.Tensor(view, 0.18215);  view = None
        convolution = torch.ops.aten.convolution.default(div, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
        convolution_1 = torch.ops.aten.convolution.default(convolution, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
        convolution_2 = torch.ops.aten.convolution.default(convolution, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_3 = torch.ops.aten.convolution.default(convolution_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul = torch.ops.aten.mul.Tensor(convolution_3, 2.0);  convolution_3 = None
        add = torch.ops.aten.add.Tensor(convolution_1, mul);  convolution_1 = mul = None
        view_1 = torch.ops.aten.view.default(add, [4, 32, 16, 1024])
        var_mean = torch.ops.aten.var_mean.correction(view_1, [2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(view_1, getitem_1);  view_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_2 = torch.ops.aten.view.default(mul_1, [4, 512, 32, 32]);  mul_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_176, 0)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(primals_175, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_2, unsqueeze_5);  view_2 = unsqueeze_5 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_2);  mul_2 = unsqueeze_2 = None
        sigmoid = torch.ops.aten.sigmoid.default(add_2)
        mul_3 = torch.ops.aten.mul.Tensor(add_2, sigmoid);  add_2 = sigmoid = None
        convolution_4 = torch.ops.aten.convolution.default(mul_3, primals_177, primals_178, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_178 = None
        convolution_5 = torch.ops.aten.convolution.default(mul_3, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_6 = torch.ops.aten.convolution.default(convolution_5, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_4 = torch.ops.aten.mul.Tensor(convolution_6, 2.0);  convolution_6 = None
        add_3 = torch.ops.aten.add.Tensor(convolution_4, mul_4);  convolution_4 = mul_4 = None
        view_3 = torch.ops.aten.view.default(add_3, [4, 32, 16, 1024])
        var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [2, 3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        view_4 = torch.ops.aten.view.default(mul_5, [4, 512, 32, 32]);  mul_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_182, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(primals_181, 0)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_6 = torch.ops.aten.mul.Tensor(view_4, unsqueeze_11);  view_4 = unsqueeze_11 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_8);  mul_6 = unsqueeze_8 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(add_5)
        mul_7 = torch.ops.aten.mul.Tensor(add_5, sigmoid_1);  add_5 = sigmoid_1 = None
        convolution_7 = torch.ops.aten.convolution.default(mul_7, primals_183, primals_184, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_184 = None
        convolution_8 = torch.ops.aten.convolution.default(mul_7, primals_185, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_9 = torch.ops.aten.convolution.default(convolution_8, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_8 = torch.ops.aten.mul.Tensor(convolution_9, 2.0);  convolution_9 = None
        add_6 = torch.ops.aten.add.Tensor(convolution_7, mul_8);  convolution_7 = mul_8 = None
        add_7 = torch.ops.aten.add.Tensor(add, add_6);  add_6 = None
        div_1 = torch.ops.aten.div.Tensor(add_7, 1);  add_7 = None
        view_5 = torch.ops.aten.view.default(div_1, [4, 512, 1024])
        view_6 = torch.ops.aten.view.default(view_5, [4, 32, 16, 1024])
        var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2 = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        view_7 = torch.ops.aten.view.default(mul_9, [4, 512, 1024]);  mul_9 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(primals_188, 0);  primals_188 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(primals_187, 0)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_7, unsqueeze_15);  view_7 = unsqueeze_15 = None
        add_9 = torch.ops.aten.add.Tensor(mul_10, unsqueeze_13);  mul_10 = unsqueeze_13 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(rsqrt_2, [2, 3]);  rsqrt_2 = None
        permute_2 = torch.ops.aten.permute.default(add_9, [0, 2, 1]);  add_9 = None
        permute_3 = torch.ops.aten.permute.default(primals_189, [1, 0])
        expand = torch.ops.aten.expand.default(permute_2, [4, 1024, 512])
        expand_1 = torch.ops.aten.expand.default(permute_3, [4, 512, 512]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(expand, expand_1);  expand_1 = None
        add_10 = torch.ops.aten.add.Tensor(bmm, primals_190);  bmm = primals_190 = None
        permute_4 = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_11 = torch.ops.aten.view.default(clone_1, [4096, 512]);  clone_1 = None
        mm = torch.ops.aten.mm.default(view_11, permute_4)
        permute_5 = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        mm_1 = torch.ops.aten.mm.default(mm, permute_5)
        view_14 = torch.ops.aten.view.default(mm_1, [4, 1024, 512]);  mm_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_14, 2.0);  view_14 = None
        add_11 = torch.ops.aten.add.Tensor(add_10, mul_11);  add_10 = mul_11 = None
        permute_6 = torch.ops.aten.permute.default(primals_193, [1, 0])
        expand_3 = torch.ops.aten.expand.default(permute_6, [4, 512, 512]);  permute_6 = None
        bmm_1 = torch.ops.aten.bmm.default(expand, expand_3);  expand_3 = None
        add_12 = torch.ops.aten.add.Tensor(bmm_1, primals_194);  bmm_1 = primals_194 = None
        permute_7 = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        mm_2 = torch.ops.aten.mm.default(view_11, permute_7)
        permute_8 = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        mm_3 = torch.ops.aten.mm.default(mm_2, permute_8)
        view_21 = torch.ops.aten.view.default(mm_3, [4, 1024, 512]);  mm_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_21, 2.0);  view_21 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, mul_12);  add_12 = mul_12 = None
        permute_9 = torch.ops.aten.permute.default(primals_197, [1, 0])
        expand_5 = torch.ops.aten.expand.default(permute_9, [4, 512, 512]);  permute_9 = None
        bmm_2 = torch.ops.aten.bmm.default(expand, expand_5);  expand = expand_5 = None
        add_14 = torch.ops.aten.add.Tensor(bmm_2, primals_198);  bmm_2 = primals_198 = None
        permute_10 = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
        mm_4 = torch.ops.aten.mm.default(view_11, permute_10)
        permute_11 = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        mm_5 = torch.ops.aten.mm.default(mm_4, permute_11)
        view_28 = torch.ops.aten.view.default(mm_5, [4, 1024, 512]);  mm_5 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_28, 2.0);  view_28 = None
        add_15 = torch.ops.aten.add.Tensor(add_14, mul_13);  add_14 = mul_13 = None
        view_32 = torch.ops.aten.view.default(add_11, [4, -1, 1, 512]);  add_11 = None
        permute_15 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        view_33 = torch.ops.aten.view.default(add_13, [4, -1, 1, 512]);  add_13 = None
        permute_16 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        view_34 = torch.ops.aten.view.default(add_15, [4, -1, 1, 512]);  add_15 = None
        permute_17 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_15, permute_16, permute_17, None, True)
        getitem_6 = _scaled_dot_product_efficient_attention[0]
        getitem_7 = _scaled_dot_product_efficient_attention[1]
        getitem_8 = _scaled_dot_product_efficient_attention[2]
        getitem_9 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_18 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3])
        view_35 = torch.ops.aten.view.default(permute_18, [4, -1, 512]);  permute_18 = None
        view_36 = torch.ops.aten.view.default(view_35, [4096, 512]);  view_35 = None
        permute_19 = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        addmm = torch.ops.aten.addmm.default(primals_202, view_36, permute_19);  primals_202 = None
        view_37 = torch.ops.aten.view.default(addmm, [4, 1024, 512]);  addmm = None
        permute_20 = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        mm_6 = torch.ops.aten.mm.default(view_36, permute_20);  view_36 = None
        permute_21 = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        mm_7 = torch.ops.aten.mm.default(mm_6, permute_21)
        view_41 = torch.ops.aten.view.default(mm_7, [4, 1024, 512]);  mm_7 = None
        mul_14 = torch.ops.aten.mul.Tensor(view_41, 2.0);  view_41 = None
        add_16 = torch.ops.aten.add.Tensor(view_37, mul_14);  view_37 = mul_14 = None
        permute_22 = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
        view_45 = torch.ops.aten.view.default(permute_22, [4, 512, 32, 32]);  permute_22 = None
        add_17 = torch.ops.aten.add.Tensor(view_45, div_1);  view_45 = div_1 = None
        div_2 = torch.ops.aten.div.Tensor(add_17, 1);  add_17 = None
        clone_5 = torch.ops.aten.clone.default(div_2, memory_format = torch.contiguous_format)
        view_46 = torch.ops.aten.view.default(clone_5, [4, 32, 16, 1024])
        var_mean_3 = torch.ops.aten.var_mean.correction(view_46, [2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3 = torch.ops.aten.sub.Tensor(view_46, getitem_11);  view_46 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        view_47 = torch.ops.aten.view.default(mul_15, [4, 512, 32, 32]);  mul_15 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(primals_206, 0)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(primals_205, 0)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        mul_16 = torch.ops.aten.mul.Tensor(view_47, unsqueeze_21);  view_47 = unsqueeze_21 = None
        add_19 = torch.ops.aten.add.Tensor(mul_16, unsqueeze_18);  mul_16 = unsqueeze_18 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(add_19)
        mul_17 = torch.ops.aten.mul.Tensor(add_19, sigmoid_2);  add_19 = sigmoid_2 = None
        convolution_10 = torch.ops.aten.convolution.default(mul_17, primals_207, primals_208, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_208 = None
        convolution_11 = torch.ops.aten.convolution.default(mul_17, primals_209, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_12 = torch.ops.aten.convolution.default(convolution_11, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_18 = torch.ops.aten.mul.Tensor(convolution_12, 2.0);  convolution_12 = None
        add_20 = torch.ops.aten.add.Tensor(convolution_10, mul_18);  convolution_10 = mul_18 = None
        view_48 = torch.ops.aten.view.default(add_20, [4, 32, 16, 1024])
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_4[0]
        getitem_13 = var_mean_4[1];  var_mean_4 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4 = torch.ops.aten.sub.Tensor(view_48, getitem_13);  view_48 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        view_49 = torch.ops.aten.view.default(mul_19, [4, 512, 32, 32]);  mul_19 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(primals_212, 0)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(unsqueeze_23, 3);  unsqueeze_23 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(primals_211, 0)
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(unsqueeze_25, 2);  unsqueeze_25 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_49, unsqueeze_27);  view_49 = unsqueeze_27 = None
        add_22 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_24);  mul_20 = unsqueeze_24 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(add_22)
        mul_21 = torch.ops.aten.mul.Tensor(add_22, sigmoid_3);  add_22 = sigmoid_3 = None
        convolution_13 = torch.ops.aten.convolution.default(mul_21, primals_213, primals_214, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_214 = None
        convolution_14 = torch.ops.aten.convolution.default(mul_21, primals_215, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_15 = torch.ops.aten.convolution.default(convolution_14, primals_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_22 = torch.ops.aten.mul.Tensor(convolution_15, 2.0);  convolution_15 = None
        add_23 = torch.ops.aten.add.Tensor(convolution_13, mul_22);  convolution_13 = mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(div_2, add_23);  div_2 = add_23 = None
        div_3 = torch.ops.aten.div.Tensor(add_24, 1);  add_24 = None
        mul_23 = torch.ops.aten.mul.Tensor(primals_217, 1);  primals_217 = None
        convolution_16 = torch.ops.aten.convolution.default(mul_23, primals_218, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_17 = torch.ops.aten.convolution.default(mul_23, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_18 = torch.ops.aten.convolution.default(convolution_17, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_24 = torch.ops.aten.mul.Tensor(convolution_18, 2.0);  convolution_18 = None
        add_25 = torch.ops.aten.add.Tensor(convolution_16, mul_24);  convolution_16 = mul_24 = None
        add_26 = torch.ops.aten.add.Tensor(div_3, add_25);  div_3 = add_25 = None
        clone_7 = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        view_50 = torch.ops.aten.view.default(clone_7, [4, 32, 16, 1024])
        var_mean_5 = torch.ops.aten.var_mean.correction(view_50, [2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_5[0]
        getitem_15 = var_mean_5[1];  var_mean_5 = None
        add_27 = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5 = torch.ops.aten.sub.Tensor(view_50, getitem_15);  view_50 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        view_51 = torch.ops.aten.view.default(mul_25, [4, 512, 32, 32]);  mul_25 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(primals_12, 0)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(unsqueeze_29, 3);  unsqueeze_29 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(primals_11, 0)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(unsqueeze_31, 2);  unsqueeze_31 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 3);  unsqueeze_32 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_51, unsqueeze_33);  view_51 = unsqueeze_33 = None
        add_28 = torch.ops.aten.add.Tensor(mul_26, unsqueeze_30);  mul_26 = unsqueeze_30 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(add_28)
        mul_27 = torch.ops.aten.mul.Tensor(add_28, sigmoid_4);  add_28 = sigmoid_4 = None
        convolution_19 = torch.ops.aten.convolution.default(mul_27, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_14 = None
        convolution_20 = torch.ops.aten.convolution.default(mul_27, primals_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_21 = torch.ops.aten.convolution.default(convolution_20, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_28 = torch.ops.aten.mul.Tensor(convolution_21, 2.0);  convolution_21 = None
        add_29 = torch.ops.aten.add.Tensor(convolution_19, mul_28);  convolution_19 = mul_28 = None
        view_52 = torch.ops.aten.view.default(add_29, [4, 32, 16, 1024])
        var_mean_6 = torch.ops.aten.var_mean.correction(view_52, [2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_6[0]
        getitem_17 = var_mean_6[1];  var_mean_6 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_6 = torch.ops.aten.sub.Tensor(view_52, getitem_17);  view_52 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        view_53 = torch.ops.aten.view.default(mul_29, [4, 512, 32, 32]);  mul_29 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(primals_18, 0)
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 2);  unsqueeze_34 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(unsqueeze_35, 3);  unsqueeze_35 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(primals_17, 0)
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(unsqueeze_37, 2);  unsqueeze_37 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, 3);  unsqueeze_38 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_53, unsqueeze_39);  view_53 = unsqueeze_39 = None
        add_31 = torch.ops.aten.add.Tensor(mul_30, unsqueeze_36);  mul_30 = unsqueeze_36 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(add_31)
        mul_31 = torch.ops.aten.mul.Tensor(add_31, sigmoid_5);  add_31 = sigmoid_5 = None
        convolution_22 = torch.ops.aten.convolution.default(mul_31, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_20 = None
        convolution_23 = torch.ops.aten.convolution.default(mul_31, primals_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_24 = torch.ops.aten.convolution.default(convolution_23, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_32 = torch.ops.aten.mul.Tensor(convolution_24, 2.0);  convolution_24 = None
        add_32 = torch.ops.aten.add.Tensor(convolution_22, mul_32);  convolution_22 = mul_32 = None
        add_33 = torch.ops.aten.add.Tensor(add_26, add_32);  add_26 = add_32 = None
        div_4 = torch.ops.aten.div.Tensor(add_33, 1.0);  add_33 = None
        clone_9 = torch.ops.aten.clone.default(div_4, memory_format = torch.contiguous_format)
        view_54 = torch.ops.aten.view.default(clone_9, [4, 32, 16, 1024])
        var_mean_7 = torch.ops.aten.var_mean.correction(view_54, [2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_7[0]
        getitem_19 = var_mean_7[1];  var_mean_7 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_7 = torch.ops.aten.sub.Tensor(view_54, getitem_19);  view_54 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        view_55 = torch.ops.aten.view.default(mul_33, [4, 512, 32, 32]);  mul_33 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(primals_24, 0)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(unsqueeze_41, 3);  unsqueeze_41 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(primals_23, 0)
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(unsqueeze_43, 2);  unsqueeze_43 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, 3);  unsqueeze_44 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_55, unsqueeze_45);  view_55 = unsqueeze_45 = None
        add_35 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_42);  mul_34 = unsqueeze_42 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(add_35)
        mul_35 = torch.ops.aten.mul.Tensor(add_35, sigmoid_6);  add_35 = sigmoid_6 = None
        convolution_25 = torch.ops.aten.convolution.default(mul_35, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_26 = None
        convolution_26 = torch.ops.aten.convolution.default(mul_35, primals_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_27 = torch.ops.aten.convolution.default(convolution_26, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_36 = torch.ops.aten.mul.Tensor(convolution_27, 2.0);  convolution_27 = None
        add_36 = torch.ops.aten.add.Tensor(convolution_25, mul_36);  convolution_25 = mul_36 = None
        view_56 = torch.ops.aten.view.default(add_36, [4, 32, 16, 1024])
        var_mean_8 = torch.ops.aten.var_mean.correction(view_56, [2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_8[0]
        getitem_21 = var_mean_8[1];  var_mean_8 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_8 = torch.ops.aten.sub.Tensor(view_56, getitem_21);  view_56 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        view_57 = torch.ops.aten.view.default(mul_37, [4, 512, 32, 32]);  mul_37 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(primals_30, 0)
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(unsqueeze_47, 3);  unsqueeze_47 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(primals_29, 0)
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(unsqueeze_49, 2);  unsqueeze_49 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, 3);  unsqueeze_50 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_57, unsqueeze_51);  view_57 = unsqueeze_51 = None
        add_38 = torch.ops.aten.add.Tensor(mul_38, unsqueeze_48);  mul_38 = unsqueeze_48 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(add_38)
        mul_39 = torch.ops.aten.mul.Tensor(add_38, sigmoid_7);  add_38 = sigmoid_7 = None
        convolution_28 = torch.ops.aten.convolution.default(mul_39, primals_31, primals_32, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_32 = None
        convolution_29 = torch.ops.aten.convolution.default(mul_39, primals_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_30 = torch.ops.aten.convolution.default(convolution_29, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_40 = torch.ops.aten.mul.Tensor(convolution_30, 2.0);  convolution_30 = None
        add_39 = torch.ops.aten.add.Tensor(convolution_28, mul_40);  convolution_28 = mul_40 = None
        add_40 = torch.ops.aten.add.Tensor(div_4, add_39);  div_4 = add_39 = None
        div_5 = torch.ops.aten.div.Tensor(add_40, 1.0);  add_40 = None
        clone_11 = torch.ops.aten.clone.default(div_5, memory_format = torch.contiguous_format)
        view_58 = torch.ops.aten.view.default(clone_11, [4, 32, 16, 1024])
        var_mean_9 = torch.ops.aten.var_mean.correction(view_58, [2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_9[0]
        getitem_23 = var_mean_9[1];  var_mean_9 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_9 = torch.ops.aten.sub.Tensor(view_58, getitem_23);  view_58 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        view_59 = torch.ops.aten.view.default(mul_41, [4, 512, 32, 32]);  mul_41 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(primals_36, 0)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(primals_35, 0)
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(unsqueeze_55, 2);  unsqueeze_55 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, 3);  unsqueeze_56 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_59, unsqueeze_57);  view_59 = unsqueeze_57 = None
        add_42 = torch.ops.aten.add.Tensor(mul_42, unsqueeze_54);  mul_42 = unsqueeze_54 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(add_42)
        mul_43 = torch.ops.aten.mul.Tensor(add_42, sigmoid_8);  add_42 = sigmoid_8 = None
        convolution_31 = torch.ops.aten.convolution.default(mul_43, primals_37, primals_38, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_38 = None
        convolution_32 = torch.ops.aten.convolution.default(mul_43, primals_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_33 = torch.ops.aten.convolution.default(convolution_32, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_44 = torch.ops.aten.mul.Tensor(convolution_33, 2.0);  convolution_33 = None
        add_43 = torch.ops.aten.add.Tensor(convolution_31, mul_44);  convolution_31 = mul_44 = None
        view_60 = torch.ops.aten.view.default(add_43, [4, 32, 16, 1024])
        var_mean_10 = torch.ops.aten.var_mean.correction(view_60, [2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_10[0]
        getitem_25 = var_mean_10[1];  var_mean_10 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_10 = torch.ops.aten.sub.Tensor(view_60, getitem_25);  view_60 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        view_61 = torch.ops.aten.view.default(mul_45, [4, 512, 32, 32]);  mul_45 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(primals_42, 0)
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, 2);  unsqueeze_58 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(primals_41, 0)
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, 3);  unsqueeze_62 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_61, unsqueeze_63);  view_61 = unsqueeze_63 = None
        add_45 = torch.ops.aten.add.Tensor(mul_46, unsqueeze_60);  mul_46 = unsqueeze_60 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(add_45)
        mul_47 = torch.ops.aten.mul.Tensor(add_45, sigmoid_9);  add_45 = sigmoid_9 = None
        convolution_34 = torch.ops.aten.convolution.default(mul_47, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_44 = None
        convolution_35 = torch.ops.aten.convolution.default(mul_47, primals_45, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_36 = torch.ops.aten.convolution.default(convolution_35, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_48 = torch.ops.aten.mul.Tensor(convolution_36, 2.0);  convolution_36 = None
        add_46 = torch.ops.aten.add.Tensor(convolution_34, mul_48);  convolution_34 = mul_48 = None
        add_47 = torch.ops.aten.add.Tensor(div_5, add_46);  div_5 = add_46 = None
        div_6 = torch.ops.aten.div.Tensor(add_47, 1.0);  add_47 = None
        iota = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_49 = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_48 = torch.ops.aten.add.Tensor(mul_49, 0);  mul_49 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        add_49 = torch.ops.aten.add.Tensor(convert_element_type, 0.0);  convert_element_type = None
        mul_50 = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_50, torch.int64);  mul_50 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1)
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(div_6, [None, None, unsqueeze_64, convert_element_type_1]);  div_6 = unsqueeze_64 = None
        clone_13 = torch.ops.aten.clone.default(_unsafe_index, memory_format = torch.channels_last);  _unsafe_index = None
        convolution_37 = torch.ops.aten.convolution.default(clone_13, primals_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_48 = None
        convolution_38 = torch.ops.aten.convolution.default(clone_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_39 = torch.ops.aten.convolution.default(convolution_38, primals_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_53 = torch.ops.aten.mul.Tensor(convolution_39, 2.0);  convolution_39 = None
        add_52 = torch.ops.aten.add.Tensor(convolution_37, mul_53);  convolution_37 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(primals_221, 1);  primals_221 = None
        convolution_40 = torch.ops.aten.convolution.default(mul_54, primals_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_41 = torch.ops.aten.convolution.default(mul_54, primals_223, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_42 = torch.ops.aten.convolution.default(convolution_41, primals_224, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_55 = torch.ops.aten.mul.Tensor(convolution_42, 2.0);  convolution_42 = None
        add_53 = torch.ops.aten.add.Tensor(convolution_40, mul_55);  convolution_40 = mul_55 = None
        add_54 = torch.ops.aten.add.Tensor(add_52, add_53);  add_52 = add_53 = None
        clone_14 = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
        view_62 = torch.ops.aten.view.default(clone_14, [4, 32, 16, 4096])
        var_mean_11 = torch.ops.aten.var_mean.correction(view_62, [2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_11[0]
        getitem_27 = var_mean_11[1];  var_mean_11 = None
        add_55 = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_11 = torch.ops.aten.sub.Tensor(view_62, getitem_27);  view_62 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        view_63 = torch.ops.aten.view.default(mul_56, [4, 512, 64, 64]);  mul_56 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(primals_52, 0)
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, 3);  unsqueeze_66 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(primals_51, 0)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        mul_57 = torch.ops.aten.mul.Tensor(view_63, unsqueeze_70);  view_63 = unsqueeze_70 = None
        add_56 = torch.ops.aten.add.Tensor(mul_57, unsqueeze_67);  mul_57 = unsqueeze_67 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(add_56)
        mul_58 = torch.ops.aten.mul.Tensor(add_56, sigmoid_10);  add_56 = sigmoid_10 = None
        convolution_43 = torch.ops.aten.convolution.default(mul_58, primals_53, primals_54, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_54 = None
        convolution_44 = torch.ops.aten.convolution.default(mul_58, primals_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_45 = torch.ops.aten.convolution.default(convolution_44, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_59 = torch.ops.aten.mul.Tensor(convolution_45, 2.0);  convolution_45 = None
        add_57 = torch.ops.aten.add.Tensor(convolution_43, mul_59);  convolution_43 = mul_59 = None
        view_64 = torch.ops.aten.view.default(add_57, [4, 32, 16, 4096])
        var_mean_12 = torch.ops.aten.var_mean.correction(view_64, [2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_12[0]
        getitem_29 = var_mean_12[1];  var_mean_12 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_12 = torch.ops.aten.sub.Tensor(view_64, getitem_29);  view_64 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        view_65 = torch.ops.aten.view.default(mul_60, [4, 512, 64, 64]);  mul_60 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(primals_58, 0)
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(primals_57, 0)
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, 2);  unsqueeze_74 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(unsqueeze_75, 3);  unsqueeze_75 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_65, unsqueeze_76);  view_65 = unsqueeze_76 = None
        add_59 = torch.ops.aten.add.Tensor(mul_61, unsqueeze_73);  mul_61 = unsqueeze_73 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(add_59)
        mul_62 = torch.ops.aten.mul.Tensor(add_59, sigmoid_11);  add_59 = sigmoid_11 = None
        convolution_46 = torch.ops.aten.convolution.default(mul_62, primals_59, primals_60, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_60 = None
        convolution_47 = torch.ops.aten.convolution.default(mul_62, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_48 = torch.ops.aten.convolution.default(convolution_47, primals_62, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_63 = torch.ops.aten.mul.Tensor(convolution_48, 2.0);  convolution_48 = None
        add_60 = torch.ops.aten.add.Tensor(convolution_46, mul_63);  convolution_46 = mul_63 = None
        add_61 = torch.ops.aten.add.Tensor(add_54, add_60);  add_54 = add_60 = None
        div_7 = torch.ops.aten.div.Tensor(add_61, 1.0);  add_61 = None
        clone_16 = torch.ops.aten.clone.default(div_7, memory_format = torch.contiguous_format)
        view_66 = torch.ops.aten.view.default(clone_16, [4, 32, 16, 4096])
        var_mean_13 = torch.ops.aten.var_mean.correction(view_66, [2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_13[0]
        getitem_31 = var_mean_13[1];  var_mean_13 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_13 = torch.ops.aten.sub.Tensor(view_66, getitem_31);  view_66 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        view_67 = torch.ops.aten.view.default(mul_64, [4, 512, 64, 64]);  mul_64 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(primals_64, 0)
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, 3);  unsqueeze_78 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(primals_63, 0)
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        mul_65 = torch.ops.aten.mul.Tensor(view_67, unsqueeze_82);  view_67 = unsqueeze_82 = None
        add_63 = torch.ops.aten.add.Tensor(mul_65, unsqueeze_79);  mul_65 = unsqueeze_79 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(add_63)
        mul_66 = torch.ops.aten.mul.Tensor(add_63, sigmoid_12);  add_63 = sigmoid_12 = None
        convolution_49 = torch.ops.aten.convolution.default(mul_66, primals_65, primals_66, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_66 = None
        convolution_50 = torch.ops.aten.convolution.default(mul_66, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_51 = torch.ops.aten.convolution.default(convolution_50, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_67 = torch.ops.aten.mul.Tensor(convolution_51, 2.0);  convolution_51 = None
        add_64 = torch.ops.aten.add.Tensor(convolution_49, mul_67);  convolution_49 = mul_67 = None
        view_68 = torch.ops.aten.view.default(add_64, [4, 32, 16, 4096])
        var_mean_14 = torch.ops.aten.var_mean.correction(view_68, [2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_14[0]
        getitem_33 = var_mean_14[1];  var_mean_14 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_14 = torch.ops.aten.sub.Tensor(view_68, getitem_33);  view_68 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        view_69 = torch.ops.aten.view.default(mul_68, [4, 512, 64, 64]);  mul_68 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(primals_70, 0)
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(primals_69, 0)
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(unsqueeze_87, 3);  unsqueeze_87 = None
        mul_69 = torch.ops.aten.mul.Tensor(view_69, unsqueeze_88);  view_69 = unsqueeze_88 = None
        add_66 = torch.ops.aten.add.Tensor(mul_69, unsqueeze_85);  mul_69 = unsqueeze_85 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(add_66)
        mul_70 = torch.ops.aten.mul.Tensor(add_66, sigmoid_13);  add_66 = sigmoid_13 = None
        convolution_52 = torch.ops.aten.convolution.default(mul_70, primals_71, primals_72, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_72 = None
        convolution_53 = torch.ops.aten.convolution.default(mul_70, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_54 = torch.ops.aten.convolution.default(convolution_53, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_71 = torch.ops.aten.mul.Tensor(convolution_54, 2.0);  convolution_54 = None
        add_67 = torch.ops.aten.add.Tensor(convolution_52, mul_71);  convolution_52 = mul_71 = None
        add_68 = torch.ops.aten.add.Tensor(div_7, add_67);  div_7 = add_67 = None
        div_8 = torch.ops.aten.div.Tensor(add_68, 1.0);  add_68 = None
        clone_18 = torch.ops.aten.clone.default(div_8, memory_format = torch.contiguous_format)
        view_70 = torch.ops.aten.view.default(clone_18, [4, 32, 16, 4096])
        var_mean_15 = torch.ops.aten.var_mean.correction(view_70, [2, 3], correction = 0, keepdim = True)
        getitem_34 = var_mean_15[0]
        getitem_35 = var_mean_15[1];  var_mean_15 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_15 = torch.ops.aten.sub.Tensor(view_70, getitem_35);  view_70 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        view_71 = torch.ops.aten.view.default(mul_72, [4, 512, 64, 64]);  mul_72 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(primals_76, 0)
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(primals_75, 0)
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        mul_73 = torch.ops.aten.mul.Tensor(view_71, unsqueeze_94);  view_71 = unsqueeze_94 = None
        add_70 = torch.ops.aten.add.Tensor(mul_73, unsqueeze_91);  mul_73 = unsqueeze_91 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(add_70)
        mul_74 = torch.ops.aten.mul.Tensor(add_70, sigmoid_14);  add_70 = sigmoid_14 = None
        convolution_55 = torch.ops.aten.convolution.default(mul_74, primals_77, primals_78, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_78 = None
        convolution_56 = torch.ops.aten.convolution.default(mul_74, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_57 = torch.ops.aten.convolution.default(convolution_56, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_75 = torch.ops.aten.mul.Tensor(convolution_57, 2.0);  convolution_57 = None
        add_71 = torch.ops.aten.add.Tensor(convolution_55, mul_75);  convolution_55 = mul_75 = None
        view_72 = torch.ops.aten.view.default(add_71, [4, 32, 16, 4096])
        var_mean_16 = torch.ops.aten.var_mean.correction(view_72, [2, 3], correction = 0, keepdim = True)
        getitem_36 = var_mean_16[0]
        getitem_37 = var_mean_16[1];  var_mean_16 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_16 = torch.ops.aten.sub.Tensor(view_72, getitem_37);  view_72 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        view_73 = torch.ops.aten.view.default(mul_76, [4, 512, 64, 64]);  mul_76 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(primals_82, 0)
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(primals_81, 0)
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_73, unsqueeze_100);  view_73 = unsqueeze_100 = None
        add_73 = torch.ops.aten.add.Tensor(mul_77, unsqueeze_97);  mul_77 = unsqueeze_97 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(add_73)
        mul_78 = torch.ops.aten.mul.Tensor(add_73, sigmoid_15);  add_73 = sigmoid_15 = None
        convolution_58 = torch.ops.aten.convolution.default(mul_78, primals_83, primals_84, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_84 = None
        convolution_59 = torch.ops.aten.convolution.default(mul_78, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_60 = torch.ops.aten.convolution.default(convolution_59, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_79 = torch.ops.aten.mul.Tensor(convolution_60, 2.0);  convolution_60 = None
        add_74 = torch.ops.aten.add.Tensor(convolution_58, mul_79);  convolution_58 = mul_79 = None
        add_75 = torch.ops.aten.add.Tensor(div_8, add_74);  div_8 = add_74 = None
        div_9 = torch.ops.aten.div.Tensor(add_75, 1.0);  add_75 = None
        iota_2 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_80 = torch.ops.aten.mul.Tensor(iota_2, 1);  iota_2 = None
        add_76 = torch.ops.aten.add.Tensor(mul_80, 0);  mul_80 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_76, torch.float32);  add_76 = None
        add_77 = torch.ops.aten.add.Tensor(convert_element_type_4, 0.0);  convert_element_type_4 = None
        mul_81 = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_81, torch.int64);  mul_81 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1)
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(div_9, [None, None, unsqueeze_101, convert_element_type_5]);  div_9 = unsqueeze_101 = None
        clone_20 = torch.ops.aten.clone.default(_unsafe_index_1, memory_format = torch.channels_last);  _unsafe_index_1 = None
        convolution_61 = torch.ops.aten.convolution.default(clone_20, primals_87, primals_88, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_88 = None
        convolution_62 = torch.ops.aten.convolution.default(clone_20, primals_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_63 = torch.ops.aten.convolution.default(convolution_62, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_84 = torch.ops.aten.mul.Tensor(convolution_63, 2.0);  convolution_63 = None
        add_80 = torch.ops.aten.add.Tensor(convolution_61, mul_84);  convolution_61 = mul_84 = None
        mul_85 = torch.ops.aten.mul.Tensor(primals_225, 1);  primals_225 = None
        convolution_64 = torch.ops.aten.convolution.default(mul_85, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_65 = torch.ops.aten.convolution.default(mul_85, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_66 = torch.ops.aten.convolution.default(convolution_65, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_86 = torch.ops.aten.mul.Tensor(convolution_66, 2.0);  convolution_66 = None
        add_81 = torch.ops.aten.add.Tensor(convolution_64, mul_86);  convolution_64 = mul_86 = None
        add_82 = torch.ops.aten.add.Tensor(add_80, add_81);  add_80 = add_81 = None
        clone_21 = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
        view_74 = torch.ops.aten.view.default(clone_21, [4, 32, 16, 16384]);  clone_21 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(view_74, [2, 3], correction = 0, keepdim = True)
        getitem_38 = var_mean_17[0]
        getitem_39 = var_mean_17[1];  var_mean_17 = None
        add_83 = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_17 = torch.ops.aten.sub.Tensor(view_74, getitem_39);  view_74 = None
        mul_87 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        view_75 = torch.ops.aten.view.default(mul_87, [4, 512, 128, 128]);  mul_87 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(primals_92, 0)
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(primals_91, 0)
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        mul_88 = torch.ops.aten.mul.Tensor(view_75, unsqueeze_107);  view_75 = unsqueeze_107 = None
        add_84 = torch.ops.aten.add.Tensor(mul_88, unsqueeze_104);  mul_88 = unsqueeze_104 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(add_84)
        mul_89 = torch.ops.aten.mul.Tensor(add_84, sigmoid_16);  add_84 = sigmoid_16 = None
        convolution_67 = torch.ops.aten.convolution.default(mul_89, primals_93, primals_94, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_94 = None
        convolution_68 = torch.ops.aten.convolution.default(mul_89, primals_95, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_69 = torch.ops.aten.convolution.default(convolution_68, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_90 = torch.ops.aten.mul.Tensor(convolution_69, 2.0);  convolution_69 = None
        add_85 = torch.ops.aten.add.Tensor(convolution_67, mul_90);  convolution_67 = mul_90 = None
        view_76 = torch.ops.aten.view.default(add_85, [4, 32, 8, 16384])
        var_mean_18 = torch.ops.aten.var_mean.correction(view_76, [2, 3], correction = 0, keepdim = True)
        getitem_40 = var_mean_18[0]
        getitem_41 = var_mean_18[1];  var_mean_18 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_18 = torch.ops.aten.sub.Tensor(view_76, getitem_41);  view_76 = None
        mul_91 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        view_77 = torch.ops.aten.view.default(mul_91, [4, 256, 128, 128]);  mul_91 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(primals_98, 0)
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(primals_97, 0)
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_77, unsqueeze_113);  view_77 = unsqueeze_113 = None
        add_87 = torch.ops.aten.add.Tensor(mul_92, unsqueeze_110);  mul_92 = unsqueeze_110 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(add_87)
        mul_93 = torch.ops.aten.mul.Tensor(add_87, sigmoid_17);  add_87 = sigmoid_17 = None
        convolution_70 = torch.ops.aten.convolution.default(mul_93, primals_99, primals_100, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_100 = None
        convolution_71 = torch.ops.aten.convolution.default(mul_93, primals_101, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_72 = torch.ops.aten.convolution.default(convolution_71, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_94 = torch.ops.aten.mul.Tensor(convolution_72, 2.0);  convolution_72 = None
        add_88 = torch.ops.aten.add.Tensor(convolution_70, mul_94);  convolution_70 = mul_94 = None
        convolution_73 = torch.ops.aten.convolution.default(add_82, primals_103, primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_104 = None
        convolution_74 = torch.ops.aten.convolution.default(add_82, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_75 = torch.ops.aten.convolution.default(convolution_74, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_95 = torch.ops.aten.mul.Tensor(convolution_75, 2.0);  convolution_75 = None
        add_89 = torch.ops.aten.add.Tensor(convolution_73, mul_95);  convolution_73 = mul_95 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, add_88);  add_89 = add_88 = None
        div_10 = torch.ops.aten.div.Tensor(add_90, 1.0);  add_90 = None
        clone_23 = torch.ops.aten.clone.default(div_10, memory_format = torch.contiguous_format)
        view_78 = torch.ops.aten.view.default(clone_23, [4, 32, 8, 16384])
        var_mean_19 = torch.ops.aten.var_mean.correction(view_78, [2, 3], correction = 0, keepdim = True)
        getitem_42 = var_mean_19[0]
        getitem_43 = var_mean_19[1];  var_mean_19 = None
        add_91 = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_19 = torch.ops.aten.sub.Tensor(view_78, getitem_43);  view_78 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        view_79 = torch.ops.aten.view.default(mul_96, [4, 256, 128, 128]);  mul_96 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(primals_108, 0)
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(primals_107, 0)
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        mul_97 = torch.ops.aten.mul.Tensor(view_79, unsqueeze_119);  view_79 = unsqueeze_119 = None
        add_92 = torch.ops.aten.add.Tensor(mul_97, unsqueeze_116);  mul_97 = unsqueeze_116 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(add_92)
        mul_98 = torch.ops.aten.mul.Tensor(add_92, sigmoid_18);  add_92 = sigmoid_18 = None
        convolution_76 = torch.ops.aten.convolution.default(mul_98, primals_109, primals_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_110 = None
        convolution_77 = torch.ops.aten.convolution.default(mul_98, primals_111, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_78 = torch.ops.aten.convolution.default(convolution_77, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_99 = torch.ops.aten.mul.Tensor(convolution_78, 2.0);  convolution_78 = None
        add_93 = torch.ops.aten.add.Tensor(convolution_76, mul_99);  convolution_76 = mul_99 = None
        view_80 = torch.ops.aten.view.default(add_93, [4, 32, 8, 16384])
        var_mean_20 = torch.ops.aten.var_mean.correction(view_80, [2, 3], correction = 0, keepdim = True)
        getitem_44 = var_mean_20[0]
        getitem_45 = var_mean_20[1];  var_mean_20 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_20 = torch.ops.aten.sub.Tensor(view_80, getitem_45);  view_80 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        view_81 = torch.ops.aten.view.default(mul_100, [4, 256, 128, 128]);  mul_100 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(primals_114, 0)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(primals_113, 0)
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        mul_101 = torch.ops.aten.mul.Tensor(view_81, unsqueeze_125);  view_81 = unsqueeze_125 = None
        add_95 = torch.ops.aten.add.Tensor(mul_101, unsqueeze_122);  mul_101 = unsqueeze_122 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(add_95)
        mul_102 = torch.ops.aten.mul.Tensor(add_95, sigmoid_19);  add_95 = sigmoid_19 = None
        convolution_79 = torch.ops.aten.convolution.default(mul_102, primals_115, primals_116, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_116 = None
        convolution_80 = torch.ops.aten.convolution.default(mul_102, primals_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_81 = torch.ops.aten.convolution.default(convolution_80, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_103 = torch.ops.aten.mul.Tensor(convolution_81, 2.0);  convolution_81 = None
        add_96 = torch.ops.aten.add.Tensor(convolution_79, mul_103);  convolution_79 = mul_103 = None
        add_97 = torch.ops.aten.add.Tensor(div_10, add_96);  div_10 = add_96 = None
        div_11 = torch.ops.aten.div.Tensor(add_97, 1.0);  add_97 = None
        clone_25 = torch.ops.aten.clone.default(div_11, memory_format = torch.contiguous_format)
        view_82 = torch.ops.aten.view.default(clone_25, [4, 32, 8, 16384])
        var_mean_21 = torch.ops.aten.var_mean.correction(view_82, [2, 3], correction = 0, keepdim = True)
        getitem_46 = var_mean_21[0]
        getitem_47 = var_mean_21[1];  var_mean_21 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_21 = torch.ops.aten.sub.Tensor(view_82, getitem_47);  view_82 = None
        mul_104 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        view_83 = torch.ops.aten.view.default(mul_104, [4, 256, 128, 128]);  mul_104 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(primals_120, 0)
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(primals_119, 0)
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
        mul_105 = torch.ops.aten.mul.Tensor(view_83, unsqueeze_131);  view_83 = unsqueeze_131 = None
        add_99 = torch.ops.aten.add.Tensor(mul_105, unsqueeze_128);  mul_105 = unsqueeze_128 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(add_99)
        mul_106 = torch.ops.aten.mul.Tensor(add_99, sigmoid_20);  add_99 = sigmoid_20 = None
        convolution_82 = torch.ops.aten.convolution.default(mul_106, primals_121, primals_122, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_122 = None
        convolution_83 = torch.ops.aten.convolution.default(mul_106, primals_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_84 = torch.ops.aten.convolution.default(convolution_83, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_107 = torch.ops.aten.mul.Tensor(convolution_84, 2.0);  convolution_84 = None
        add_100 = torch.ops.aten.add.Tensor(convolution_82, mul_107);  convolution_82 = mul_107 = None
        view_84 = torch.ops.aten.view.default(add_100, [4, 32, 8, 16384])
        var_mean_22 = torch.ops.aten.var_mean.correction(view_84, [2, 3], correction = 0, keepdim = True)
        getitem_48 = var_mean_22[0]
        getitem_49 = var_mean_22[1];  var_mean_22 = None
        add_101 = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_22 = torch.ops.aten.sub.Tensor(view_84, getitem_49);  view_84 = None
        mul_108 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        view_85 = torch.ops.aten.view.default(mul_108, [4, 256, 128, 128]);  mul_108 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(primals_126, 0)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(primals_125, 0)
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_85, unsqueeze_137);  view_85 = unsqueeze_137 = None
        add_102 = torch.ops.aten.add.Tensor(mul_109, unsqueeze_134);  mul_109 = unsqueeze_134 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(add_102)
        mul_110 = torch.ops.aten.mul.Tensor(add_102, sigmoid_21);  add_102 = sigmoid_21 = None
        convolution_85 = torch.ops.aten.convolution.default(mul_110, primals_127, primals_128, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_128 = None
        convolution_86 = torch.ops.aten.convolution.default(mul_110, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_87 = torch.ops.aten.convolution.default(convolution_86, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_111 = torch.ops.aten.mul.Tensor(convolution_87, 2.0);  convolution_87 = None
        add_103 = torch.ops.aten.add.Tensor(convolution_85, mul_111);  convolution_85 = mul_111 = None
        add_104 = torch.ops.aten.add.Tensor(div_11, add_103);  div_11 = add_103 = None
        div_12 = torch.ops.aten.div.Tensor(add_104, 1.0);  add_104 = None
        iota_4 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_112 = torch.ops.aten.mul.Tensor(iota_4, 1);  iota_4 = None
        add_105 = torch.ops.aten.add.Tensor(mul_112, 0);  mul_112 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        add_106 = torch.ops.aten.add.Tensor(convert_element_type_8, 0.0);  convert_element_type_8 = None
        mul_113 = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_113, torch.int64);  mul_113 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1)
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(div_12, [None, None, unsqueeze_138, convert_element_type_9]);  div_12 = unsqueeze_138 = None
        clone_27 = torch.ops.aten.clone.default(_unsafe_index_2, memory_format = torch.channels_last);  _unsafe_index_2 = None
        convolution_88 = torch.ops.aten.convolution.default(clone_27, primals_131, primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_132 = None
        convolution_89 = torch.ops.aten.convolution.default(clone_27, primals_133, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_90 = torch.ops.aten.convolution.default(convolution_89, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_116 = torch.ops.aten.mul.Tensor(convolution_90, 2.0);  convolution_90 = None
        add_109 = torch.ops.aten.add.Tensor(convolution_88, mul_116);  convolution_88 = mul_116 = None
        mul_117 = torch.ops.aten.mul.Tensor(primals_229, 1);  primals_229 = None
        convolution_91 = torch.ops.aten.convolution.default(mul_117, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_92 = torch.ops.aten.convolution.default(mul_117, primals_231, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_93 = torch.ops.aten.convolution.default(convolution_92, primals_232, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_118 = torch.ops.aten.mul.Tensor(convolution_93, 2.0);  convolution_93 = None
        add_110 = torch.ops.aten.add.Tensor(convolution_91, mul_118);  convolution_91 = mul_118 = None
        add_111 = torch.ops.aten.add.Tensor(add_109, add_110);  add_109 = add_110 = None
        clone_28 = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
        view_86 = torch.ops.aten.view.default(clone_28, [4, 32, 8, 65536]);  clone_28 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(view_86, [2, 3], correction = 0, keepdim = True)
        getitem_50 = var_mean_23[0]
        getitem_51 = var_mean_23[1];  var_mean_23 = None
        add_112 = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_23 = torch.ops.aten.sub.Tensor(view_86, getitem_51);  view_86 = None
        mul_119 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        view_87 = torch.ops.aten.view.default(mul_119, [4, 256, 256, 256]);  mul_119 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(primals_136, 0)
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(primals_135, 0)
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_120 = torch.ops.aten.mul.Tensor(view_87, unsqueeze_144);  view_87 = unsqueeze_144 = None
        add_113 = torch.ops.aten.add.Tensor(mul_120, unsqueeze_141);  mul_120 = unsqueeze_141 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(add_113)
        mul_121 = torch.ops.aten.mul.Tensor(add_113, sigmoid_22);  add_113 = sigmoid_22 = None
        convolution_94 = torch.ops.aten.convolution.default(mul_121, primals_137, primals_138, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_138 = None
        convolution_95 = torch.ops.aten.convolution.default(mul_121, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_96 = torch.ops.aten.convolution.default(convolution_95, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_122 = torch.ops.aten.mul.Tensor(convolution_96, 2.0);  convolution_96 = None
        add_114 = torch.ops.aten.add.Tensor(convolution_94, mul_122);  convolution_94 = mul_122 = None
        view_88 = torch.ops.aten.view.default(add_114, [4, 32, 4, 65536])
        var_mean_24 = torch.ops.aten.var_mean.correction(view_88, [2, 3], correction = 0, keepdim = True)
        getitem_52 = var_mean_24[0]
        getitem_53 = var_mean_24[1];  var_mean_24 = None
        add_115 = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_24 = torch.ops.aten.sub.Tensor(view_88, getitem_53);  view_88 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        view_89 = torch.ops.aten.view.default(mul_123, [4, 128, 256, 256]);  mul_123 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(primals_142, 0)
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(primals_141, 0)
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        mul_124 = torch.ops.aten.mul.Tensor(view_89, unsqueeze_150);  view_89 = unsqueeze_150 = None
        add_116 = torch.ops.aten.add.Tensor(mul_124, unsqueeze_147);  mul_124 = unsqueeze_147 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(add_116)
        mul_125 = torch.ops.aten.mul.Tensor(add_116, sigmoid_23);  add_116 = sigmoid_23 = None
        convolution_97 = torch.ops.aten.convolution.default(mul_125, primals_143, primals_144, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_144 = None
        convolution_98 = torch.ops.aten.convolution.default(mul_125, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_99 = torch.ops.aten.convolution.default(convolution_98, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_126 = torch.ops.aten.mul.Tensor(convolution_99, 2.0);  convolution_99 = None
        add_117 = torch.ops.aten.add.Tensor(convolution_97, mul_126);  convolution_97 = mul_126 = None
        convolution_100 = torch.ops.aten.convolution.default(add_111, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
        convolution_101 = torch.ops.aten.convolution.default(add_111, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_102 = torch.ops.aten.convolution.default(convolution_101, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_127 = torch.ops.aten.mul.Tensor(convolution_102, 2.0);  convolution_102 = None
        add_118 = torch.ops.aten.add.Tensor(convolution_100, mul_127);  convolution_100 = mul_127 = None
        add_119 = torch.ops.aten.add.Tensor(add_118, add_117);  add_118 = add_117 = None
        div_13 = torch.ops.aten.div.Tensor(add_119, 1.0);  add_119 = None
        clone_30 = torch.ops.aten.clone.default(div_13, memory_format = torch.contiguous_format)
        view_90 = torch.ops.aten.view.default(clone_30, [4, 32, 4, 65536])
        var_mean_25 = torch.ops.aten.var_mean.correction(view_90, [2, 3], correction = 0, keepdim = True)
        getitem_54 = var_mean_25[0]
        getitem_55 = var_mean_25[1];  var_mean_25 = None
        add_120 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_25 = torch.ops.aten.sub.Tensor(view_90, getitem_55);  view_90 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        view_91 = torch.ops.aten.view.default(mul_128, [4, 128, 256, 256]);  mul_128 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(primals_152, 0)
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(primals_151, 0)
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_129 = torch.ops.aten.mul.Tensor(view_91, unsqueeze_156);  view_91 = unsqueeze_156 = None
        add_121 = torch.ops.aten.add.Tensor(mul_129, unsqueeze_153);  mul_129 = unsqueeze_153 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(add_121)
        mul_130 = torch.ops.aten.mul.Tensor(add_121, sigmoid_24);  add_121 = sigmoid_24 = None
        convolution_103 = torch.ops.aten.convolution.default(mul_130, primals_153, primals_154, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_154 = None
        convolution_104 = torch.ops.aten.convolution.default(mul_130, primals_155, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_105 = torch.ops.aten.convolution.default(convolution_104, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_131 = torch.ops.aten.mul.Tensor(convolution_105, 2.0);  convolution_105 = None
        add_122 = torch.ops.aten.add.Tensor(convolution_103, mul_131);  convolution_103 = mul_131 = None
        view_92 = torch.ops.aten.view.default(add_122, [4, 32, 4, 65536])
        var_mean_26 = torch.ops.aten.var_mean.correction(view_92, [2, 3], correction = 0, keepdim = True)
        getitem_56 = var_mean_26[0]
        getitem_57 = var_mean_26[1];  var_mean_26 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_26 = torch.ops.aten.sub.Tensor(view_92, getitem_57);  view_92 = None
        mul_132 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        view_93 = torch.ops.aten.view.default(mul_132, [4, 128, 256, 256]);  mul_132 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(primals_158, 0)
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(primals_157, 0)
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        mul_133 = torch.ops.aten.mul.Tensor(view_93, unsqueeze_162);  view_93 = unsqueeze_162 = None
        add_124 = torch.ops.aten.add.Tensor(mul_133, unsqueeze_159);  mul_133 = unsqueeze_159 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(add_124)
        mul_134 = torch.ops.aten.mul.Tensor(add_124, sigmoid_25);  add_124 = sigmoid_25 = None
        convolution_106 = torch.ops.aten.convolution.default(mul_134, primals_159, primals_160, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_160 = None
        convolution_107 = torch.ops.aten.convolution.default(mul_134, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_108 = torch.ops.aten.convolution.default(convolution_107, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_135 = torch.ops.aten.mul.Tensor(convolution_108, 2.0);  convolution_108 = None
        add_125 = torch.ops.aten.add.Tensor(convolution_106, mul_135);  convolution_106 = mul_135 = None
        add_126 = torch.ops.aten.add.Tensor(div_13, add_125);  div_13 = add_125 = None
        div_14 = torch.ops.aten.div.Tensor(add_126, 1.0);  add_126 = None
        clone_32 = torch.ops.aten.clone.default(div_14, memory_format = torch.contiguous_format)
        view_94 = torch.ops.aten.view.default(clone_32, [4, 32, 4, 65536])
        var_mean_27 = torch.ops.aten.var_mean.correction(view_94, [2, 3], correction = 0, keepdim = True)
        getitem_58 = var_mean_27[0]
        getitem_59 = var_mean_27[1];  var_mean_27 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_27 = torch.ops.aten.sub.Tensor(view_94, getitem_59);  view_94 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        view_95 = torch.ops.aten.view.default(mul_136, [4, 128, 256, 256]);  mul_136 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(primals_164, 0)
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(primals_163, 0)
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_137 = torch.ops.aten.mul.Tensor(view_95, unsqueeze_168);  view_95 = unsqueeze_168 = None
        add_128 = torch.ops.aten.add.Tensor(mul_137, unsqueeze_165);  mul_137 = unsqueeze_165 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(add_128)
        mul_138 = torch.ops.aten.mul.Tensor(add_128, sigmoid_26);  add_128 = sigmoid_26 = None
        convolution_109 = torch.ops.aten.convolution.default(mul_138, primals_165, primals_166, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_166 = None
        convolution_110 = torch.ops.aten.convolution.default(mul_138, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_111 = torch.ops.aten.convolution.default(convolution_110, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_139 = torch.ops.aten.mul.Tensor(convolution_111, 2.0);  convolution_111 = None
        add_129 = torch.ops.aten.add.Tensor(convolution_109, mul_139);  convolution_109 = mul_139 = None
        view_96 = torch.ops.aten.view.default(add_129, [4, 32, 4, 65536])
        var_mean_28 = torch.ops.aten.var_mean.correction(view_96, [2, 3], correction = 0, keepdim = True)
        getitem_60 = var_mean_28[0]
        getitem_61 = var_mean_28[1];  var_mean_28 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_28 = torch.ops.aten.sub.Tensor(view_96, getitem_61);  view_96 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        view_97 = torch.ops.aten.view.default(mul_140, [4, 128, 256, 256]);  mul_140 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(primals_170, 0)
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(primals_169, 0)
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        mul_141 = torch.ops.aten.mul.Tensor(view_97, unsqueeze_174);  view_97 = unsqueeze_174 = None
        add_131 = torch.ops.aten.add.Tensor(mul_141, unsqueeze_171);  mul_141 = unsqueeze_171 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(add_131)
        mul_142 = torch.ops.aten.mul.Tensor(add_131, sigmoid_27);  add_131 = sigmoid_27 = None
        convolution_112 = torch.ops.aten.convolution.default(mul_142, primals_171, primals_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_172 = None
        convolution_113 = torch.ops.aten.convolution.default(mul_142, primals_173, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_114 = torch.ops.aten.convolution.default(convolution_113, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_143 = torch.ops.aten.mul.Tensor(convolution_114, 2.0);  convolution_114 = None
        add_132 = torch.ops.aten.add.Tensor(convolution_112, mul_143);  convolution_112 = mul_143 = None
        add_133 = torch.ops.aten.add.Tensor(div_14, add_132);  div_14 = add_132 = None
        div_15 = torch.ops.aten.div.Tensor(add_133, 1.0);  add_133 = None
        clone_34 = torch.ops.aten.clone.default(div_15, memory_format = torch.contiguous_format);  div_15 = None
        view_98 = torch.ops.aten.view.default(clone_34, [4, 32, 4, 65536])
        var_mean_29 = torch.ops.aten.var_mean.correction(view_98, [2, 3], correction = 0, keepdim = True)
        getitem_62 = var_mean_29[0]
        getitem_63 = var_mean_29[1];  var_mean_29 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_29 = torch.ops.aten.sub.Tensor(view_98, getitem_63);  view_98 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        view_99 = torch.ops.aten.view.default(mul_144, [4, 128, 256, 256]);  mul_144 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(primals_234, 0)
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(primals_233, 0)
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_145 = torch.ops.aten.mul.Tensor(view_99, unsqueeze_180);  view_99 = unsqueeze_180 = None
        add_135 = torch.ops.aten.add.Tensor(mul_145, unsqueeze_177);  mul_145 = unsqueeze_177 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(add_135)
        mul_146 = torch.ops.aten.mul.Tensor(add_135, sigmoid_28);  add_135 = sigmoid_28 = None
        convolution_115 = torch.ops.aten.convolution.default(mul_146, primals_235, primals_236, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_236 = None
        convolution_116 = torch.ops.aten.convolution.default(mul_146, primals_237, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_117 = torch.ops.aten.convolution.default(convolution_116, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_147 = torch.ops.aten.mul.Tensor(convolution_117, 2.0);  convolution_117 = None
        add_136 = torch.ops.aten.add.Tensor(convolution_115, mul_147);  convolution_115 = mul_147 = None
        clamp_min = torch.ops.aten.clamp_min.default(add_136, -1)
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 1);  clamp_min = None
        ge = torch.ops.aten.ge.Scalar(add_136, -1)
        le = torch.ops.aten.le.Scalar(add_136, 1);  add_136 = None
        logical_and = torch.ops.aten.logical_and.default(ge, le);  ge = le = None
        permute_53 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_57 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_59 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_66 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_70 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_75 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_79 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_84 = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_88 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        return (clamp_max, primals_5, primals_7, primals_9, primals_10, primals_11, primals_12, primals_13, primals_15, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_73, primals_74, primals_75, primals_76, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_173, primals_174, primals_175, primals_176, primals_177, primals_179, primals_180, primals_181, primals_182, primals_183, primals_185, primals_186, primals_187, primals_189, primals_193, primals_197, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, div, convolution, convolution_2, add, getitem_1, rsqrt, mul_3, convolution_5, add_3, getitem_3, rsqrt_1, mul_7, convolution_8, view_5, squeeze_4, squeeze_5, view_11, mm, mm_2, mm_4, permute_15, permute_16, permute_17, getitem_6, getitem_7, getitem_8, getitem_9, mm_6, clone_5, getitem_11, rsqrt_3, mul_17, convolution_11, add_20, getitem_13, rsqrt_4, mul_21, convolution_14, mul_23, convolution_17, clone_7, getitem_15, rsqrt_5, mul_27, convolution_20, add_29, getitem_17, rsqrt_6, mul_31, convolution_23, clone_9, getitem_19, rsqrt_7, mul_35, convolution_26, add_36, getitem_21, rsqrt_8, mul_39, convolution_29, clone_11, getitem_23, rsqrt_9, mul_43, convolution_32, add_43, getitem_25, rsqrt_10, mul_47, convolution_35, convert_element_type_1, clone_13, convolution_38, mul_54, convolution_41, clone_14, getitem_27, rsqrt_11, mul_58, convolution_44, add_57, getitem_29, rsqrt_12, mul_62, convolution_47, clone_16, getitem_31, rsqrt_13, mul_66, convolution_50, add_64, getitem_33, rsqrt_14, mul_70, convolution_53, clone_18, getitem_35, rsqrt_15, mul_74, convolution_56, add_71, getitem_37, rsqrt_16, mul_78, convolution_59, convert_element_type_5, clone_20, convolution_62, mul_85, convolution_65, add_82, getitem_39, rsqrt_17, mul_89, convolution_68, add_85, getitem_41, rsqrt_18, mul_93, convolution_71, convolution_74, clone_23, getitem_43, rsqrt_19, mul_98, convolution_77, add_93, getitem_45, rsqrt_20, mul_102, convolution_80, clone_25, getitem_47, rsqrt_21, mul_106, convolution_83, add_100, getitem_49, rsqrt_22, mul_110, convolution_86, convert_element_type_9, clone_27, convolution_89, mul_117, convolution_92, add_111, getitem_51, rsqrt_23, mul_121, convolution_95, add_114, getitem_53, rsqrt_24, mul_125, convolution_98, convolution_101, clone_30, getitem_55, rsqrt_25, mul_130, convolution_104, add_122, getitem_57, rsqrt_26, mul_134, convolution_107, clone_32, getitem_59, rsqrt_27, mul_138, convolution_110, add_129, getitem_61, rsqrt_28, mul_142, convolution_113, clone_34, getitem_63, rsqrt_29, mul_146, convolution_116, logical_and, permute_53, permute_57, permute_59, permute_66, permute_70, permute_75, permute_79, permute_84, permute_88)
        
def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 32, 32), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (4, 32, 32), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf2, (4, 32, 32), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf3, (4, 32, 32), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (4, 4, 1, 1), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512, 4, 3, 3), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf8, (4, 4, 3, 3), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512, 4, 1, 1), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 512, 3, 3), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4, 512, 3, 3), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512, 4, 1, 1), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512, 512, 3, 3), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf20, (4, 512, 3, 3), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512, 4, 1, 1), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 512, 3, 3), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf26, (4, 512, 3, 3), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512, 4, 1, 1), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512, 512, 3, 3), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4, 512, 3, 3), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 4, 1, 1), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512, 512, 3, 3), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf38, (4, 512, 3, 3), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512, 4, 1, 1), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf40, (512,), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512,), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512, 512, 3, 3), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf44, (4, 512, 3, 3), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512, 4, 1, 1), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512, 512, 3, 3), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf48, (4, 512, 3, 3), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512, 4, 1, 1), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512, 512, 3, 3), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf54, (4, 512, 3, 3), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 4, 1, 1), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512,), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512, 512, 3, 3), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512,), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf60, (4, 512, 3, 3), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512, 4, 1, 1), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512,), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512, 512, 3, 3), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf66, (4, 512, 3, 3), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 4, 1, 1), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512, 512, 3, 3), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf72, (4, 512, 3, 3), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 4, 1, 1), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512, 512, 3, 3), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf78, (4, 512, 3, 3), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512, 4, 1, 1), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512, 512, 3, 3), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf84, (4, 512, 3, 3), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 4, 1, 1), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512, 512, 3, 3), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf88, (4, 512, 3, 3), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512, 4, 1, 1), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512,), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf91, (512,), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256, 512, 3, 3), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf94, (4, 512, 3, 3), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 4, 1, 1), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256, 256, 3, 3), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf99, (256,), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf100, (4, 256, 3, 3), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf101, (256, 4, 1, 1), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256, 512, 1, 1), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256,), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf104, (4, 512, 1, 1), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf105, (256, 4, 1, 1), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf106, (256,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf107, (256,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (256, 256, 3, 3), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf109, (256,), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf110, (4, 256, 3, 3), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 4, 1, 1), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf113, (256,), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf114, (256, 256, 3, 3), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf115, (256,), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf116, (4, 256, 3, 3), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf117, (256, 4, 1, 1), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf118, (256,), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf119, (256,), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256, 256, 3, 3), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256,), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf122, (4, 256, 3, 3), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256, 4, 1, 1), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256, 256, 3, 3), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf128, (4, 256, 3, 3), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256, 4, 1, 1), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256, 256, 3, 3), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256,), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf132, (4, 256, 3, 3), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256, 4, 1, 1), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf135, (256,), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf136, (128, 256, 3, 3), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128,), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf138, (4, 256, 3, 3), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (128, 4, 1, 1), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf140, (128,), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf141, (128,), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128, 128, 3, 3), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf143, (128,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf144, (4, 128, 3, 3), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf145, (128, 4, 1, 1), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (128, 256, 1, 1), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf147, (128,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf148, (4, 256, 1, 1), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf149, (128, 4, 1, 1), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf150, (128,), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf151, (128,), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf152, (128, 128, 3, 3), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf153, (128,), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf154, (4, 128, 3, 3), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf155, (128, 4, 1, 1), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf156, (128,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf157, (128,), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf158, (128, 128, 3, 3), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf159, (128,), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf160, (4, 128, 3, 3), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (128, 4, 1, 1), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf162, (128,), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf163, (128,), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf164, (128, 128, 3, 3), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf165, (128,), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf166, (4, 128, 3, 3), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf167, (128, 4, 1, 1), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf168, (128,), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf169, (128,), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf170, (128, 128, 3, 3), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf171, (128,), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf172, (4, 128, 3, 3), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf173, (128, 4, 1, 1), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf176, (512, 512, 3, 3), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf177, (512,), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf178, (4, 512, 3, 3), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf179, (512, 4, 1, 1), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512,), is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512, 512, 3, 3), is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf184, (4, 512, 3, 3), is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf185, (512, 4, 1, 1), is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512,), is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512, 512), is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512,), is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf190, (4, 512), is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf191, (512, 4), is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf192, (512, 512), is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf193, (512,), is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4, 512), is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf195, (512, 4), is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf196, (512, 512), is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf197, (512,), is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf198, (4, 512), is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf199, (512, 4), is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf200, (512, 512), is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf201, (512,), is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf202, (4, 512), is_leaf=True)  # primals_203
    buf203 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512, 4), is_leaf=True)  # primals_204
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # primals_205
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # primals_206
    buf206 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512, 512, 3, 3), is_leaf=True)  # primals_207
    buf207 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf207, (512,), is_leaf=True)  # primals_208
    buf208 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf208, (4, 512, 3, 3), is_leaf=True)  # primals_209
    buf209 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512, 4, 1, 1), is_leaf=True)  # primals_210
    buf210 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512,), is_leaf=True)  # primals_211
    buf211 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512,), is_leaf=True)  # primals_212
    buf212 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512, 512, 3, 3), is_leaf=True)  # primals_213
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # primals_214
    buf214 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf214, (4, 512, 3, 3), is_leaf=True)  # primals_215
    buf215 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf215, (512, 4, 1, 1), is_leaf=True)  # primals_216
    buf216 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf216, (4, 512, 32, 32), is_leaf=True)  # primals_217
    buf217 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512, 512, 1, 1), is_leaf=True)  # primals_218
    buf218 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf218, (4, 512, 1, 1), is_leaf=True)  # primals_219
    buf219 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf219, (512, 4, 1, 1), is_leaf=True)  # primals_220
    buf220 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf220, (4, 256, 64, 64), is_leaf=True)  # primals_221
    buf221 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512, 256, 1, 1), is_leaf=True)  # primals_222
    buf222 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf222, (4, 256, 1, 1), is_leaf=True)  # primals_223
    buf223 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512, 4, 1, 1), is_leaf=True)  # primals_224
    buf224 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf224, (4, 128, 128, 128), is_leaf=True)  # primals_225
    buf225 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf225, (512, 128, 1, 1), is_leaf=True)  # primals_226
    buf226 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf226, (4, 128, 1, 1), is_leaf=True)  # primals_227
    buf227 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf227, (512, 4, 1, 1), is_leaf=True)  # primals_228
    buf228 = reader.storage(None, 134217728, device=device(type='cuda', index=0))
    reader.tensor(buf228, (4, 128, 256, 256), is_leaf=True)  # primals_229
    buf229 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf229, (256, 128, 1, 1), is_leaf=True)  # primals_230
    buf230 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf230, (4, 128, 1, 1), is_leaf=True)  # primals_231
    buf231 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf231, (256, 4, 1, 1), is_leaf=True)  # primals_232
    buf232 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf232, (128,), is_leaf=True)  # primals_233
    buf233 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf233, (128,), is_leaf=True)  # primals_234
    buf234 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf234, (3, 128, 3, 3), is_leaf=True)  # primals_235
    buf235 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf235, (3,), is_leaf=True)  # primals_236
    buf236 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf236, (4, 128, 3, 3), is_leaf=True)  # primals_237
    buf237 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf237, (3, 4, 1, 1), is_leaf=True)  # primals_238
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)