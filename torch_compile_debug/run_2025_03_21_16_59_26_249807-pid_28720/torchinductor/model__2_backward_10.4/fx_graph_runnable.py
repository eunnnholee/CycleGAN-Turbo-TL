
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
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



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

    
    
    def forward(self, primals_4, primals_6, primals_7, primals_8, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, inductor_random_default_1, inductor_random_default, view, clamp_max, clamp_max_1, index_put, add_10, device_put_1, div, cat, getitem_1, rsqrt, getitem_3, rsqrt_1, view_13, view_14, view_15, getitem_4, getitem_5, getitem_6, getitem_7, mul_9, addmm_2, mul_13, view_28, view_29, view_30, getitem_12, getitem_13, getitem_14, getitem_15, mul_15, addmm_6, mul_19, view_43, view_44, view_45, getitem_20, getitem_21, getitem_22, getitem_23, mul_21, addmm_10, mul_25, view_58, view_59, view_60, getitem_28, getitem_29, getitem_30, getitem_31, mul_27, addmm_14, add_38, getitem_35, rsqrt_9, view_73, view_74, view_75, getitem_36, getitem_37, getitem_38, getitem_39, mul_33, addmm_18, mul_37, view_88, view_89, view_90, getitem_44, getitem_45, getitem_46, getitem_47, mul_39, addmm_22, mul_43, view_103, view_104, view_105, getitem_52, getitem_53, getitem_54, getitem_55, mul_45, addmm_26, mul_49, view_118, view_119, view_120, getitem_60, getitem_61, getitem_62, getitem_63, mul_51, addmm_30, add_62, getitem_67, rsqrt_17, view_133, view_134, view_135, getitem_68, getitem_69, getitem_70, getitem_71, mul_57, addmm_34, mul_61, view_148, view_149, view_150, getitem_76, getitem_77, getitem_78, getitem_79, mul_63, addmm_38, mul_67, view_163, view_164, view_165, getitem_84, getitem_85, getitem_86, getitem_87, mul_69, addmm_42, mul_73, view_178, view_179, view_180, getitem_92, getitem_93, getitem_94, getitem_95, mul_75, addmm_46, mul_79, mm, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, permute_129, gt_4, gt_5, permute_135, div_39, permute_137, permute_138, div_40, permute_139, permute_145, div_41, permute_146, permute_147, div_42, permute_148, permute_154, div_43, permute_155, permute_156, div_44, permute_157, permute_163, div_45, permute_164, permute_165, div_46, permute_166, permute_172, permute_174, permute_175, div_48, permute_176, permute_182, div_49, permute_183, permute_184, div_50, permute_185, permute_191, div_51, permute_192, permute_193, div_52, permute_194, permute_200, div_53, permute_201, permute_202, div_54, permute_203, permute_209, permute_211, permute_212, div_56, permute_213, permute_219, div_57, permute_220, permute_221, div_58, permute_222, permute_228, div_59, permute_229, permute_230, div_60, permute_231, permute_237, div_61, permute_238, permute_239, div_62, permute_240, permute_246, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7):
        full_default_9 = torch.ops.aten.full.default([2, 1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sigmoid_12 = torch.ops.aten.sigmoid.default(addmm_49);  addmm_49 = None
        sub_41 = torch.ops.aten.sub.Tensor(sigmoid_12, full_default_9);  sigmoid_12 = full_default_9 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_41, tangents_1);  sub_41 = None
        view_197 = torch.ops.aten.view.default(tangents_1, [2]);  tangents_1 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(view_197, 1);  view_197 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        expand_33 = torch.ops.aten.expand.default(unsqueeze_19, [2, 3, 3]);  unsqueeze_19 = None
        div_19 = torch.ops.aten.div.Scalar(expand_33, 9);  expand_33 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(convolution_6, 1);  convolution_6 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(squeeze_13);  squeeze_13 = None
        full_default_3 = torch.ops.aten.full.default([2, 3, 3], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_42 = torch.ops.aten.sub.Tensor(sigmoid_13, full_default_3);  sigmoid_13 = None
        mul_113 = torch.ops.aten.mul.Tensor(sub_42, div_19);  sub_42 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(convolution_3, 1);  convolution_3 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(squeeze_12);  squeeze_12 = None
        sub_43 = torch.ops.aten.sub.Tensor(sigmoid_14, full_default_3);  sigmoid_14 = full_default_3 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_43, div_19);  sub_43 = div_19 = None
        mm_1 = torch.ops.aten.mm.default(mul_112, permute_125);  permute_125 = None
        permute_126 = torch.ops.aten.permute.default(mul_112, [1, 0])
        mm_2 = torch.ops.aten.mm.default(permute_126, where_2);  permute_126 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_112, [0], True);  mul_112 = None
        view_199 = torch.ops.aten.view.default(sum_37, [1]);  sum_37 = None
        add_93 = torch.ops.aten.add.Tensor(tangents_7, mm_2);  tangents_7 = mm_2 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(sum_33, 2.0)
        sum_34 = torch.ops.aten.sum.dim_IntList(pow_23, [0], True);  pow_23 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
        clamp_min_15 = torch.ops.aten.clamp_min.default(pow_24, 1e-12);  pow_24 = None
        expand_29 = torch.ops.aten.expand.default(clamp_min_15, [1]);  clamp_min_15 = None
        div_17 = torch.ops.aten.div.Tensor(sum_33, expand_29);  expand_29 = None
        mul_107 = torch.ops.aten.mul.Tensor(div_17, sum_33);  sum_33 = None
        sum_36 = torch.ops.aten.sum.default(mul_107);  mul_107 = None
        div_21 = torch.ops.aten.div.Tensor(primals_178, sum_36)
        div_22 = torch.ops.aten.div.Tensor(div_21, sum_36);  div_21 = None
        neg_3 = torch.ops.aten.neg.default(add_93)
        mul_115 = torch.ops.aten.mul.Tensor(neg_3, div_22);  neg_3 = div_22 = None
        div_23 = torch.ops.aten.div.Tensor(add_93, sum_36);  add_93 = sum_36 = None
        sum_38 = torch.ops.aten.sum.default(mul_115);  mul_115 = None
        mul_116 = torch.ops.aten.mul.Tensor(sum_38, div_17);  sum_38 = None
        view_200 = torch.ops.aten.view.default(mul_116, [1, 1]);  mul_116 = None
        view_194 = torch.ops.aten.view.default(primals_178, [1, -1]);  primals_178 = None
        permute_123 = torch.ops.aten.permute.default(view_194, [1, 0]);  view_194 = None
        mul_104 = torch.ops.aten.mul.Tensor(permute_123, primals_179);  permute_123 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_104, [1]);  mul_104 = None
        expand_27 = torch.ops.aten.expand.default(clamp_min_14, [256]);  clamp_min_14 = None
        div_16 = torch.ops.aten.div.Tensor(sum_31, expand_27);  sum_31 = expand_27 = None
        mul_117 = torch.ops.aten.mul.Tensor(view_200, div_16);  view_200 = div_16 = None
        add_94 = torch.ops.aten.add.Tensor(div_23, mul_117);  div_23 = mul_117 = None
        gt_3 = torch.ops.aten.gt.Scalar(where_2, 0);  where_2 = None
        mul_118 = torch.ops.aten.mul.Tensor(mm_1, 0.2)
        where_3 = torch.ops.aten.where.self(gt_3, mm_1, mul_118);  gt_3 = mm_1 = mul_118 = None
        mm_3 = torch.ops.aten.mm.default(where_3, permute_129);  permute_129 = None
        permute_130 = torch.ops.aten.permute.default(where_3, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_130, mm);  permute_130 = mm = None
        sum_39 = torch.ops.aten.sum.dim_IntList(where_3, [0], True);  where_3 = None
        view_202 = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
        add_95 = torch.ops.aten.add.Tensor(tangents_6, mm_4);  tangents_6 = mm_4 = None
        div_24 = torch.ops.aten.div.Tensor(primals_174, sum_30);  primals_174 = None
        div_25 = torch.ops.aten.div.Tensor(div_24, sum_30);  div_24 = None
        neg_4 = torch.ops.aten.neg.default(add_95)
        mul_119 = torch.ops.aten.mul.Tensor(neg_4, div_25);  neg_4 = div_25 = None
        div_26 = torch.ops.aten.div.Tensor(add_95, sum_30);  add_95 = sum_30 = None
        sum_40 = torch.ops.aten.sum.default(mul_119);  mul_119 = None
        mul_120 = torch.ops.aten.mul.Tensor(sum_40, div_14);  sum_40 = div_14 = None
        view_203 = torch.ops.aten.view.default(mul_120, [256, 1]);  mul_120 = None
        mul_121 = torch.ops.aten.mul.Tensor(view_203, div_13);  view_203 = div_13 = None
        add_96 = torch.ops.aten.add.Tensor(div_26, mul_121);  div_26 = mul_121 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(mul_113, 1);  mul_113 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(unsqueeze_22, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(unsqueeze_22, convolution_5, div_12, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  unsqueeze_22 = convolution_5 = div_12 = None
        getitem_100 = convolution_backward[0]
        getitem_101 = convolution_backward[1];  convolution_backward = None
        add_97 = torch.ops.aten.add.Tensor(tangents_5, getitem_101);  tangents_5 = getitem_101 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(sum_21, 2.0)
        sum_22 = torch.ops.aten.sum.dim_IntList(pow_15, [0], True);  pow_15 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
        clamp_min_11 = torch.ops.aten.clamp_min.default(pow_16, 1e-12);  pow_16 = None
        expand_21 = torch.ops.aten.expand.default(clamp_min_11, [1]);  clamp_min_11 = None
        div_11 = torch.ops.aten.div.Tensor(sum_21, expand_21);  expand_21 = None
        mul_98 = torch.ops.aten.mul.Tensor(div_11, sum_21);  sum_21 = None
        sum_24 = torch.ops.aten.sum.default(mul_98);  mul_98 = None
        div_27 = torch.ops.aten.div.Tensor(primals_170, sum_24)
        div_28 = torch.ops.aten.div.Tensor(div_27, sum_24);  div_27 = None
        neg_5 = torch.ops.aten.neg.default(add_97)
        mul_122 = torch.ops.aten.mul.Tensor(neg_5, div_28);  neg_5 = div_28 = None
        div_29 = torch.ops.aten.div.Tensor(add_97, sum_24);  add_97 = sum_24 = None
        sum_42 = torch.ops.aten.sum.default(mul_122);  mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(sum_42, div_11);  sum_42 = None
        view_205 = torch.ops.aten.view.default(mul_123, [1, 1]);  mul_123 = None
        view_192 = torch.ops.aten.view.default(primals_170, [1, -1]);  primals_170 = None
        permute_120 = torch.ops.aten.permute.default(view_192, [1, 0]);  view_192 = None
        mul_95 = torch.ops.aten.mul.Tensor(permute_120, primals_171);  permute_120 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_95, [1]);  mul_95 = None
        expand_19 = torch.ops.aten.expand.default(clamp_min_10, [256]);  clamp_min_10 = None
        div_10 = torch.ops.aten.div.Tensor(sum_19, expand_19);  sum_19 = expand_19 = None
        mul_124 = torch.ops.aten.mul.Tensor(view_205, div_10);  view_205 = div_10 = None
        view_206 = torch.ops.aten.view.default(mul_124, [1, 256, 1, 1]);  mul_124 = None
        add_98 = torch.ops.aten.add.Tensor(div_29, view_206);  div_29 = view_206 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_100, constant_pad_nd_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False]);  getitem_100 = constant_pad_nd_2 = primals_169 = None
        getitem_103 = convolution_backward_1[0];  convolution_backward_1 = None
        constant_pad_nd_3 = torch.ops.aten.constant_pad_nd.default(getitem_103, [-1, -1, -1, -1]);  getitem_103 = None
        mul_125 = torch.ops.aten.mul.Tensor(constant_pad_nd_3, 0.2)
        where_4 = torch.ops.aten.where.self(gt_4, constant_pad_nd_3, mul_125);  gt_4 = constant_pad_nd_3 = mul_125 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        permute_77 = torch.ops.aten.permute.default(add_62, [1, 0, 2])
        slice_9 = torch.ops.aten.slice.Tensor(permute_77, 1, 1, 9223372036854775807);  permute_77 = None
        permute_116 = torch.ops.aten.permute.default(slice_9, [0, 2, 1]);  slice_9 = None
        view_188 = torch.ops.aten.view.default(permute_116, [2, 768, 7, 7]);  permute_116 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, view_188, div_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = view_188 = div_9 = None
        getitem_106 = convolution_backward_2[0]
        getitem_107 = convolution_backward_2[1];  convolution_backward_2 = None
        add_99 = torch.ops.aten.add.Tensor(tangents_4, getitem_107);  tangents_4 = getitem_107 = None
        div_30 = torch.ops.aten.div.Tensor(primals_165, sum_18);  primals_165 = None
        div_31 = torch.ops.aten.div.Tensor(div_30, sum_18);  div_30 = None
        neg_6 = torch.ops.aten.neg.default(add_99)
        mul_126 = torch.ops.aten.mul.Tensor(neg_6, div_31);  neg_6 = div_31 = None
        div_32 = torch.ops.aten.div.Tensor(add_99, sum_18);  add_99 = sum_18 = None
        sum_44 = torch.ops.aten.sum.default(mul_126);  mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(sum_44, div_8);  sum_44 = div_8 = None
        view_207 = torch.ops.aten.view.default(mul_127, [256, 1]);  mul_127 = None
        mul_128 = torch.ops.aten.mul.Tensor(view_207, div_7);  view_207 = div_7 = None
        view_208 = torch.ops.aten.view.default(mul_128, [256, 768, 3, 3]);  mul_128 = None
        add_100 = torch.ops.aten.add.Tensor(div_32, view_208);  div_32 = view_208 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(mul_114, 1);  mul_114 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(unsqueeze_23, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(unsqueeze_23, convolution_2, div_6, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  unsqueeze_23 = convolution_2 = div_6 = None
        getitem_109 = convolution_backward_3[0]
        getitem_110 = convolution_backward_3[1];  convolution_backward_3 = None
        add_101 = torch.ops.aten.add.Tensor(tangents_3, getitem_110);  tangents_3 = getitem_110 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sum_9, 2.0)
        sum_10 = torch.ops.aten.sum.dim_IntList(pow_7, [0], True);  pow_7 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(pow_8, 1e-12);  pow_8 = None
        expand_13 = torch.ops.aten.expand.default(clamp_min_7, [1]);  clamp_min_7 = None
        div_5 = torch.ops.aten.div.Tensor(sum_9, expand_13);  expand_13 = None
        mul_89 = torch.ops.aten.mul.Tensor(div_5, sum_9);  sum_9 = None
        sum_12 = torch.ops.aten.sum.default(mul_89);  mul_89 = None
        div_33 = torch.ops.aten.div.Tensor(primals_161, sum_12)
        div_34 = torch.ops.aten.div.Tensor(div_33, sum_12);  div_33 = None
        neg_7 = torch.ops.aten.neg.default(add_101)
        mul_129 = torch.ops.aten.mul.Tensor(neg_7, div_34);  neg_7 = div_34 = None
        div_35 = torch.ops.aten.div.Tensor(add_101, sum_12);  add_101 = sum_12 = None
        sum_46 = torch.ops.aten.sum.default(mul_129);  mul_129 = None
        mul_130 = torch.ops.aten.mul.Tensor(sum_46, div_5);  sum_46 = None
        view_209 = torch.ops.aten.view.default(mul_130, [1, 1]);  mul_130 = None
        view_190 = torch.ops.aten.view.default(primals_161, [1, -1]);  primals_161 = None
        permute_118 = torch.ops.aten.permute.default(view_190, [1, 0]);  view_190 = None
        mul_86 = torch.ops.aten.mul.Tensor(permute_118, primals_162);  permute_118 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_86, [1]);  mul_86 = None
        expand_11 = torch.ops.aten.expand.default(clamp_min_6, [256]);  clamp_min_6 = None
        div_4 = torch.ops.aten.div.Tensor(sum_7, expand_11);  sum_7 = expand_11 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_209, div_4);  view_209 = div_4 = None
        view_210 = torch.ops.aten.view.default(mul_131, [1, 256, 1, 1]);  mul_131 = None
        add_102 = torch.ops.aten.add.Tensor(div_35, view_210);  div_35 = view_210 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(getitem_109, constant_pad_nd_1, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False]);  getitem_109 = constant_pad_nd_1 = primals_160 = None
        getitem_112 = convolution_backward_4[0];  convolution_backward_4 = None
        constant_pad_nd_4 = torch.ops.aten.constant_pad_nd.default(getitem_112, [-1, -1, -1, -1]);  getitem_112 = None
        mul_132 = torch.ops.aten.mul.Tensor(constant_pad_nd_4, 0.2)
        where_5 = torch.ops.aten.where.self(gt_5, constant_pad_nd_4, mul_132);  gt_5 = constant_pad_nd_4 = mul_132 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        permute_40 = torch.ops.aten.permute.default(add_38, [1, 0, 2])
        slice_6 = torch.ops.aten.slice.Tensor(permute_40, 1, 1, 9223372036854775807);  permute_40 = None
        permute_115 = torch.ops.aten.permute.default(slice_6, [0, 2, 1]);  slice_6 = None
        view_187 = torch.ops.aten.view.default(permute_115, [2, 768, 7, 7]);  permute_115 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, view_187, div_3, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = view_187 = div_3 = None
        getitem_115 = convolution_backward_5[0]
        getitem_116 = convolution_backward_5[1];  convolution_backward_5 = None
        add_103 = torch.ops.aten.add.Tensor(tangents_2, getitem_116);  tangents_2 = getitem_116 = None
        div_36 = torch.ops.aten.div.Tensor(primals_156, sum_6);  primals_156 = None
        div_37 = torch.ops.aten.div.Tensor(div_36, sum_6);  div_36 = None
        neg_8 = torch.ops.aten.neg.default(add_103)
        mul_133 = torch.ops.aten.mul.Tensor(neg_8, div_37);  neg_8 = div_37 = None
        div_38 = torch.ops.aten.div.Tensor(add_103, sum_6);  add_103 = sum_6 = None
        sum_48 = torch.ops.aten.sum.default(mul_133);  mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(sum_48, div_2);  sum_48 = div_2 = None
        view_211 = torch.ops.aten.view.default(mul_134, [256, 1]);  mul_134 = None
        mul_135 = torch.ops.aten.mul.Tensor(view_211, div_1);  view_211 = div_1 = None
        view_212 = torch.ops.aten.view.default(mul_135, [256, 768, 3, 3]);  mul_135 = None
        add_104 = torch.ops.aten.add.Tensor(div_38, view_212);  div_38 = view_212 = None
        view_213 = torch.ops.aten.view.default(getitem_106, [2, 768, 49]);  getitem_106 = None
        permute_133 = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        full_default_13 = torch.ops.aten.full.default([2, 50, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(full_default_13, permute_133, 1, 1, 9223372036854775807);  permute_133 = None
        view_214 = torch.ops.aten.view.default(getitem_115, [2, 768, 49]);  getitem_115 = None
        permute_134 = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(full_default_13, permute_134, 1, 1, 9223372036854775807);  permute_134 = None
        mm_5 = torch.ops.aten.mm.default(mm_3, permute_135);  mm_3 = permute_135 = None
        mul_137 = torch.ops.aten.mul.Tensor(mm_5, primals_153);  mm_5 = primals_153 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, 768)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_137, [1], True)
        mul_139 = torch.ops.aten.mul.Tensor(mul_137, mul_79);  mul_137 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_139, [1], True);  mul_139 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_79, sum_50);  mul_79 = sum_50 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_138, sum_49);  mul_138 = sum_49 = None
        sub_46 = torch.ops.aten.sub.Tensor(sub_45, mul_140);  sub_45 = mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(div_39, sub_46);  div_39 = sub_46 = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default_13, mul_141, 1, 0);  full_default_13 = mul_141 = None
        permute_136 = torch.ops.aten.permute.default(select_scatter, [1, 0, 2]);  select_scatter = None
        clone_64 = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format)
        view_215 = torch.ops.aten.view.default(clone_64, [100, 768]);  clone_64 = None
        mm_6 = torch.ops.aten.mm.default(view_215, permute_137);  view_215 = permute_137 = None
        view_216 = torch.ops.aten.view.default(mm_6, [50, 2, 3072]);  mm_6 = None
        view_184 = torch.ops.aten.view.default(addmm_46, [50, 2, 3072]);  addmm_46 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_216, view_184)
        mul_77 = torch.ops.aten.mul.Tensor(view_184, 1.702);  view_184 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(mul_77);  mul_77 = None
        mul_143 = torch.ops.aten.mul.Tensor(view_216, sigmoid_11);  view_216 = None
        sub_47 = torch.ops.aten.sub.Tensor(1, sigmoid_11)
        mul_144 = torch.ops.aten.mul.Tensor(sigmoid_11, sub_47);  sigmoid_11 = sub_47 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_142, mul_144);  mul_142 = mul_144 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, 1.702);  mul_145 = None
        add_105 = torch.ops.aten.add.Tensor(mul_143, mul_146);  mul_143 = mul_146 = None
        view_217 = torch.ops.aten.view.default(add_105, [100, 3072]);  add_105 = None
        mm_7 = torch.ops.aten.mm.default(view_217, permute_138);  view_217 = permute_138 = None
        view_218 = torch.ops.aten.view.default(mm_7, [50, 2, 768]);  mm_7 = None
        mul_148 = torch.ops.aten.mul.Tensor(view_218, primals_147);  view_218 = primals_147 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, 768)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150 = torch.ops.aten.mul.Tensor(mul_148, mul_75);  mul_148 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_75, sum_52);  mul_75 = sum_52 = None
        sub_49 = torch.ops.aten.sub.Tensor(mul_149, sum_51);  mul_149 = sum_51 = None
        sub_50 = torch.ops.aten.sub.Tensor(sub_49, mul_151);  sub_49 = mul_151 = None
        mul_152 = torch.ops.aten.mul.Tensor(div_40, sub_50);  div_40 = sub_50 = None
        add_106 = torch.ops.aten.add.Tensor(permute_136, mul_152);  permute_136 = mul_152 = None
        clone_66 = torch.ops.aten.clone.default(add_106, memory_format = torch.contiguous_format)
        view_219 = torch.ops.aten.view.default(clone_66, [100, 768]);  clone_66 = None
        mm_8 = torch.ops.aten.mm.default(view_219, permute_139);  view_219 = permute_139 = None
        view_220 = torch.ops.aten.view.default(mm_8, [50, 2, 12, 64]);  mm_8 = None
        permute_140 = torch.ops.aten.permute.default(view_220, [1, 2, 0, 3]);  view_220 = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_140, view_178, view_179, view_180, None, getitem_92, getitem_93, getitem_94, getitem_95, 0.0, [True, True, True, False]);  permute_140 = view_178 = view_179 = view_180 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = None
        getitem_118 = _scaled_dot_product_efficient_attention_backward[0]
        getitem_119 = _scaled_dot_product_efficient_attention_backward[1]
        getitem_120 = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
        clone_67 = torch.ops.aten.clone.default(getitem_120, memory_format = torch.contiguous_format);  getitem_120 = None
        view_221 = torch.ops.aten.view.default(clone_67, [24, 50, 64]);  clone_67 = None
        clone_68 = torch.ops.aten.clone.default(getitem_119, memory_format = torch.contiguous_format);  getitem_119 = None
        view_222 = torch.ops.aten.view.default(clone_68, [24, 50, 64]);  clone_68 = None
        clone_69 = torch.ops.aten.clone.default(getitem_118, memory_format = torch.contiguous_format);  getitem_118 = None
        view_223 = torch.ops.aten.view.default(clone_69, [24, 50, 64]);  clone_69 = None
        permute_141 = torch.ops.aten.permute.default(view_221, [1, 0, 2]);  view_221 = None
        clone_70 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_224 = torch.ops.aten.view.default(clone_70, [50, 2, 768]);  clone_70 = None
        permute_142 = torch.ops.aten.permute.default(view_222, [1, 0, 2]);  view_222 = None
        clone_71 = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_225 = torch.ops.aten.view.default(clone_71, [50, 2, 768]);  clone_71 = None
        permute_143 = torch.ops.aten.permute.default(view_223, [1, 0, 2]);  view_223 = None
        clone_72 = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_226 = torch.ops.aten.view.default(clone_72, [50, 2, 768]);  clone_72 = None
        full_default_21 = torch.ops.aten.full.default([3, 50, 2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_21, view_224, 0, 2);  view_224 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default_21, view_225, 0, 1);  view_225 = None
        add_107 = torch.ops.aten.add.Tensor(select_scatter_1, select_scatter_2);  select_scatter_1 = select_scatter_2 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default_21, view_226, 0, 0);  view_226 = None
        add_108 = torch.ops.aten.add.Tensor(add_107, select_scatter_3);  add_107 = select_scatter_3 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(add_108, 3);  add_108 = None
        permute_144 = torch.ops.aten.permute.default(unsqueeze_24, [3, 1, 2, 0, 4]);  unsqueeze_24 = None
        squeeze_14 = torch.ops.aten.squeeze.dim(permute_144, 0);  permute_144 = None
        clone_73 = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_227 = torch.ops.aten.view.default(clone_73, [50, 2, 2304]);  clone_73 = None
        view_228 = torch.ops.aten.view.default(view_227, [100, 2304]);  view_227 = None
        mm_9 = torch.ops.aten.mm.default(view_228, permute_145);  view_228 = permute_145 = None
        view_229 = torch.ops.aten.view.default(mm_9, [50, 2, 768]);  mm_9 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_229, primals_141);  view_229 = primals_141 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_154, 768)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
        mul_156 = torch.ops.aten.mul.Tensor(mul_154, mul_73);  mul_154 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_73, sum_54);  mul_73 = sum_54 = None
        sub_52 = torch.ops.aten.sub.Tensor(mul_155, sum_53);  mul_155 = sum_53 = None
        sub_53 = torch.ops.aten.sub.Tensor(sub_52, mul_157);  sub_52 = mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(div_41, sub_53);  div_41 = sub_53 = None
        add_109 = torch.ops.aten.add.Tensor(add_106, mul_158);  add_106 = mul_158 = None
        clone_75 = torch.ops.aten.clone.default(add_109, memory_format = torch.contiguous_format)
        view_230 = torch.ops.aten.view.default(clone_75, [100, 768]);  clone_75 = None
        mm_10 = torch.ops.aten.mm.default(view_230, permute_146);  view_230 = permute_146 = None
        view_231 = torch.ops.aten.view.default(mm_10, [50, 2, 3072]);  mm_10 = None
        view_169 = torch.ops.aten.view.default(addmm_42, [50, 2, 3072]);  addmm_42 = None
        mul_159 = torch.ops.aten.mul.Tensor(view_231, view_169)
        mul_71 = torch.ops.aten.mul.Tensor(view_169, 1.702);  view_169 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(mul_71);  mul_71 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_231, sigmoid_10);  view_231 = None
        sub_54 = torch.ops.aten.sub.Tensor(1, sigmoid_10)
        mul_161 = torch.ops.aten.mul.Tensor(sigmoid_10, sub_54);  sigmoid_10 = sub_54 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_159, mul_161);  mul_159 = mul_161 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, 1.702);  mul_162 = None
        add_110 = torch.ops.aten.add.Tensor(mul_160, mul_163);  mul_160 = mul_163 = None
        view_232 = torch.ops.aten.view.default(add_110, [100, 3072]);  add_110 = None
        mm_11 = torch.ops.aten.mm.default(view_232, permute_147);  view_232 = permute_147 = None
        view_233 = torch.ops.aten.view.default(mm_11, [50, 2, 768]);  mm_11 = None
        mul_165 = torch.ops.aten.mul.Tensor(view_233, primals_135);  view_233 = primals_135 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, 768)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_165, [2], True)
        mul_167 = torch.ops.aten.mul.Tensor(mul_165, mul_69);  mul_165 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_167, [2], True);  mul_167 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_69, sum_56);  mul_69 = sum_56 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_166, sum_55);  mul_166 = sum_55 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, mul_168);  sub_56 = mul_168 = None
        mul_169 = torch.ops.aten.mul.Tensor(div_42, sub_57);  div_42 = sub_57 = None
        add_111 = torch.ops.aten.add.Tensor(add_109, mul_169);  add_109 = mul_169 = None
        clone_77 = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
        view_234 = torch.ops.aten.view.default(clone_77, [100, 768]);  clone_77 = None
        mm_12 = torch.ops.aten.mm.default(view_234, permute_148);  view_234 = permute_148 = None
        view_235 = torch.ops.aten.view.default(mm_12, [50, 2, 12, 64]);  mm_12 = None
        permute_149 = torch.ops.aten.permute.default(view_235, [1, 2, 0, 3]);  view_235 = None
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_149, view_163, view_164, view_165, None, getitem_84, getitem_85, getitem_86, getitem_87, 0.0, [True, True, True, False]);  permute_149 = view_163 = view_164 = view_165 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = None
        getitem_122 = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_123 = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_124 = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
        clone_78 = torch.ops.aten.clone.default(getitem_124, memory_format = torch.contiguous_format);  getitem_124 = None
        view_236 = torch.ops.aten.view.default(clone_78, [24, 50, 64]);  clone_78 = None
        clone_79 = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format);  getitem_123 = None
        view_237 = torch.ops.aten.view.default(clone_79, [24, 50, 64]);  clone_79 = None
        clone_80 = torch.ops.aten.clone.default(getitem_122, memory_format = torch.contiguous_format);  getitem_122 = None
        view_238 = torch.ops.aten.view.default(clone_80, [24, 50, 64]);  clone_80 = None
        permute_150 = torch.ops.aten.permute.default(view_236, [1, 0, 2]);  view_236 = None
        clone_81 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_239 = torch.ops.aten.view.default(clone_81, [50, 2, 768]);  clone_81 = None
        permute_151 = torch.ops.aten.permute.default(view_237, [1, 0, 2]);  view_237 = None
        clone_82 = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
        view_240 = torch.ops.aten.view.default(clone_82, [50, 2, 768]);  clone_82 = None
        permute_152 = torch.ops.aten.permute.default(view_238, [1, 0, 2]);  view_238 = None
        clone_83 = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_241 = torch.ops.aten.view.default(clone_83, [50, 2, 768]);  clone_83 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(full_default_21, view_239, 0, 2);  view_239 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(full_default_21, view_240, 0, 1);  view_240 = None
        add_112 = torch.ops.aten.add.Tensor(select_scatter_4, select_scatter_5);  select_scatter_4 = select_scatter_5 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(full_default_21, view_241, 0, 0);  view_241 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, select_scatter_6);  add_112 = select_scatter_6 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(add_113, 3);  add_113 = None
        permute_153 = torch.ops.aten.permute.default(unsqueeze_25, [3, 1, 2, 0, 4]);  unsqueeze_25 = None
        squeeze_15 = torch.ops.aten.squeeze.dim(permute_153, 0);  permute_153 = None
        clone_84 = torch.ops.aten.clone.default(squeeze_15, memory_format = torch.contiguous_format);  squeeze_15 = None
        view_242 = torch.ops.aten.view.default(clone_84, [50, 2, 2304]);  clone_84 = None
        view_243 = torch.ops.aten.view.default(view_242, [100, 2304]);  view_242 = None
        mm_13 = torch.ops.aten.mm.default(view_243, permute_154);  view_243 = permute_154 = None
        view_244 = torch.ops.aten.view.default(mm_13, [50, 2, 768]);  mm_13 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_244, primals_129);  view_244 = primals_129 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, 768)
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
        mul_173 = torch.ops.aten.mul.Tensor(mul_171, mul_67);  mul_171 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_67, sum_58);  mul_67 = sum_58 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_172, sum_57);  mul_172 = sum_57 = None
        sub_60 = torch.ops.aten.sub.Tensor(sub_59, mul_174);  sub_59 = mul_174 = None
        mul_175 = torch.ops.aten.mul.Tensor(div_43, sub_60);  div_43 = sub_60 = None
        add_114 = torch.ops.aten.add.Tensor(add_111, mul_175);  add_111 = mul_175 = None
        clone_86 = torch.ops.aten.clone.default(add_114, memory_format = torch.contiguous_format)
        view_245 = torch.ops.aten.view.default(clone_86, [100, 768]);  clone_86 = None
        mm_14 = torch.ops.aten.mm.default(view_245, permute_155);  view_245 = permute_155 = None
        view_246 = torch.ops.aten.view.default(mm_14, [50, 2, 3072]);  mm_14 = None
        view_154 = torch.ops.aten.view.default(addmm_38, [50, 2, 3072]);  addmm_38 = None
        mul_176 = torch.ops.aten.mul.Tensor(view_246, view_154)
        mul_65 = torch.ops.aten.mul.Tensor(view_154, 1.702);  view_154 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(mul_65);  mul_65 = None
        mul_177 = torch.ops.aten.mul.Tensor(view_246, sigmoid_9);  view_246 = None
        sub_61 = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_178 = torch.ops.aten.mul.Tensor(sigmoid_9, sub_61);  sigmoid_9 = sub_61 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_176, mul_178);  mul_176 = mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, 1.702);  mul_179 = None
        add_115 = torch.ops.aten.add.Tensor(mul_177, mul_180);  mul_177 = mul_180 = None
        view_247 = torch.ops.aten.view.default(add_115, [100, 3072]);  add_115 = None
        mm_15 = torch.ops.aten.mm.default(view_247, permute_156);  view_247 = permute_156 = None
        view_248 = torch.ops.aten.view.default(mm_15, [50, 2, 768]);  mm_15 = None
        mul_182 = torch.ops.aten.mul.Tensor(view_248, primals_123);  view_248 = primals_123 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_182, 768)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_182, [2], True)
        mul_184 = torch.ops.aten.mul.Tensor(mul_182, mul_63);  mul_182 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_184, [2], True);  mul_184 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_63, sum_60);  mul_63 = sum_60 = None
        sub_63 = torch.ops.aten.sub.Tensor(mul_183, sum_59);  mul_183 = sum_59 = None
        sub_64 = torch.ops.aten.sub.Tensor(sub_63, mul_185);  sub_63 = mul_185 = None
        mul_186 = torch.ops.aten.mul.Tensor(div_44, sub_64);  div_44 = sub_64 = None
        add_116 = torch.ops.aten.add.Tensor(add_114, mul_186);  add_114 = mul_186 = None
        clone_88 = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
        view_249 = torch.ops.aten.view.default(clone_88, [100, 768]);  clone_88 = None
        mm_16 = torch.ops.aten.mm.default(view_249, permute_157);  view_249 = permute_157 = None
        view_250 = torch.ops.aten.view.default(mm_16, [50, 2, 12, 64]);  mm_16 = None
        permute_158 = torch.ops.aten.permute.default(view_250, [1, 2, 0, 3]);  view_250 = None
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_158, view_148, view_149, view_150, None, getitem_76, getitem_77, getitem_78, getitem_79, 0.0, [True, True, True, False]);  permute_158 = view_148 = view_149 = view_150 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = None
        getitem_126 = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_127 = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_128 = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
        clone_89 = torch.ops.aten.clone.default(getitem_128, memory_format = torch.contiguous_format);  getitem_128 = None
        view_251 = torch.ops.aten.view.default(clone_89, [24, 50, 64]);  clone_89 = None
        clone_90 = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        view_252 = torch.ops.aten.view.default(clone_90, [24, 50, 64]);  clone_90 = None
        clone_91 = torch.ops.aten.clone.default(getitem_126, memory_format = torch.contiguous_format);  getitem_126 = None
        view_253 = torch.ops.aten.view.default(clone_91, [24, 50, 64]);  clone_91 = None
        permute_159 = torch.ops.aten.permute.default(view_251, [1, 0, 2]);  view_251 = None
        clone_92 = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_254 = torch.ops.aten.view.default(clone_92, [50, 2, 768]);  clone_92 = None
        permute_160 = torch.ops.aten.permute.default(view_252, [1, 0, 2]);  view_252 = None
        clone_93 = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_255 = torch.ops.aten.view.default(clone_93, [50, 2, 768]);  clone_93 = None
        permute_161 = torch.ops.aten.permute.default(view_253, [1, 0, 2]);  view_253 = None
        clone_94 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_256 = torch.ops.aten.view.default(clone_94, [50, 2, 768]);  clone_94 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(full_default_21, view_254, 0, 2);  view_254 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(full_default_21, view_255, 0, 1);  view_255 = None
        add_117 = torch.ops.aten.add.Tensor(select_scatter_7, select_scatter_8);  select_scatter_7 = select_scatter_8 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(full_default_21, view_256, 0, 0);  view_256 = None
        add_118 = torch.ops.aten.add.Tensor(add_117, select_scatter_9);  add_117 = select_scatter_9 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(add_118, 3);  add_118 = None
        permute_162 = torch.ops.aten.permute.default(unsqueeze_26, [3, 1, 2, 0, 4]);  unsqueeze_26 = None
        squeeze_16 = torch.ops.aten.squeeze.dim(permute_162, 0);  permute_162 = None
        clone_95 = torch.ops.aten.clone.default(squeeze_16, memory_format = torch.contiguous_format);  squeeze_16 = None
        view_257 = torch.ops.aten.view.default(clone_95, [50, 2, 2304]);  clone_95 = None
        view_258 = torch.ops.aten.view.default(view_257, [100, 2304]);  view_257 = None
        mm_17 = torch.ops.aten.mm.default(view_258, permute_163);  view_258 = permute_163 = None
        view_259 = torch.ops.aten.view.default(mm_17, [50, 2, 768]);  mm_17 = None
        mul_188 = torch.ops.aten.mul.Tensor(view_259, primals_117);  view_259 = primals_117 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, 768)
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_188, [2], True)
        mul_190 = torch.ops.aten.mul.Tensor(mul_188, mul_61);  mul_188 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_61, sum_62);  mul_61 = sum_62 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_189, sum_61);  mul_189 = sum_61 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, mul_191);  sub_66 = mul_191 = None
        mul_192 = torch.ops.aten.mul.Tensor(div_45, sub_67);  div_45 = sub_67 = None
        add_119 = torch.ops.aten.add.Tensor(add_116, mul_192);  add_116 = mul_192 = None
        clone_97 = torch.ops.aten.clone.default(add_119, memory_format = torch.contiguous_format)
        view_260 = torch.ops.aten.view.default(clone_97, [100, 768]);  clone_97 = None
        mm_18 = torch.ops.aten.mm.default(view_260, permute_164);  view_260 = permute_164 = None
        view_261 = torch.ops.aten.view.default(mm_18, [50, 2, 3072]);  mm_18 = None
        view_139 = torch.ops.aten.view.default(addmm_34, [50, 2, 3072]);  addmm_34 = None
        mul_193 = torch.ops.aten.mul.Tensor(view_261, view_139)
        mul_59 = torch.ops.aten.mul.Tensor(view_139, 1.702);  view_139 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(mul_59);  mul_59 = None
        mul_194 = torch.ops.aten.mul.Tensor(view_261, sigmoid_8);  view_261 = None
        sub_68 = torch.ops.aten.sub.Tensor(1, sigmoid_8)
        mul_195 = torch.ops.aten.mul.Tensor(sigmoid_8, sub_68);  sigmoid_8 = sub_68 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_193, mul_195);  mul_193 = mul_195 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, 1.702);  mul_196 = None
        add_120 = torch.ops.aten.add.Tensor(mul_194, mul_197);  mul_194 = mul_197 = None
        view_262 = torch.ops.aten.view.default(add_120, [100, 3072]);  add_120 = None
        mm_19 = torch.ops.aten.mm.default(view_262, permute_165);  view_262 = permute_165 = None
        view_263 = torch.ops.aten.view.default(mm_19, [50, 2, 768]);  mm_19 = None
        mul_199 = torch.ops.aten.mul.Tensor(view_263, primals_111);  view_263 = primals_111 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, 768)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_199, [2], True)
        mul_201 = torch.ops.aten.mul.Tensor(mul_199, mul_57);  mul_199 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_57, sum_64);  mul_57 = sum_64 = None
        sub_70 = torch.ops.aten.sub.Tensor(mul_200, sum_63);  mul_200 = sum_63 = None
        sub_71 = torch.ops.aten.sub.Tensor(sub_70, mul_202);  sub_70 = mul_202 = None
        mul_203 = torch.ops.aten.mul.Tensor(div_46, sub_71);  div_46 = sub_71 = None
        add_121 = torch.ops.aten.add.Tensor(add_119, mul_203);  add_119 = mul_203 = None
        clone_99 = torch.ops.aten.clone.default(add_121, memory_format = torch.contiguous_format)
        view_264 = torch.ops.aten.view.default(clone_99, [100, 768]);  clone_99 = None
        mm_20 = torch.ops.aten.mm.default(view_264, permute_166);  view_264 = permute_166 = None
        view_265 = torch.ops.aten.view.default(mm_20, [50, 2, 12, 64]);  mm_20 = None
        permute_167 = torch.ops.aten.permute.default(view_265, [1, 2, 0, 3]);  view_265 = None
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, view_133, view_134, view_135, None, getitem_68, getitem_69, getitem_70, getitem_71, 0.0, [True, True, True, False]);  permute_167 = view_133 = view_134 = view_135 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        getitem_130 = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_131 = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_132 = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
        clone_100 = torch.ops.aten.clone.default(getitem_132, memory_format = torch.contiguous_format);  getitem_132 = None
        view_266 = torch.ops.aten.view.default(clone_100, [24, 50, 64]);  clone_100 = None
        clone_101 = torch.ops.aten.clone.default(getitem_131, memory_format = torch.contiguous_format);  getitem_131 = None
        view_267 = torch.ops.aten.view.default(clone_101, [24, 50, 64]);  clone_101 = None
        clone_102 = torch.ops.aten.clone.default(getitem_130, memory_format = torch.contiguous_format);  getitem_130 = None
        view_268 = torch.ops.aten.view.default(clone_102, [24, 50, 64]);  clone_102 = None
        permute_168 = torch.ops.aten.permute.default(view_266, [1, 0, 2]);  view_266 = None
        clone_103 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_269 = torch.ops.aten.view.default(clone_103, [50, 2, 768]);  clone_103 = None
        permute_169 = torch.ops.aten.permute.default(view_267, [1, 0, 2]);  view_267 = None
        clone_104 = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_270 = torch.ops.aten.view.default(clone_104, [50, 2, 768]);  clone_104 = None
        permute_170 = torch.ops.aten.permute.default(view_268, [1, 0, 2]);  view_268 = None
        clone_105 = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_271 = torch.ops.aten.view.default(clone_105, [50, 2, 768]);  clone_105 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(full_default_21, view_269, 0, 2);  view_269 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(full_default_21, view_270, 0, 1);  view_270 = None
        add_122 = torch.ops.aten.add.Tensor(select_scatter_10, select_scatter_11);  select_scatter_10 = select_scatter_11 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(full_default_21, view_271, 0, 0);  view_271 = None
        add_123 = torch.ops.aten.add.Tensor(add_122, select_scatter_12);  add_122 = select_scatter_12 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(add_123, 3);  add_123 = None
        permute_171 = torch.ops.aten.permute.default(unsqueeze_27, [3, 1, 2, 0, 4]);  unsqueeze_27 = None
        squeeze_17 = torch.ops.aten.squeeze.dim(permute_171, 0);  permute_171 = None
        clone_106 = torch.ops.aten.clone.default(squeeze_17, memory_format = torch.contiguous_format);  squeeze_17 = None
        view_272 = torch.ops.aten.view.default(clone_106, [50, 2, 2304]);  clone_106 = None
        view_273 = torch.ops.aten.view.default(view_272, [100, 2304]);  view_272 = None
        mm_21 = torch.ops.aten.mm.default(view_273, permute_172);  view_273 = permute_172 = None
        view_274 = torch.ops.aten.view.default(mm_21, [50, 2, 768]);  mm_21 = None
        mul_205 = torch.ops.aten.mul.Tensor(view_274, primals_105);  view_274 = primals_105 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, 768)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_205, [2], True)
        clone_34 = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format);  add_62 = None
        sub_23 = torch.ops.aten.sub.Tensor(clone_34, getitem_67);  clone_34 = getitem_67 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_205, mul_55);  mul_205 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_207, [2], True);  mul_207 = None
        mul_208 = torch.ops.aten.mul.Tensor(mul_55, sum_66);  mul_55 = sum_66 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_206, sum_65);  mul_206 = sum_65 = None
        sub_74 = torch.ops.aten.sub.Tensor(sub_73, mul_208);  sub_73 = mul_208 = None
        div_47 = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        mul_209 = torch.ops.aten.mul.Tensor(div_47, sub_74);  div_47 = sub_74 = None
        add_124 = torch.ops.aten.add.Tensor(add_121, mul_209);  add_121 = mul_209 = None
        permute_173 = torch.ops.aten.permute.default(slice_scatter_1, [1, 0, 2]);  slice_scatter_1 = None
        add_125 = torch.ops.aten.add.Tensor(add_124, permute_173);  add_124 = permute_173 = None
        clone_108 = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
        view_275 = torch.ops.aten.view.default(clone_108, [100, 768]);  clone_108 = None
        mm_22 = torch.ops.aten.mm.default(view_275, permute_174);  view_275 = permute_174 = None
        view_276 = torch.ops.aten.view.default(mm_22, [50, 2, 3072]);  mm_22 = None
        view_124 = torch.ops.aten.view.default(addmm_30, [50, 2, 3072]);  addmm_30 = None
        mul_210 = torch.ops.aten.mul.Tensor(view_276, view_124)
        mul_53 = torch.ops.aten.mul.Tensor(view_124, 1.702);  view_124 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(mul_53);  mul_53 = None
        mul_211 = torch.ops.aten.mul.Tensor(view_276, sigmoid_7);  view_276 = None
        sub_75 = torch.ops.aten.sub.Tensor(1, sigmoid_7)
        mul_212 = torch.ops.aten.mul.Tensor(sigmoid_7, sub_75);  sigmoid_7 = sub_75 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_210, mul_212);  mul_210 = mul_212 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 1.702);  mul_213 = None
        add_126 = torch.ops.aten.add.Tensor(mul_211, mul_214);  mul_211 = mul_214 = None
        view_277 = torch.ops.aten.view.default(add_126, [100, 3072]);  add_126 = None
        mm_23 = torch.ops.aten.mm.default(view_277, permute_175);  view_277 = permute_175 = None
        view_278 = torch.ops.aten.view.default(mm_23, [50, 2, 768]);  mm_23 = None
        mul_216 = torch.ops.aten.mul.Tensor(view_278, primals_99);  view_278 = primals_99 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, 768)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
        mul_218 = torch.ops.aten.mul.Tensor(mul_216, mul_51);  mul_216 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_51, sum_68);  mul_51 = sum_68 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_217, sum_67);  mul_217 = sum_67 = None
        sub_78 = torch.ops.aten.sub.Tensor(sub_77, mul_219);  sub_77 = mul_219 = None
        mul_220 = torch.ops.aten.mul.Tensor(div_48, sub_78);  div_48 = sub_78 = None
        add_127 = torch.ops.aten.add.Tensor(add_125, mul_220);  add_125 = mul_220 = None
        clone_110 = torch.ops.aten.clone.default(add_127, memory_format = torch.contiguous_format)
        view_279 = torch.ops.aten.view.default(clone_110, [100, 768]);  clone_110 = None
        mm_24 = torch.ops.aten.mm.default(view_279, permute_176);  view_279 = permute_176 = None
        view_280 = torch.ops.aten.view.default(mm_24, [50, 2, 12, 64]);  mm_24 = None
        permute_177 = torch.ops.aten.permute.default(view_280, [1, 2, 0, 3]);  view_280 = None
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_177, view_118, view_119, view_120, None, getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_177 = view_118 = view_119 = view_120 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_134 = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_135 = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_136 = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
        clone_111 = torch.ops.aten.clone.default(getitem_136, memory_format = torch.contiguous_format);  getitem_136 = None
        view_281 = torch.ops.aten.view.default(clone_111, [24, 50, 64]);  clone_111 = None
        clone_112 = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format);  getitem_135 = None
        view_282 = torch.ops.aten.view.default(clone_112, [24, 50, 64]);  clone_112 = None
        clone_113 = torch.ops.aten.clone.default(getitem_134, memory_format = torch.contiguous_format);  getitem_134 = None
        view_283 = torch.ops.aten.view.default(clone_113, [24, 50, 64]);  clone_113 = None
        permute_178 = torch.ops.aten.permute.default(view_281, [1, 0, 2]);  view_281 = None
        clone_114 = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_284 = torch.ops.aten.view.default(clone_114, [50, 2, 768]);  clone_114 = None
        permute_179 = torch.ops.aten.permute.default(view_282, [1, 0, 2]);  view_282 = None
        clone_115 = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_285 = torch.ops.aten.view.default(clone_115, [50, 2, 768]);  clone_115 = None
        permute_180 = torch.ops.aten.permute.default(view_283, [1, 0, 2]);  view_283 = None
        clone_116 = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_286 = torch.ops.aten.view.default(clone_116, [50, 2, 768]);  clone_116 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(full_default_21, view_284, 0, 2);  view_284 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(full_default_21, view_285, 0, 1);  view_285 = None
        add_128 = torch.ops.aten.add.Tensor(select_scatter_13, select_scatter_14);  select_scatter_13 = select_scatter_14 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(full_default_21, view_286, 0, 0);  view_286 = None
        add_129 = torch.ops.aten.add.Tensor(add_128, select_scatter_15);  add_128 = select_scatter_15 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(add_129, 3);  add_129 = None
        permute_181 = torch.ops.aten.permute.default(unsqueeze_28, [3, 1, 2, 0, 4]);  unsqueeze_28 = None
        squeeze_18 = torch.ops.aten.squeeze.dim(permute_181, 0);  permute_181 = None
        clone_117 = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_287 = torch.ops.aten.view.default(clone_117, [50, 2, 2304]);  clone_117 = None
        view_288 = torch.ops.aten.view.default(view_287, [100, 2304]);  view_287 = None
        mm_25 = torch.ops.aten.mm.default(view_288, permute_182);  view_288 = permute_182 = None
        view_289 = torch.ops.aten.view.default(mm_25, [50, 2, 768]);  mm_25 = None
        mul_222 = torch.ops.aten.mul.Tensor(view_289, primals_93);  view_289 = primals_93 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, 768)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224 = torch.ops.aten.mul.Tensor(mul_222, mul_49);  mul_222 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_49, sum_70);  mul_49 = sum_70 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_223, sum_69);  mul_223 = sum_69 = None
        sub_81 = torch.ops.aten.sub.Tensor(sub_80, mul_225);  sub_80 = mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(div_49, sub_81);  div_49 = sub_81 = None
        add_130 = torch.ops.aten.add.Tensor(add_127, mul_226);  add_127 = mul_226 = None
        clone_119 = torch.ops.aten.clone.default(add_130, memory_format = torch.contiguous_format)
        view_290 = torch.ops.aten.view.default(clone_119, [100, 768]);  clone_119 = None
        mm_26 = torch.ops.aten.mm.default(view_290, permute_183);  view_290 = permute_183 = None
        view_291 = torch.ops.aten.view.default(mm_26, [50, 2, 3072]);  mm_26 = None
        view_109 = torch.ops.aten.view.default(addmm_26, [50, 2, 3072]);  addmm_26 = None
        mul_227 = torch.ops.aten.mul.Tensor(view_291, view_109)
        mul_47 = torch.ops.aten.mul.Tensor(view_109, 1.702);  view_109 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(mul_47);  mul_47 = None
        mul_228 = torch.ops.aten.mul.Tensor(view_291, sigmoid_6);  view_291 = None
        sub_82 = torch.ops.aten.sub.Tensor(1, sigmoid_6)
        mul_229 = torch.ops.aten.mul.Tensor(sigmoid_6, sub_82);  sigmoid_6 = sub_82 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_227, mul_229);  mul_227 = mul_229 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_230, 1.702);  mul_230 = None
        add_131 = torch.ops.aten.add.Tensor(mul_228, mul_231);  mul_228 = mul_231 = None
        view_292 = torch.ops.aten.view.default(add_131, [100, 3072]);  add_131 = None
        mm_27 = torch.ops.aten.mm.default(view_292, permute_184);  view_292 = permute_184 = None
        view_293 = torch.ops.aten.view.default(mm_27, [50, 2, 768]);  mm_27 = None
        mul_233 = torch.ops.aten.mul.Tensor(view_293, primals_87);  view_293 = primals_87 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, 768)
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_233, [2], True)
        mul_235 = torch.ops.aten.mul.Tensor(mul_233, mul_45);  mul_233 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_235, [2], True);  mul_235 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_45, sum_72);  mul_45 = sum_72 = None
        sub_84 = torch.ops.aten.sub.Tensor(mul_234, sum_71);  mul_234 = sum_71 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, mul_236);  sub_84 = mul_236 = None
        mul_237 = torch.ops.aten.mul.Tensor(div_50, sub_85);  div_50 = sub_85 = None
        add_132 = torch.ops.aten.add.Tensor(add_130, mul_237);  add_130 = mul_237 = None
        clone_121 = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
        view_294 = torch.ops.aten.view.default(clone_121, [100, 768]);  clone_121 = None
        mm_28 = torch.ops.aten.mm.default(view_294, permute_185);  view_294 = permute_185 = None
        view_295 = torch.ops.aten.view.default(mm_28, [50, 2, 12, 64]);  mm_28 = None
        permute_186 = torch.ops.aten.permute.default(view_295, [1, 2, 0, 3]);  view_295 = None
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_186, view_103, view_104, view_105, None, getitem_52, getitem_53, getitem_54, getitem_55, 0.0, [True, True, True, False]);  permute_186 = view_103 = view_104 = view_105 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = None
        getitem_138 = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_139 = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_140 = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
        clone_122 = torch.ops.aten.clone.default(getitem_140, memory_format = torch.contiguous_format);  getitem_140 = None
        view_296 = torch.ops.aten.view.default(clone_122, [24, 50, 64]);  clone_122 = None
        clone_123 = torch.ops.aten.clone.default(getitem_139, memory_format = torch.contiguous_format);  getitem_139 = None
        view_297 = torch.ops.aten.view.default(clone_123, [24, 50, 64]);  clone_123 = None
        clone_124 = torch.ops.aten.clone.default(getitem_138, memory_format = torch.contiguous_format);  getitem_138 = None
        view_298 = torch.ops.aten.view.default(clone_124, [24, 50, 64]);  clone_124 = None
        permute_187 = torch.ops.aten.permute.default(view_296, [1, 0, 2]);  view_296 = None
        clone_125 = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_299 = torch.ops.aten.view.default(clone_125, [50, 2, 768]);  clone_125 = None
        permute_188 = torch.ops.aten.permute.default(view_297, [1, 0, 2]);  view_297 = None
        clone_126 = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_300 = torch.ops.aten.view.default(clone_126, [50, 2, 768]);  clone_126 = None
        permute_189 = torch.ops.aten.permute.default(view_298, [1, 0, 2]);  view_298 = None
        clone_127 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_301 = torch.ops.aten.view.default(clone_127, [50, 2, 768]);  clone_127 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(full_default_21, view_299, 0, 2);  view_299 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(full_default_21, view_300, 0, 1);  view_300 = None
        add_133 = torch.ops.aten.add.Tensor(select_scatter_16, select_scatter_17);  select_scatter_16 = select_scatter_17 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(full_default_21, view_301, 0, 0);  view_301 = None
        add_134 = torch.ops.aten.add.Tensor(add_133, select_scatter_18);  add_133 = select_scatter_18 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(add_134, 3);  add_134 = None
        permute_190 = torch.ops.aten.permute.default(unsqueeze_29, [3, 1, 2, 0, 4]);  unsqueeze_29 = None
        squeeze_19 = torch.ops.aten.squeeze.dim(permute_190, 0);  permute_190 = None
        clone_128 = torch.ops.aten.clone.default(squeeze_19, memory_format = torch.contiguous_format);  squeeze_19 = None
        view_302 = torch.ops.aten.view.default(clone_128, [50, 2, 2304]);  clone_128 = None
        view_303 = torch.ops.aten.view.default(view_302, [100, 2304]);  view_302 = None
        mm_29 = torch.ops.aten.mm.default(view_303, permute_191);  view_303 = permute_191 = None
        view_304 = torch.ops.aten.view.default(mm_29, [50, 2, 768]);  mm_29 = None
        mul_239 = torch.ops.aten.mul.Tensor(view_304, primals_81);  view_304 = primals_81 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, 768)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_239, [2], True)
        mul_241 = torch.ops.aten.mul.Tensor(mul_239, mul_43);  mul_239 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_241, [2], True);  mul_241 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_43, sum_74);  mul_43 = sum_74 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_240, sum_73);  mul_240 = sum_73 = None
        sub_88 = torch.ops.aten.sub.Tensor(sub_87, mul_242);  sub_87 = mul_242 = None
        mul_243 = torch.ops.aten.mul.Tensor(div_51, sub_88);  div_51 = sub_88 = None
        add_135 = torch.ops.aten.add.Tensor(add_132, mul_243);  add_132 = mul_243 = None
        clone_130 = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format)
        view_305 = torch.ops.aten.view.default(clone_130, [100, 768]);  clone_130 = None
        mm_30 = torch.ops.aten.mm.default(view_305, permute_192);  view_305 = permute_192 = None
        view_306 = torch.ops.aten.view.default(mm_30, [50, 2, 3072]);  mm_30 = None
        view_94 = torch.ops.aten.view.default(addmm_22, [50, 2, 3072]);  addmm_22 = None
        mul_244 = torch.ops.aten.mul.Tensor(view_306, view_94)
        mul_41 = torch.ops.aten.mul.Tensor(view_94, 1.702);  view_94 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(mul_41);  mul_41 = None
        mul_245 = torch.ops.aten.mul.Tensor(view_306, sigmoid_5);  view_306 = None
        sub_89 = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_246 = torch.ops.aten.mul.Tensor(sigmoid_5, sub_89);  sigmoid_5 = sub_89 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_244, mul_246);  mul_244 = mul_246 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, 1.702);  mul_247 = None
        add_136 = torch.ops.aten.add.Tensor(mul_245, mul_248);  mul_245 = mul_248 = None
        view_307 = torch.ops.aten.view.default(add_136, [100, 3072]);  add_136 = None
        mm_31 = torch.ops.aten.mm.default(view_307, permute_193);  view_307 = permute_193 = None
        view_308 = torch.ops.aten.view.default(mm_31, [50, 2, 768]);  mm_31 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_308, primals_75);  view_308 = primals_75 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, 768)
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
        mul_252 = torch.ops.aten.mul.Tensor(mul_250, mul_39);  mul_250 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_39, sum_76);  mul_39 = sum_76 = None
        sub_91 = torch.ops.aten.sub.Tensor(mul_251, sum_75);  mul_251 = sum_75 = None
        sub_92 = torch.ops.aten.sub.Tensor(sub_91, mul_253);  sub_91 = mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(div_52, sub_92);  div_52 = sub_92 = None
        add_137 = torch.ops.aten.add.Tensor(add_135, mul_254);  add_135 = mul_254 = None
        clone_132 = torch.ops.aten.clone.default(add_137, memory_format = torch.contiguous_format)
        view_309 = torch.ops.aten.view.default(clone_132, [100, 768]);  clone_132 = None
        mm_32 = torch.ops.aten.mm.default(view_309, permute_194);  view_309 = permute_194 = None
        view_310 = torch.ops.aten.view.default(mm_32, [50, 2, 12, 64]);  mm_32 = None
        permute_195 = torch.ops.aten.permute.default(view_310, [1, 2, 0, 3]);  view_310 = None
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_195, view_88, view_89, view_90, None, getitem_44, getitem_45, getitem_46, getitem_47, 0.0, [True, True, True, False]);  permute_195 = view_88 = view_89 = view_90 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        getitem_142 = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_143 = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_144 = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
        clone_133 = torch.ops.aten.clone.default(getitem_144, memory_format = torch.contiguous_format);  getitem_144 = None
        view_311 = torch.ops.aten.view.default(clone_133, [24, 50, 64]);  clone_133 = None
        clone_134 = torch.ops.aten.clone.default(getitem_143, memory_format = torch.contiguous_format);  getitem_143 = None
        view_312 = torch.ops.aten.view.default(clone_134, [24, 50, 64]);  clone_134 = None
        clone_135 = torch.ops.aten.clone.default(getitem_142, memory_format = torch.contiguous_format);  getitem_142 = None
        view_313 = torch.ops.aten.view.default(clone_135, [24, 50, 64]);  clone_135 = None
        permute_196 = torch.ops.aten.permute.default(view_311, [1, 0, 2]);  view_311 = None
        clone_136 = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_314 = torch.ops.aten.view.default(clone_136, [50, 2, 768]);  clone_136 = None
        permute_197 = torch.ops.aten.permute.default(view_312, [1, 0, 2]);  view_312 = None
        clone_137 = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_315 = torch.ops.aten.view.default(clone_137, [50, 2, 768]);  clone_137 = None
        permute_198 = torch.ops.aten.permute.default(view_313, [1, 0, 2]);  view_313 = None
        clone_138 = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_316 = torch.ops.aten.view.default(clone_138, [50, 2, 768]);  clone_138 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(full_default_21, view_314, 0, 2);  view_314 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(full_default_21, view_315, 0, 1);  view_315 = None
        add_138 = torch.ops.aten.add.Tensor(select_scatter_19, select_scatter_20);  select_scatter_19 = select_scatter_20 = None
        select_scatter_21 = torch.ops.aten.select_scatter.default(full_default_21, view_316, 0, 0);  view_316 = None
        add_139 = torch.ops.aten.add.Tensor(add_138, select_scatter_21);  add_138 = select_scatter_21 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(add_139, 3);  add_139 = None
        permute_199 = torch.ops.aten.permute.default(unsqueeze_30, [3, 1, 2, 0, 4]);  unsqueeze_30 = None
        squeeze_20 = torch.ops.aten.squeeze.dim(permute_199, 0);  permute_199 = None
        clone_139 = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
        view_317 = torch.ops.aten.view.default(clone_139, [50, 2, 2304]);  clone_139 = None
        view_318 = torch.ops.aten.view.default(view_317, [100, 2304]);  view_317 = None
        mm_33 = torch.ops.aten.mm.default(view_318, permute_200);  view_318 = permute_200 = None
        view_319 = torch.ops.aten.view.default(mm_33, [50, 2, 768]);  mm_33 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_319, primals_69);  view_319 = primals_69 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, 768)
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_256, [2], True)
        mul_258 = torch.ops.aten.mul.Tensor(mul_256, mul_37);  mul_256 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_258, [2], True);  mul_258 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_37, sum_78);  mul_37 = sum_78 = None
        sub_94 = torch.ops.aten.sub.Tensor(mul_257, sum_77);  mul_257 = sum_77 = None
        sub_95 = torch.ops.aten.sub.Tensor(sub_94, mul_259);  sub_94 = mul_259 = None
        mul_260 = torch.ops.aten.mul.Tensor(div_53, sub_95);  div_53 = sub_95 = None
        add_140 = torch.ops.aten.add.Tensor(add_137, mul_260);  add_137 = mul_260 = None
        clone_141 = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
        view_320 = torch.ops.aten.view.default(clone_141, [100, 768]);  clone_141 = None
        mm_34 = torch.ops.aten.mm.default(view_320, permute_201);  view_320 = permute_201 = None
        view_321 = torch.ops.aten.view.default(mm_34, [50, 2, 3072]);  mm_34 = None
        view_79 = torch.ops.aten.view.default(addmm_18, [50, 2, 3072]);  addmm_18 = None
        mul_261 = torch.ops.aten.mul.Tensor(view_321, view_79)
        mul_35 = torch.ops.aten.mul.Tensor(view_79, 1.702);  view_79 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(mul_35);  mul_35 = None
        mul_262 = torch.ops.aten.mul.Tensor(view_321, sigmoid_4);  view_321 = None
        sub_96 = torch.ops.aten.sub.Tensor(1, sigmoid_4)
        mul_263 = torch.ops.aten.mul.Tensor(sigmoid_4, sub_96);  sigmoid_4 = sub_96 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_261, mul_263);  mul_261 = mul_263 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, 1.702);  mul_264 = None
        add_141 = torch.ops.aten.add.Tensor(mul_262, mul_265);  mul_262 = mul_265 = None
        view_322 = torch.ops.aten.view.default(add_141, [100, 3072]);  add_141 = None
        mm_35 = torch.ops.aten.mm.default(view_322, permute_202);  view_322 = permute_202 = None
        view_323 = torch.ops.aten.view.default(mm_35, [50, 2, 768]);  mm_35 = None
        mul_267 = torch.ops.aten.mul.Tensor(view_323, primals_63);  view_323 = primals_63 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269 = torch.ops.aten.mul.Tensor(mul_267, mul_33);  mul_267 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_33, sum_80);  mul_33 = sum_80 = None
        sub_98 = torch.ops.aten.sub.Tensor(mul_268, sum_79);  mul_268 = sum_79 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, mul_270);  sub_98 = mul_270 = None
        mul_271 = torch.ops.aten.mul.Tensor(div_54, sub_99);  div_54 = sub_99 = None
        add_142 = torch.ops.aten.add.Tensor(add_140, mul_271);  add_140 = mul_271 = None
        clone_143 = torch.ops.aten.clone.default(add_142, memory_format = torch.contiguous_format)
        view_324 = torch.ops.aten.view.default(clone_143, [100, 768]);  clone_143 = None
        mm_36 = torch.ops.aten.mm.default(view_324, permute_203);  view_324 = permute_203 = None
        view_325 = torch.ops.aten.view.default(mm_36, [50, 2, 12, 64]);  mm_36 = None
        permute_204 = torch.ops.aten.permute.default(view_325, [1, 2, 0, 3]);  view_325 = None
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_204, view_73, view_74, view_75, None, getitem_36, getitem_37, getitem_38, getitem_39, 0.0, [True, True, True, False]);  permute_204 = view_73 = view_74 = view_75 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = None
        getitem_146 = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_147 = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_148 = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
        clone_144 = torch.ops.aten.clone.default(getitem_148, memory_format = torch.contiguous_format);  getitem_148 = None
        view_326 = torch.ops.aten.view.default(clone_144, [24, 50, 64]);  clone_144 = None
        clone_145 = torch.ops.aten.clone.default(getitem_147, memory_format = torch.contiguous_format);  getitem_147 = None
        view_327 = torch.ops.aten.view.default(clone_145, [24, 50, 64]);  clone_145 = None
        clone_146 = torch.ops.aten.clone.default(getitem_146, memory_format = torch.contiguous_format);  getitem_146 = None
        view_328 = torch.ops.aten.view.default(clone_146, [24, 50, 64]);  clone_146 = None
        permute_205 = torch.ops.aten.permute.default(view_326, [1, 0, 2]);  view_326 = None
        clone_147 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_329 = torch.ops.aten.view.default(clone_147, [50, 2, 768]);  clone_147 = None
        permute_206 = torch.ops.aten.permute.default(view_327, [1, 0, 2]);  view_327 = None
        clone_148 = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_330 = torch.ops.aten.view.default(clone_148, [50, 2, 768]);  clone_148 = None
        permute_207 = torch.ops.aten.permute.default(view_328, [1, 0, 2]);  view_328 = None
        clone_149 = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_331 = torch.ops.aten.view.default(clone_149, [50, 2, 768]);  clone_149 = None
        select_scatter_22 = torch.ops.aten.select_scatter.default(full_default_21, view_329, 0, 2);  view_329 = None
        select_scatter_23 = torch.ops.aten.select_scatter.default(full_default_21, view_330, 0, 1);  view_330 = None
        add_143 = torch.ops.aten.add.Tensor(select_scatter_22, select_scatter_23);  select_scatter_22 = select_scatter_23 = None
        select_scatter_24 = torch.ops.aten.select_scatter.default(full_default_21, view_331, 0, 0);  view_331 = None
        add_144 = torch.ops.aten.add.Tensor(add_143, select_scatter_24);  add_143 = select_scatter_24 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(add_144, 3);  add_144 = None
        permute_208 = torch.ops.aten.permute.default(unsqueeze_31, [3, 1, 2, 0, 4]);  unsqueeze_31 = None
        squeeze_21 = torch.ops.aten.squeeze.dim(permute_208, 0);  permute_208 = None
        clone_150 = torch.ops.aten.clone.default(squeeze_21, memory_format = torch.contiguous_format);  squeeze_21 = None
        view_332 = torch.ops.aten.view.default(clone_150, [50, 2, 2304]);  clone_150 = None
        view_333 = torch.ops.aten.view.default(view_332, [100, 2304]);  view_332 = None
        mm_37 = torch.ops.aten.mm.default(view_333, permute_209);  view_333 = permute_209 = None
        view_334 = torch.ops.aten.view.default(mm_37, [50, 2, 768]);  mm_37 = None
        mul_273 = torch.ops.aten.mul.Tensor(view_334, primals_57);  view_334 = primals_57 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_273, 768)
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_273, [2], True)
        clone_18 = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(clone_18, getitem_35);  clone_18 = getitem_35 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_273, mul_31);  mul_273 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_275, [2], True);  mul_275 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_31, sum_82);  mul_31 = sum_82 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_274, sum_81);  mul_274 = sum_81 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, mul_276);  sub_101 = mul_276 = None
        div_55 = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        mul_277 = torch.ops.aten.mul.Tensor(div_55, sub_102);  div_55 = sub_102 = None
        add_145 = torch.ops.aten.add.Tensor(add_142, mul_277);  add_142 = mul_277 = None
        permute_210 = torch.ops.aten.permute.default(slice_scatter_4, [1, 0, 2]);  slice_scatter_4 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, permute_210);  add_145 = permute_210 = None
        clone_152 = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
        view_335 = torch.ops.aten.view.default(clone_152, [100, 768]);  clone_152 = None
        mm_38 = torch.ops.aten.mm.default(view_335, permute_211);  view_335 = permute_211 = None
        view_336 = torch.ops.aten.view.default(mm_38, [50, 2, 3072]);  mm_38 = None
        view_64 = torch.ops.aten.view.default(addmm_14, [50, 2, 3072]);  addmm_14 = None
        mul_278 = torch.ops.aten.mul.Tensor(view_336, view_64)
        mul_29 = torch.ops.aten.mul.Tensor(view_64, 1.702);  view_64 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(mul_29);  mul_29 = None
        mul_279 = torch.ops.aten.mul.Tensor(view_336, sigmoid_3);  view_336 = None
        sub_103 = torch.ops.aten.sub.Tensor(1, sigmoid_3)
        mul_280 = torch.ops.aten.mul.Tensor(sigmoid_3, sub_103);  sigmoid_3 = sub_103 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_278, mul_280);  mul_278 = mul_280 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, 1.702);  mul_281 = None
        add_147 = torch.ops.aten.add.Tensor(mul_279, mul_282);  mul_279 = mul_282 = None
        view_337 = torch.ops.aten.view.default(add_147, [100, 3072]);  add_147 = None
        mm_39 = torch.ops.aten.mm.default(view_337, permute_212);  view_337 = permute_212 = None
        view_338 = torch.ops.aten.view.default(mm_39, [50, 2, 768]);  mm_39 = None
        mul_284 = torch.ops.aten.mul.Tensor(view_338, primals_51);  view_338 = primals_51 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, 768)
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_284, [2], True)
        mul_286 = torch.ops.aten.mul.Tensor(mul_284, mul_27);  mul_284 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_27, sum_84);  mul_27 = sum_84 = None
        sub_105 = torch.ops.aten.sub.Tensor(mul_285, sum_83);  mul_285 = sum_83 = None
        sub_106 = torch.ops.aten.sub.Tensor(sub_105, mul_287);  sub_105 = mul_287 = None
        mul_288 = torch.ops.aten.mul.Tensor(div_56, sub_106);  div_56 = sub_106 = None
        add_148 = torch.ops.aten.add.Tensor(add_146, mul_288);  add_146 = mul_288 = None
        clone_154 = torch.ops.aten.clone.default(add_148, memory_format = torch.contiguous_format)
        view_339 = torch.ops.aten.view.default(clone_154, [100, 768]);  clone_154 = None
        mm_40 = torch.ops.aten.mm.default(view_339, permute_213);  view_339 = permute_213 = None
        view_340 = torch.ops.aten.view.default(mm_40, [50, 2, 12, 64]);  mm_40 = None
        permute_214 = torch.ops.aten.permute.default(view_340, [1, 2, 0, 3]);  view_340 = None
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_214, view_58, view_59, view_60, None, getitem_28, getitem_29, getitem_30, getitem_31, 0.0, [True, True, True, False]);  permute_214 = view_58 = view_59 = view_60 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = None
        getitem_150 = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_151 = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_152 = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
        clone_155 = torch.ops.aten.clone.default(getitem_152, memory_format = torch.contiguous_format);  getitem_152 = None
        view_341 = torch.ops.aten.view.default(clone_155, [24, 50, 64]);  clone_155 = None
        clone_156 = torch.ops.aten.clone.default(getitem_151, memory_format = torch.contiguous_format);  getitem_151 = None
        view_342 = torch.ops.aten.view.default(clone_156, [24, 50, 64]);  clone_156 = None
        clone_157 = torch.ops.aten.clone.default(getitem_150, memory_format = torch.contiguous_format);  getitem_150 = None
        view_343 = torch.ops.aten.view.default(clone_157, [24, 50, 64]);  clone_157 = None
        permute_215 = torch.ops.aten.permute.default(view_341, [1, 0, 2]);  view_341 = None
        clone_158 = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_344 = torch.ops.aten.view.default(clone_158, [50, 2, 768]);  clone_158 = None
        permute_216 = torch.ops.aten.permute.default(view_342, [1, 0, 2]);  view_342 = None
        clone_159 = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_345 = torch.ops.aten.view.default(clone_159, [50, 2, 768]);  clone_159 = None
        permute_217 = torch.ops.aten.permute.default(view_343, [1, 0, 2]);  view_343 = None
        clone_160 = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_346 = torch.ops.aten.view.default(clone_160, [50, 2, 768]);  clone_160 = None
        select_scatter_25 = torch.ops.aten.select_scatter.default(full_default_21, view_344, 0, 2);  view_344 = None
        select_scatter_26 = torch.ops.aten.select_scatter.default(full_default_21, view_345, 0, 1);  view_345 = None
        add_149 = torch.ops.aten.add.Tensor(select_scatter_25, select_scatter_26);  select_scatter_25 = select_scatter_26 = None
        select_scatter_27 = torch.ops.aten.select_scatter.default(full_default_21, view_346, 0, 0);  view_346 = None
        add_150 = torch.ops.aten.add.Tensor(add_149, select_scatter_27);  add_149 = select_scatter_27 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(add_150, 3);  add_150 = None
        permute_218 = torch.ops.aten.permute.default(unsqueeze_32, [3, 1, 2, 0, 4]);  unsqueeze_32 = None
        squeeze_22 = torch.ops.aten.squeeze.dim(permute_218, 0);  permute_218 = None
        clone_161 = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_347 = torch.ops.aten.view.default(clone_161, [50, 2, 2304]);  clone_161 = None
        view_348 = torch.ops.aten.view.default(view_347, [100, 2304]);  view_347 = None
        mm_41 = torch.ops.aten.mm.default(view_348, permute_219);  view_348 = permute_219 = None
        view_349 = torch.ops.aten.view.default(mm_41, [50, 2, 768]);  mm_41 = None
        mul_290 = torch.ops.aten.mul.Tensor(view_349, primals_45);  view_349 = primals_45 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 768)
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292 = torch.ops.aten.mul.Tensor(mul_290, mul_25);  mul_290 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_25, sum_86);  mul_25 = sum_86 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_291, sum_85);  mul_291 = sum_85 = None
        sub_109 = torch.ops.aten.sub.Tensor(sub_108, mul_293);  sub_108 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(div_57, sub_109);  div_57 = sub_109 = None
        add_151 = torch.ops.aten.add.Tensor(add_148, mul_294);  add_148 = mul_294 = None
        clone_163 = torch.ops.aten.clone.default(add_151, memory_format = torch.contiguous_format)
        view_350 = torch.ops.aten.view.default(clone_163, [100, 768]);  clone_163 = None
        mm_42 = torch.ops.aten.mm.default(view_350, permute_220);  view_350 = permute_220 = None
        view_351 = torch.ops.aten.view.default(mm_42, [50, 2, 3072]);  mm_42 = None
        view_49 = torch.ops.aten.view.default(addmm_10, [50, 2, 3072]);  addmm_10 = None
        mul_295 = torch.ops.aten.mul.Tensor(view_351, view_49)
        mul_23 = torch.ops.aten.mul.Tensor(view_49, 1.702);  view_49 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(mul_23);  mul_23 = None
        mul_296 = torch.ops.aten.mul.Tensor(view_351, sigmoid_2);  view_351 = None
        sub_110 = torch.ops.aten.sub.Tensor(1, sigmoid_2)
        mul_297 = torch.ops.aten.mul.Tensor(sigmoid_2, sub_110);  sigmoid_2 = sub_110 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_295, mul_297);  mul_295 = mul_297 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, 1.702);  mul_298 = None
        add_152 = torch.ops.aten.add.Tensor(mul_296, mul_299);  mul_296 = mul_299 = None
        view_352 = torch.ops.aten.view.default(add_152, [100, 3072]);  add_152 = None
        mm_43 = torch.ops.aten.mm.default(view_352, permute_221);  view_352 = permute_221 = None
        view_353 = torch.ops.aten.view.default(mm_43, [50, 2, 768]);  mm_43 = None
        mul_301 = torch.ops.aten.mul.Tensor(view_353, primals_39);  view_353 = primals_39 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, 768)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_301, [2], True)
        mul_303 = torch.ops.aten.mul.Tensor(mul_301, mul_21);  mul_301 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_303, [2], True);  mul_303 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_21, sum_88);  mul_21 = sum_88 = None
        sub_112 = torch.ops.aten.sub.Tensor(mul_302, sum_87);  mul_302 = sum_87 = None
        sub_113 = torch.ops.aten.sub.Tensor(sub_112, mul_304);  sub_112 = mul_304 = None
        mul_305 = torch.ops.aten.mul.Tensor(div_58, sub_113);  div_58 = sub_113 = None
        add_153 = torch.ops.aten.add.Tensor(add_151, mul_305);  add_151 = mul_305 = None
        clone_165 = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
        view_354 = torch.ops.aten.view.default(clone_165, [100, 768]);  clone_165 = None
        mm_44 = torch.ops.aten.mm.default(view_354, permute_222);  view_354 = permute_222 = None
        view_355 = torch.ops.aten.view.default(mm_44, [50, 2, 12, 64]);  mm_44 = None
        permute_223 = torch.ops.aten.permute.default(view_355, [1, 2, 0, 3]);  view_355 = None
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_223, view_43, view_44, view_45, None, getitem_20, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False]);  permute_223 = view_43 = view_44 = view_45 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = None
        getitem_154 = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_155 = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_156 = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
        clone_166 = torch.ops.aten.clone.default(getitem_156, memory_format = torch.contiguous_format);  getitem_156 = None
        view_356 = torch.ops.aten.view.default(clone_166, [24, 50, 64]);  clone_166 = None
        clone_167 = torch.ops.aten.clone.default(getitem_155, memory_format = torch.contiguous_format);  getitem_155 = None
        view_357 = torch.ops.aten.view.default(clone_167, [24, 50, 64]);  clone_167 = None
        clone_168 = torch.ops.aten.clone.default(getitem_154, memory_format = torch.contiguous_format);  getitem_154 = None
        view_358 = torch.ops.aten.view.default(clone_168, [24, 50, 64]);  clone_168 = None
        permute_224 = torch.ops.aten.permute.default(view_356, [1, 0, 2]);  view_356 = None
        clone_169 = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_359 = torch.ops.aten.view.default(clone_169, [50, 2, 768]);  clone_169 = None
        permute_225 = torch.ops.aten.permute.default(view_357, [1, 0, 2]);  view_357 = None
        clone_170 = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_360 = torch.ops.aten.view.default(clone_170, [50, 2, 768]);  clone_170 = None
        permute_226 = torch.ops.aten.permute.default(view_358, [1, 0, 2]);  view_358 = None
        clone_171 = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_361 = torch.ops.aten.view.default(clone_171, [50, 2, 768]);  clone_171 = None
        select_scatter_28 = torch.ops.aten.select_scatter.default(full_default_21, view_359, 0, 2);  view_359 = None
        select_scatter_29 = torch.ops.aten.select_scatter.default(full_default_21, view_360, 0, 1);  view_360 = None
        add_154 = torch.ops.aten.add.Tensor(select_scatter_28, select_scatter_29);  select_scatter_28 = select_scatter_29 = None
        select_scatter_30 = torch.ops.aten.select_scatter.default(full_default_21, view_361, 0, 0);  view_361 = None
        add_155 = torch.ops.aten.add.Tensor(add_154, select_scatter_30);  add_154 = select_scatter_30 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(add_155, 3);  add_155 = None
        permute_227 = torch.ops.aten.permute.default(unsqueeze_33, [3, 1, 2, 0, 4]);  unsqueeze_33 = None
        squeeze_23 = torch.ops.aten.squeeze.dim(permute_227, 0);  permute_227 = None
        clone_172 = torch.ops.aten.clone.default(squeeze_23, memory_format = torch.contiguous_format);  squeeze_23 = None
        view_362 = torch.ops.aten.view.default(clone_172, [50, 2, 2304]);  clone_172 = None
        view_363 = torch.ops.aten.view.default(view_362, [100, 2304]);  view_362 = None
        mm_45 = torch.ops.aten.mm.default(view_363, permute_228);  view_363 = permute_228 = None
        view_364 = torch.ops.aten.view.default(mm_45, [50, 2, 768]);  mm_45 = None
        mul_307 = torch.ops.aten.mul.Tensor(view_364, primals_33);  view_364 = primals_33 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, 768)
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
        mul_309 = torch.ops.aten.mul.Tensor(mul_307, mul_19);  mul_307 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_19, sum_90);  mul_19 = sum_90 = None
        sub_115 = torch.ops.aten.sub.Tensor(mul_308, sum_89);  mul_308 = sum_89 = None
        sub_116 = torch.ops.aten.sub.Tensor(sub_115, mul_310);  sub_115 = mul_310 = None
        mul_311 = torch.ops.aten.mul.Tensor(div_59, sub_116);  div_59 = sub_116 = None
        add_156 = torch.ops.aten.add.Tensor(add_153, mul_311);  add_153 = mul_311 = None
        clone_174 = torch.ops.aten.clone.default(add_156, memory_format = torch.contiguous_format)
        view_365 = torch.ops.aten.view.default(clone_174, [100, 768]);  clone_174 = None
        mm_46 = torch.ops.aten.mm.default(view_365, permute_229);  view_365 = permute_229 = None
        view_366 = torch.ops.aten.view.default(mm_46, [50, 2, 3072]);  mm_46 = None
        view_34 = torch.ops.aten.view.default(addmm_6, [50, 2, 3072]);  addmm_6 = None
        mul_312 = torch.ops.aten.mul.Tensor(view_366, view_34)
        mul_17 = torch.ops.aten.mul.Tensor(view_34, 1.702);  view_34 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(mul_17);  mul_17 = None
        mul_313 = torch.ops.aten.mul.Tensor(view_366, sigmoid_1);  view_366 = None
        sub_117 = torch.ops.aten.sub.Tensor(1, sigmoid_1)
        mul_314 = torch.ops.aten.mul.Tensor(sigmoid_1, sub_117);  sigmoid_1 = sub_117 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_312, mul_314);  mul_312 = mul_314 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, 1.702);  mul_315 = None
        add_157 = torch.ops.aten.add.Tensor(mul_313, mul_316);  mul_313 = mul_316 = None
        view_367 = torch.ops.aten.view.default(add_157, [100, 3072]);  add_157 = None
        mm_47 = torch.ops.aten.mm.default(view_367, permute_230);  view_367 = permute_230 = None
        view_368 = torch.ops.aten.view.default(mm_47, [50, 2, 768]);  mm_47 = None
        mul_318 = torch.ops.aten.mul.Tensor(view_368, primals_27);  view_368 = primals_27 = None
        mul_319 = torch.ops.aten.mul.Tensor(mul_318, 768)
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320 = torch.ops.aten.mul.Tensor(mul_318, mul_15);  mul_318 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_15, sum_92);  mul_15 = sum_92 = None
        sub_119 = torch.ops.aten.sub.Tensor(mul_319, sum_91);  mul_319 = sum_91 = None
        sub_120 = torch.ops.aten.sub.Tensor(sub_119, mul_321);  sub_119 = mul_321 = None
        mul_322 = torch.ops.aten.mul.Tensor(div_60, sub_120);  div_60 = sub_120 = None
        add_158 = torch.ops.aten.add.Tensor(add_156, mul_322);  add_156 = mul_322 = None
        clone_176 = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
        view_369 = torch.ops.aten.view.default(clone_176, [100, 768]);  clone_176 = None
        mm_48 = torch.ops.aten.mm.default(view_369, permute_231);  view_369 = permute_231 = None
        view_370 = torch.ops.aten.view.default(mm_48, [50, 2, 12, 64]);  mm_48 = None
        permute_232 = torch.ops.aten.permute.default(view_370, [1, 2, 0, 3]);  view_370 = None
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_232, view_28, view_29, view_30, None, getitem_12, getitem_13, getitem_14, getitem_15, 0.0, [True, True, True, False]);  permute_232 = view_28 = view_29 = view_30 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = None
        getitem_158 = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_159 = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_160 = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
        clone_177 = torch.ops.aten.clone.default(getitem_160, memory_format = torch.contiguous_format);  getitem_160 = None
        view_371 = torch.ops.aten.view.default(clone_177, [24, 50, 64]);  clone_177 = None
        clone_178 = torch.ops.aten.clone.default(getitem_159, memory_format = torch.contiguous_format);  getitem_159 = None
        view_372 = torch.ops.aten.view.default(clone_178, [24, 50, 64]);  clone_178 = None
        clone_179 = torch.ops.aten.clone.default(getitem_158, memory_format = torch.contiguous_format);  getitem_158 = None
        view_373 = torch.ops.aten.view.default(clone_179, [24, 50, 64]);  clone_179 = None
        permute_233 = torch.ops.aten.permute.default(view_371, [1, 0, 2]);  view_371 = None
        clone_180 = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_374 = torch.ops.aten.view.default(clone_180, [50, 2, 768]);  clone_180 = None
        permute_234 = torch.ops.aten.permute.default(view_372, [1, 0, 2]);  view_372 = None
        clone_181 = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_375 = torch.ops.aten.view.default(clone_181, [50, 2, 768]);  clone_181 = None
        permute_235 = torch.ops.aten.permute.default(view_373, [1, 0, 2]);  view_373 = None
        clone_182 = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_376 = torch.ops.aten.view.default(clone_182, [50, 2, 768]);  clone_182 = None
        select_scatter_31 = torch.ops.aten.select_scatter.default(full_default_21, view_374, 0, 2);  view_374 = None
        select_scatter_32 = torch.ops.aten.select_scatter.default(full_default_21, view_375, 0, 1);  view_375 = None
        add_159 = torch.ops.aten.add.Tensor(select_scatter_31, select_scatter_32);  select_scatter_31 = select_scatter_32 = None
        select_scatter_33 = torch.ops.aten.select_scatter.default(full_default_21, view_376, 0, 0);  view_376 = None
        add_160 = torch.ops.aten.add.Tensor(add_159, select_scatter_33);  add_159 = select_scatter_33 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(add_160, 3);  add_160 = None
        permute_236 = torch.ops.aten.permute.default(unsqueeze_34, [3, 1, 2, 0, 4]);  unsqueeze_34 = None
        squeeze_24 = torch.ops.aten.squeeze.dim(permute_236, 0);  permute_236 = None
        clone_183 = torch.ops.aten.clone.default(squeeze_24, memory_format = torch.contiguous_format);  squeeze_24 = None
        view_377 = torch.ops.aten.view.default(clone_183, [50, 2, 2304]);  clone_183 = None
        view_378 = torch.ops.aten.view.default(view_377, [100, 2304]);  view_377 = None
        mm_49 = torch.ops.aten.mm.default(view_378, permute_237);  view_378 = permute_237 = None
        view_379 = torch.ops.aten.view.default(mm_49, [50, 2, 768]);  mm_49 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_379, primals_21);  view_379 = primals_21 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_324, 768)
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
        mul_326 = torch.ops.aten.mul.Tensor(mul_324, mul_13);  mul_324 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
        mul_327 = torch.ops.aten.mul.Tensor(mul_13, sum_94);  mul_13 = sum_94 = None
        sub_122 = torch.ops.aten.sub.Tensor(mul_325, sum_93);  mul_325 = sum_93 = None
        sub_123 = torch.ops.aten.sub.Tensor(sub_122, mul_327);  sub_122 = mul_327 = None
        mul_328 = torch.ops.aten.mul.Tensor(div_61, sub_123);  div_61 = sub_123 = None
        add_161 = torch.ops.aten.add.Tensor(add_158, mul_328);  add_158 = mul_328 = None
        clone_185 = torch.ops.aten.clone.default(add_161, memory_format = torch.contiguous_format)
        view_380 = torch.ops.aten.view.default(clone_185, [100, 768]);  clone_185 = None
        mm_50 = torch.ops.aten.mm.default(view_380, permute_238);  view_380 = permute_238 = None
        view_381 = torch.ops.aten.view.default(mm_50, [50, 2, 3072]);  mm_50 = None
        view_19 = torch.ops.aten.view.default(addmm_2, [50, 2, 3072]);  addmm_2 = None
        mul_329 = torch.ops.aten.mul.Tensor(view_381, view_19)
        mul_11 = torch.ops.aten.mul.Tensor(view_19, 1.702);  view_19 = None
        sigmoid = torch.ops.aten.sigmoid.default(mul_11);  mul_11 = None
        mul_330 = torch.ops.aten.mul.Tensor(view_381, sigmoid);  view_381 = None
        sub_124 = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_331 = torch.ops.aten.mul.Tensor(sigmoid, sub_124);  sigmoid = sub_124 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_329, mul_331);  mul_329 = mul_331 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_332, 1.702);  mul_332 = None
        add_162 = torch.ops.aten.add.Tensor(mul_330, mul_333);  mul_330 = mul_333 = None
        view_382 = torch.ops.aten.view.default(add_162, [100, 3072]);  add_162 = None
        mm_51 = torch.ops.aten.mm.default(view_382, permute_239);  view_382 = permute_239 = None
        view_383 = torch.ops.aten.view.default(mm_51, [50, 2, 768]);  mm_51 = None
        mul_335 = torch.ops.aten.mul.Tensor(view_383, primals_15);  view_383 = primals_15 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, 768)
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
        mul_337 = torch.ops.aten.mul.Tensor(mul_335, mul_9);  mul_335 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_9, sum_96);  mul_9 = sum_96 = None
        sub_126 = torch.ops.aten.sub.Tensor(mul_336, sum_95);  mul_336 = sum_95 = None
        sub_127 = torch.ops.aten.sub.Tensor(sub_126, mul_338);  sub_126 = mul_338 = None
        mul_339 = torch.ops.aten.mul.Tensor(div_62, sub_127);  div_62 = sub_127 = None
        add_163 = torch.ops.aten.add.Tensor(add_161, mul_339);  add_161 = mul_339 = None
        clone_187 = torch.ops.aten.clone.default(add_163, memory_format = torch.contiguous_format)
        view_384 = torch.ops.aten.view.default(clone_187, [100, 768]);  clone_187 = None
        mm_52 = torch.ops.aten.mm.default(view_384, permute_240);  view_384 = permute_240 = None
        view_385 = torch.ops.aten.view.default(mm_52, [50, 2, 12, 64]);  mm_52 = None
        permute_241 = torch.ops.aten.permute.default(view_385, [1, 2, 0, 3]);  view_385 = None
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_241, view_13, view_14, view_15, None, getitem_4, getitem_5, getitem_6, getitem_7, 0.0, [True, True, True, False]);  permute_241 = view_13 = view_14 = view_15 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = None
        getitem_162 = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_163 = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_164 = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
        clone_188 = torch.ops.aten.clone.default(getitem_164, memory_format = torch.contiguous_format);  getitem_164 = None
        view_386 = torch.ops.aten.view.default(clone_188, [24, 50, 64]);  clone_188 = None
        clone_189 = torch.ops.aten.clone.default(getitem_163, memory_format = torch.contiguous_format);  getitem_163 = None
        view_387 = torch.ops.aten.view.default(clone_189, [24, 50, 64]);  clone_189 = None
        clone_190 = torch.ops.aten.clone.default(getitem_162, memory_format = torch.contiguous_format);  getitem_162 = None
        view_388 = torch.ops.aten.view.default(clone_190, [24, 50, 64]);  clone_190 = None
        permute_242 = torch.ops.aten.permute.default(view_386, [1, 0, 2]);  view_386 = None
        clone_191 = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_389 = torch.ops.aten.view.default(clone_191, [50, 2, 768]);  clone_191 = None
        permute_243 = torch.ops.aten.permute.default(view_387, [1, 0, 2]);  view_387 = None
        clone_192 = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_390 = torch.ops.aten.view.default(clone_192, [50, 2, 768]);  clone_192 = None
        permute_244 = torch.ops.aten.permute.default(view_388, [1, 0, 2]);  view_388 = None
        clone_193 = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_391 = torch.ops.aten.view.default(clone_193, [50, 2, 768]);  clone_193 = None
        select_scatter_34 = torch.ops.aten.select_scatter.default(full_default_21, view_389, 0, 2);  view_389 = None
        select_scatter_35 = torch.ops.aten.select_scatter.default(full_default_21, view_390, 0, 1);  view_390 = None
        add_164 = torch.ops.aten.add.Tensor(select_scatter_34, select_scatter_35);  select_scatter_34 = select_scatter_35 = None
        select_scatter_36 = torch.ops.aten.select_scatter.default(full_default_21, view_391, 0, 0);  full_default_21 = view_391 = None
        add_165 = torch.ops.aten.add.Tensor(add_164, select_scatter_36);  add_164 = select_scatter_36 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(add_165, 3);  add_165 = None
        permute_245 = torch.ops.aten.permute.default(unsqueeze_35, [3, 1, 2, 0, 4]);  unsqueeze_35 = None
        squeeze_25 = torch.ops.aten.squeeze.dim(permute_245, 0);  permute_245 = None
        clone_194 = torch.ops.aten.clone.default(squeeze_25, memory_format = torch.contiguous_format);  squeeze_25 = None
        view_392 = torch.ops.aten.view.default(clone_194, [50, 2, 2304]);  clone_194 = None
        view_393 = torch.ops.aten.view.default(view_392, [100, 2304]);  view_392 = None
        mm_53 = torch.ops.aten.mm.default(view_393, permute_246);  view_393 = permute_246 = None
        view_394 = torch.ops.aten.view.default(mm_53, [50, 2, 768]);  mm_53 = None
        mul_341 = torch.ops.aten.mul.Tensor(view_394, primals_9);  view_394 = primals_9 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        add_12 = torch.ops.aten.add.Tensor(cat, primals_6);  cat = primals_6 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_12, getitem_1);  add_12 = getitem_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, primals_7)
        add_14 = torch.ops.aten.add.Tensor(mul_6, primals_8);  mul_6 = primals_8 = None
        permute_3 = torch.ops.aten.permute.default(add_14, [1, 0, 2]);  add_14 = None
        clone_2 = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(clone_2, getitem_3);  clone_2 = getitem_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_341, mul_7);  mul_341 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_7, sum_98);  mul_7 = sum_98 = None
        sub_129 = torch.ops.aten.sub.Tensor(mul_342, sum_97);  mul_342 = sum_97 = None
        sub_130 = torch.ops.aten.sub.Tensor(sub_129, mul_344);  sub_129 = mul_344 = None
        div_63 = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_345 = torch.ops.aten.mul.Tensor(div_63, sub_130);  div_63 = sub_130 = None
        add_166 = torch.ops.aten.add.Tensor(add_163, mul_345);  add_163 = mul_345 = None
        permute_247 = torch.ops.aten.permute.default(add_166, [1, 0, 2]);  add_166 = None
        mul_347 = torch.ops.aten.mul.Tensor(permute_247, primals_7);  permute_247 = primals_7 = None
        mul_348 = torch.ops.aten.mul.Tensor(mul_347, 768)
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
        mul_349 = torch.ops.aten.mul.Tensor(mul_347, mul_5);  mul_347 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_5, sum_100);  mul_5 = sum_100 = None
        sub_132 = torch.ops.aten.sub.Tensor(mul_348, sum_99);  mul_348 = sum_99 = None
        sub_133 = torch.ops.aten.sub.Tensor(sub_132, mul_350);  sub_132 = mul_350 = None
        div_64 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_351 = torch.ops.aten.mul.Tensor(div_64, sub_133);  div_64 = sub_133 = None
        slice_12 = torch.ops.aten.slice.Tensor(mul_351, 1, 1, 50);  mul_351 = None
        permute_248 = torch.ops.aten.permute.default(slice_12, [0, 2, 1]);  slice_12 = None
        view_395 = torch.ops.aten.view.default(permute_248, [2, 768, 7, 7]);  permute_248 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_395, div, primals_4, [0], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  view_395 = div = primals_4 = None
        getitem_166 = convolution_backward_6[0];  convolution_backward_6 = None
        div_65 = torch.ops.aten.div.Tensor(getitem_166, device_put_1);  getitem_166 = device_put_1 = None
        _adaptive_avg_pool2d_backward = torch.ops.aten._adaptive_avg_pool2d_backward.default(div_65, add_10);  div_65 = add_10 = None
        mul_352 = torch.ops.aten.mul.Tensor(_adaptive_avg_pool2d_backward, 0.5);  _adaptive_avg_pool2d_backward = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(index_put, 1);  index_put = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_1);  mul_352 = unsqueeze_1 = None
        permute_249 = torch.ops.aten.permute.default(mul_353, [0, 2, 3, 1]);  mul_353 = None
        full_default_57 = torch.ops.aten.full.default([2, 258, 258, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand = torch.ops.aten.expand.default(view, [2, 256, 256]);  view = None
        index_put_1 = torch.ops.aten.index_put.default(full_default_57, [expand, clamp_max, clamp_max_1], permute_249, True);  full_default_57 = expand = clamp_max = clamp_max_1 = permute_249 = None
        permute_250 = torch.ops.aten.permute.default(index_put_1, [0, 3, 1, 2]);  index_put_1 = None
        constant_pad_nd_5 = torch.ops.aten.constant_pad_nd.default(permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]);  permute_250 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(constant_pad_nd_5, [1, 2, 3], True)
        add_2 = torch.ops.aten.add.Tensor(inductor_random_default, 0.5);  inductor_random_default = None
        mul_354 = torch.ops.aten.mul.Tensor(constant_pad_nd_5, add_2);  constant_pad_nd_5 = add_2 = None
        neg_9 = torch.ops.aten.neg.default(mul_354)
        sum_102 = torch.ops.aten.sum.dim_IntList(neg_9, [1, 2, 3], True);  neg_9 = None
        add_167 = torch.ops.aten.add.Tensor(sum_101, sum_102);  sum_101 = sum_102 = None
        expand_35 = torch.ops.aten.expand.default(add_167, [2, 3, 256, 256]);  add_167 = None
        div_66 = torch.ops.aten.div.Scalar(expand_35, 196608);  expand_35 = None
        add_168 = torch.ops.aten.add.Tensor(mul_354, div_66);  mul_354 = div_66 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(add_168, [1], True)
        mul = torch.ops.aten.mul.Tensor(inductor_random_default_1, 2);  inductor_random_default_1 = None
        mul_355 = torch.ops.aten.mul.Tensor(add_168, mul);  add_168 = mul = None
        neg_10 = torch.ops.aten.neg.default(mul_355)
        sum_104 = torch.ops.aten.sum.dim_IntList(neg_10, [1], True);  neg_10 = None
        add_169 = torch.ops.aten.add.Tensor(sum_103, sum_104);  sum_103 = sum_104 = None
        expand_36 = torch.ops.aten.expand.default(add_169, [2, 3, 256, 256]);  add_169 = None
        div_67 = torch.ops.aten.div.Scalar(expand_36, 3);  expand_36 = None
        add_170 = torch.ops.aten.add.Tensor(mul_355, div_67);  mul_355 = div_67 = None
        copy__2 = torch.ops.aten.copy_.default(primals_162, div_5);  primals_162 = div_5 = copy__2 = None
        copy__6 = torch.ops.aten.copy_.default(primals_171, div_11);  primals_171 = div_11 = copy__6 = None
        copy__10 = torch.ops.aten.copy_.default(primals_179, div_17);  primals_179 = div_17 = copy__10 = None
        return (add_170, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, add_104, None, None, sum_47, None, add_102, None, None, sum_45, add_100, None, None, sum_43, None, add_98, None, None, sum_41, add_96, None, None, view_202, add_94, None, None, view_199)
        
def load_args(reader):
    buf0 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf0, (768, 3, 32, 32), is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50, 768), is_leaf=True)  # primals_6
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # primals_8
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_9
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_15
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_21
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_27
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_33
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_39
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # primals_45
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_51
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # primals_57
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_63
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_69
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_75
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # primals_81
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_87
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # primals_93
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_99
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # primals_105
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_111
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_117
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_123
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # primals_129
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # primals_135
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # primals_141
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_147
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # primals_153
    buf29 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256, 768, 3, 3), is_leaf=True)  # primals_156
    buf30 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256, 1, 4, 4), is_leaf=True)  # primals_160
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1, 256, 1, 1), is_leaf=True)  # primals_161
    buf32 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1,), is_leaf=True)  # primals_162
    buf33 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256, 768, 3, 3), is_leaf=True)  # primals_165
    buf34 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256, 1, 4, 4), is_leaf=True)  # primals_169
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1, 256, 1, 1), is_leaf=True)  # primals_170
    buf36 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1,), is_leaf=True)  # primals_171
    buf37 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256, 512), is_leaf=True)  # primals_174
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1, 256), is_leaf=True)  # primals_178
    buf39 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1,), is_leaf=True)  # primals_179
    buf40 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2, 1, 1, 1), is_leaf=True)  # inductor_random_default_1
    buf41 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf41, (2, 1, 1, 1), is_leaf=True)  # inductor_random_default
    buf42 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf42, (2, 1, 1), dtype=torch.int64, is_leaf=True)  # view
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf43, (2, 256, 256), dtype=torch.int64, is_leaf=True)  # clamp_max
    buf44 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf44, (2, 256, 256), dtype=torch.int64, is_leaf=True)  # clamp_max_1
    buf45 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2, 256, 256), is_leaf=True)  # index_put
    buf46 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2, 3, 256, 256), is_leaf=True)  # add_10
    buf47 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3, 1, 1), is_leaf=True)  # device_put_1
    buf48 = reader.storage(None, 1204224, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2, 3, 224, 224), is_leaf=True)  # div
    buf49 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf49, (2, 50, 768), is_leaf=True)  # cat
    buf50 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf50, (2, 50, 1), is_leaf=True)  # getitem_1
    buf51 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf51, (2, 50, 1), is_leaf=True)  # rsqrt
    buf52 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf52, (50, 2, 1), is_leaf=True)  # getitem_3
    buf53 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf53, (50, 2, 1), is_leaf=True)  # rsqrt_1
    buf54 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_13
    reader.tensor(buf54, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_14
    reader.tensor(buf54, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_15
    buf55 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf55, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_4
    buf56 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf56, (2, 12, 64), is_leaf=True)  # getitem_5
    buf57 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf57, (), dtype=torch.int64, is_leaf=True)  # getitem_6
    buf58 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf58, (), dtype=torch.int64, is_leaf=True)  # getitem_7
    buf59 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf59, (50, 2, 768), is_leaf=True)  # mul_9
    buf60 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf60, (100, 3072), is_leaf=True)  # addmm_2
    buf61 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf61, (50, 2, 768), is_leaf=True)  # mul_13
    buf62 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf62, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_28
    reader.tensor(buf62, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_29
    reader.tensor(buf62, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_30
    buf63 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf63, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_12
    buf64 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2, 12, 64), is_leaf=True)  # getitem_13
    buf65 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf65, (), dtype=torch.int64, is_leaf=True)  # getitem_14
    buf66 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf66, (), dtype=torch.int64, is_leaf=True)  # getitem_15
    buf67 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf67, (50, 2, 768), is_leaf=True)  # mul_15
    buf68 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf68, (100, 3072), is_leaf=True)  # addmm_6
    buf69 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf69, (50, 2, 768), is_leaf=True)  # mul_19
    buf70 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf70, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_43
    reader.tensor(buf70, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_44
    reader.tensor(buf70, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_45
    buf71 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf71, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_20
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (2, 12, 64), is_leaf=True)  # getitem_21
    buf73 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf73, (), dtype=torch.int64, is_leaf=True)  # getitem_22
    buf74 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf74, (), dtype=torch.int64, is_leaf=True)  # getitem_23
    buf75 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf75, (50, 2, 768), is_leaf=True)  # mul_21
    buf76 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf76, (100, 3072), is_leaf=True)  # addmm_10
    buf77 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf77, (50, 2, 768), is_leaf=True)  # mul_25
    buf78 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_58
    reader.tensor(buf78, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_59
    reader.tensor(buf78, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_60
    buf79 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf79, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_28
    buf80 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf80, (2, 12, 64), is_leaf=True)  # getitem_29
    buf81 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf81, (), dtype=torch.int64, is_leaf=True)  # getitem_30
    buf82 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf82, (), dtype=torch.int64, is_leaf=True)  # getitem_31
    buf83 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf83, (50, 2, 768), is_leaf=True)  # mul_27
    buf84 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf84, (100, 3072), is_leaf=True)  # addmm_14
    buf85 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf85, (50, 2, 768), (768, 38400, 1), is_leaf=True)  # add_38
    buf86 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf86, (50, 2, 1), is_leaf=True)  # getitem_35
    buf87 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf87, (50, 2, 1), is_leaf=True)  # rsqrt_9
    buf88 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf88, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_73
    reader.tensor(buf88, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_74
    reader.tensor(buf88, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_75
    buf89 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_36
    buf90 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2, 12, 64), is_leaf=True)  # getitem_37
    buf91 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf91, (), dtype=torch.int64, is_leaf=True)  # getitem_38
    buf92 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf92, (), dtype=torch.int64, is_leaf=True)  # getitem_39
    buf93 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf93, (50, 2, 768), is_leaf=True)  # mul_33
    buf94 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf94, (100, 3072), is_leaf=True)  # addmm_18
    buf95 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf95, (50, 2, 768), is_leaf=True)  # mul_37
    buf96 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf96, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_88
    reader.tensor(buf96, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_89
    reader.tensor(buf96, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_90
    buf97 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf97, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_44
    buf98 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf98, (2, 12, 64), is_leaf=True)  # getitem_45
    buf99 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf99, (), dtype=torch.int64, is_leaf=True)  # getitem_46
    buf100 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf100, (), dtype=torch.int64, is_leaf=True)  # getitem_47
    buf101 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf101, (50, 2, 768), is_leaf=True)  # mul_39
    buf102 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf102, (100, 3072), is_leaf=True)  # addmm_22
    buf103 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf103, (50, 2, 768), is_leaf=True)  # mul_43
    buf104 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf104, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_103
    reader.tensor(buf104, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_104
    reader.tensor(buf104, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_105
    buf105 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf105, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_52
    buf106 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf106, (2, 12, 64), is_leaf=True)  # getitem_53
    buf107 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf107, (), dtype=torch.int64, is_leaf=True)  # getitem_54
    buf108 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf108, (), dtype=torch.int64, is_leaf=True)  # getitem_55
    buf109 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf109, (50, 2, 768), is_leaf=True)  # mul_45
    buf110 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf110, (100, 3072), is_leaf=True)  # addmm_26
    buf111 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf111, (50, 2, 768), is_leaf=True)  # mul_49
    buf112 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf112, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_118
    reader.tensor(buf112, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_119
    reader.tensor(buf112, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_120
    buf113 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_60
    buf114 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2, 12, 64), is_leaf=True)  # getitem_61
    buf115 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf115, (), dtype=torch.int64, is_leaf=True)  # getitem_62
    buf116 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf116, (), dtype=torch.int64, is_leaf=True)  # getitem_63
    buf117 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf117, (50, 2, 768), is_leaf=True)  # mul_51
    buf118 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf118, (100, 3072), is_leaf=True)  # addmm_30
    buf119 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf119, (50, 2, 768), (768, 38400, 1), is_leaf=True)  # add_62
    buf120 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf120, (50, 2, 1), is_leaf=True)  # getitem_67
    buf121 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf121, (50, 2, 1), is_leaf=True)  # rsqrt_17
    buf122 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf122, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_133
    reader.tensor(buf122, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_134
    reader.tensor(buf122, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_135
    buf123 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf123, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_68
    buf124 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf124, (2, 12, 64), is_leaf=True)  # getitem_69
    buf125 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf125, (), dtype=torch.int64, is_leaf=True)  # getitem_70
    buf126 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf126, (), dtype=torch.int64, is_leaf=True)  # getitem_71
    buf127 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf127, (50, 2, 768), is_leaf=True)  # mul_57
    buf128 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf128, (100, 3072), is_leaf=True)  # addmm_34
    buf129 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf129, (50, 2, 768), is_leaf=True)  # mul_61
    buf130 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf130, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_148
    reader.tensor(buf130, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_149
    reader.tensor(buf130, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_150
    buf131 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf131, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_76
    buf132 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (2, 12, 64), is_leaf=True)  # getitem_77
    buf133 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf133, (), dtype=torch.int64, is_leaf=True)  # getitem_78
    buf134 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf134, (), dtype=torch.int64, is_leaf=True)  # getitem_79
    buf135 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf135, (50, 2, 768), is_leaf=True)  # mul_63
    buf136 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf136, (100, 3072), is_leaf=True)  # addmm_38
    buf137 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf137, (50, 2, 768), is_leaf=True)  # mul_67
    buf138 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf138, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_163
    reader.tensor(buf138, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_164
    reader.tensor(buf138, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_165
    buf139 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf139, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_84
    buf140 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (2, 12, 64), is_leaf=True)  # getitem_85
    buf141 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf141, (), dtype=torch.int64, is_leaf=True)  # getitem_86
    buf142 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf142, (), dtype=torch.int64, is_leaf=True)  # getitem_87
    buf143 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf143, (50, 2, 768), is_leaf=True)  # mul_69
    buf144 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf144, (100, 3072), is_leaf=True)  # addmm_42
    buf145 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf145, (50, 2, 768), is_leaf=True)  # mul_73
    buf146 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf146, (2, 12, 50, 64), (768, 64, 1536, 1), is_leaf=True)  # view_178
    reader.tensor(buf146, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=76800, is_leaf=True)  # view_179
    reader.tensor(buf146, (2, 12, 50, 64), (768, 64, 1536, 1), storage_offset=153600, is_leaf=True)  # view_180
    buf147 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf147, (2, 12, 50, 64), (38400, 64, 768, 1), is_leaf=True)  # getitem_92
    buf148 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf148, (2, 12, 64), is_leaf=True)  # getitem_93
    buf149 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf149, (), dtype=torch.int64, is_leaf=True)  # getitem_94
    buf150 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf150, (), dtype=torch.int64, is_leaf=True)  # getitem_95
    buf151 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf151, (50, 2, 768), is_leaf=True)  # mul_75
    buf152 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf152, (100, 3072), is_leaf=True)  # addmm_46
    buf153 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (2, 768), is_leaf=True)  # mul_79
    buf154 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf154, (2, 512), is_leaf=True)  # mm
    buf155 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf155, (6912,), is_leaf=True)  # div_1
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # div_2
    buf157 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf157, (), is_leaf=True)  # sum_6
    buf158 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256, 768, 3, 3), is_leaf=True)  # div_3
    buf159 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf159, (2, 256, 9, 9), (20736, 1, 2304, 256), is_leaf=True)  # constant_pad_nd_1
    buf160 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf160, (2, 256, 6, 6), (9216, 1, 1536, 256), is_leaf=True)  # convolution_2
    buf161 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1,), is_leaf=True)  # clamp_min_6
    buf162 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf162, (1,), is_leaf=True)  # sum_9
    buf163 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf163, (1, 256, 1, 1), is_leaf=True)  # div_6
    buf164 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf164, (2, 1, 3, 3), (9, 1, 3, 1), is_leaf=True)  # convolution_3
    buf165 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf165, (6912,), is_leaf=True)  # div_7
    buf166 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256,), is_leaf=True)  # div_8
    buf167 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf167, (), is_leaf=True)  # sum_18
    buf168 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf168, (256, 768, 3, 3), is_leaf=True)  # div_9
    buf169 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf169, (2, 256, 9, 9), (20736, 1, 2304, 256), is_leaf=True)  # constant_pad_nd_2
    buf170 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf170, (2, 256, 6, 6), (9216, 1, 1536, 256), is_leaf=True)  # convolution_5
    buf171 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1,), is_leaf=True)  # clamp_min_10
    buf172 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1,), is_leaf=True)  # sum_21
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1, 256, 1, 1), is_leaf=True)  # div_12
    buf174 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf174, (2, 1, 3, 3), (9, 1, 3, 1), is_leaf=True)  # convolution_6
    buf175 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf175, (512,), is_leaf=True)  # div_13
    buf176 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256,), is_leaf=True)  # div_14
    buf177 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf177, (), is_leaf=True)  # sum_30
    buf178 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf178, (2, 256), is_leaf=True)  # where_2
    buf179 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf179, (1,), is_leaf=True)  # clamp_min_14
    buf180 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf180, (1,), is_leaf=True)  # sum_33
    buf181 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf181, (2, 1), is_leaf=True)  # addmm_49
    buf182 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1, 256), is_leaf=True)  # permute_125
    buf183 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf183, (256, 512), is_leaf=True)  # permute_129
    buf184 = reader.storage(None, 25088, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf184, (2, 256, 7, 7), (12544, 1, 1792, 256), dtype=torch.bool, is_leaf=True)  # gt_4
    buf185 = reader.storage(None, 25088, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf185, (2, 256, 7, 7), (12544, 1, 1792, 256), dtype=torch.bool, is_leaf=True)  # gt_5
    buf186 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf186, (512, 768), (1, 512), is_leaf=True)  # permute_135
    buf187 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf187, (2, 1), is_leaf=True)  # div_39
    buf188 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768, 3072), is_leaf=True)  # permute_137
    buf189 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf189, (3072, 768), is_leaf=True)  # permute_138
    buf190 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf190, (50, 2, 1), is_leaf=True)  # div_40
    buf191 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 768), is_leaf=True)  # permute_139
    buf192 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf192, (2304, 768), is_leaf=True)  # permute_145
    buf193 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf193, (50, 2, 1), is_leaf=True)  # div_41
    buf194 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768, 3072), is_leaf=True)  # permute_146
    buf195 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf195, (3072, 768), is_leaf=True)  # permute_147
    buf196 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf196, (50, 2, 1), is_leaf=True)  # div_42
    buf197 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768, 768), is_leaf=True)  # permute_148
    buf198 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf198, (2304, 768), is_leaf=True)  # permute_154
    buf199 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf199, (50, 2, 1), is_leaf=True)  # div_43
    buf200 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768, 3072), is_leaf=True)  # permute_155
    buf201 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf201, (3072, 768), is_leaf=True)  # permute_156
    buf202 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf202, (50, 2, 1), is_leaf=True)  # div_44
    buf203 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768, 768), is_leaf=True)  # permute_157
    buf204 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf204, (2304, 768), is_leaf=True)  # permute_163
    buf205 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf205, (50, 2, 1), is_leaf=True)  # div_45
    buf206 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf206, (768, 3072), is_leaf=True)  # permute_164
    buf207 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf207, (3072, 768), is_leaf=True)  # permute_165
    buf208 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf208, (50, 2, 1), is_leaf=True)  # div_46
    buf209 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf209, (768, 768), is_leaf=True)  # permute_166
    buf210 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf210, (2304, 768), is_leaf=True)  # permute_172
    buf211 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf211, (768, 3072), is_leaf=True)  # permute_174
    buf212 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf212, (3072, 768), is_leaf=True)  # permute_175
    buf213 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf213, (50, 2, 1), is_leaf=True)  # div_48
    buf214 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf214, (768, 768), is_leaf=True)  # permute_176
    buf215 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf215, (2304, 768), is_leaf=True)  # permute_182
    buf216 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf216, (50, 2, 1), is_leaf=True)  # div_49
    buf217 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf217, (768, 3072), is_leaf=True)  # permute_183
    buf218 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf218, (3072, 768), is_leaf=True)  # permute_184
    buf219 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf219, (50, 2, 1), is_leaf=True)  # div_50
    buf220 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf220, (768, 768), is_leaf=True)  # permute_185
    buf221 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf221, (2304, 768), is_leaf=True)  # permute_191
    buf222 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf222, (50, 2, 1), is_leaf=True)  # div_51
    buf223 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf223, (768, 3072), is_leaf=True)  # permute_192
    buf224 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf224, (3072, 768), is_leaf=True)  # permute_193
    buf225 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf225, (50, 2, 1), is_leaf=True)  # div_52
    buf226 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf226, (768, 768), is_leaf=True)  # permute_194
    buf227 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf227, (2304, 768), is_leaf=True)  # permute_200
    buf228 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf228, (50, 2, 1), is_leaf=True)  # div_53
    buf229 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf229, (768, 3072), is_leaf=True)  # permute_201
    buf230 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf230, (3072, 768), is_leaf=True)  # permute_202
    buf231 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf231, (50, 2, 1), is_leaf=True)  # div_54
    buf232 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf232, (768, 768), is_leaf=True)  # permute_203
    buf233 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf233, (2304, 768), is_leaf=True)  # permute_209
    buf234 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf234, (768, 3072), is_leaf=True)  # permute_211
    buf235 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf235, (3072, 768), is_leaf=True)  # permute_212
    buf236 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf236, (50, 2, 1), is_leaf=True)  # div_56
    buf237 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf237, (768, 768), is_leaf=True)  # permute_213
    buf238 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf238, (2304, 768), is_leaf=True)  # permute_219
    buf239 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf239, (50, 2, 1), is_leaf=True)  # div_57
    buf240 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf240, (768, 3072), is_leaf=True)  # permute_220
    buf241 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf241, (3072, 768), is_leaf=True)  # permute_221
    buf242 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf242, (50, 2, 1), is_leaf=True)  # div_58
    buf243 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf243, (768, 768), is_leaf=True)  # permute_222
    buf244 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf244, (2304, 768), is_leaf=True)  # permute_228
    buf245 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf245, (50, 2, 1), is_leaf=True)  # div_59
    buf246 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf246, (768, 3072), is_leaf=True)  # permute_229
    buf247 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf247, (3072, 768), is_leaf=True)  # permute_230
    buf248 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf248, (50, 2, 1), is_leaf=True)  # div_60
    buf249 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf249, (768, 768), is_leaf=True)  # permute_231
    buf250 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf250, (2304, 768), is_leaf=True)  # permute_237
    buf251 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf251, (50, 2, 1), is_leaf=True)  # div_61
    buf252 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf252, (768, 3072), is_leaf=True)  # permute_238
    buf253 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf253, (3072, 768), is_leaf=True)  # permute_239
    buf254 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf254, (50, 2, 1), is_leaf=True)  # div_62
    buf255 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf255, (768, 768), is_leaf=True)  # permute_240
    buf256 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf256, (2304, 768), is_leaf=True)  # permute_246
    buf257 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf257, (2, 1), is_leaf=True)  # tangents_1
    buf258 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf258, (256, 768, 3, 3), is_leaf=True)  # tangents_2
    buf259 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1, 256, 1, 1), is_leaf=True)  # tangents_3
    buf260 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf260, (256, 768, 3, 3), is_leaf=True)  # tangents_4
    buf261 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1, 256, 1, 1), is_leaf=True)  # tangents_5
    buf262 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf262, (256, 512), is_leaf=True)  # tangents_6
    buf263 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1, 256), is_leaf=True)  # tangents_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)