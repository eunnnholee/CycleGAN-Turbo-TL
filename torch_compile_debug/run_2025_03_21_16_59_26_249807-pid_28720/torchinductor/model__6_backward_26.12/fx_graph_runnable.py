
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

    
    
    def forward(self, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, mm, view_187, view_188, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, gt_4, gt_5, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7):
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
        permute_129 = torch.ops.aten.permute.default(where_3, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_129, mm);  permute_129 = mm = None
        sum_39 = torch.ops.aten.sum.dim_IntList(where_3, [0], True);  where_3 = None
        view_202 = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
        add_95 = torch.ops.aten.add.Tensor(tangents_6, mm_3);  tangents_6 = mm_3 = None
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
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, view_188, div_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_4 = view_188 = div_9 = None
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
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, view_187, div_3, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_5 = view_187 = div_3 = None
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
        copy__2 = torch.ops.aten.copy_.default(primals_162, div_5);  primals_162 = div_5 = copy__2 = None
        copy__6 = torch.ops.aten.copy_.default(primals_171, div_11);  primals_171 = div_11 = copy__6 = None
        copy__10 = torch.ops.aten.copy_.default(primals_179, div_17);  primals_179 = div_17 = copy__10 = None
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, add_104, None, None, sum_47, None, add_102, None, None, sum_45, add_100, None, None, sum_43, None, add_98, None, None, sum_41, add_96, None, None, view_202, add_94, None, None, view_199)
        
def load_args(reader):
    buf0 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf0, (256, 768, 3, 3), is_leaf=True)  # primals_156
    buf1 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (256, 1, 4, 4), is_leaf=True)  # primals_160
    buf2 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 256, 1, 1), is_leaf=True)  # primals_161
    buf3 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1,), is_leaf=True)  # primals_162
    buf4 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256, 768, 3, 3), is_leaf=True)  # primals_165
    buf5 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256, 1, 4, 4), is_leaf=True)  # primals_169
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1, 256, 1, 1), is_leaf=True)  # primals_170
    buf7 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1,), is_leaf=True)  # primals_171
    buf8 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf8, (256, 512), is_leaf=True)  # primals_174
    buf9 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1, 256), is_leaf=True)  # primals_178
    buf10 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1,), is_leaf=True)  # primals_179
    buf11 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2, 512), is_leaf=True)  # mm
    buf12 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf12, (2, 768, 7, 7), (37632, 1, 5376, 768), storage_offset=768, is_leaf=True)  # view_187
    buf13 = reader.storage(None, 307200, device=device(type='cuda', index=0))
    reader.tensor(buf13, (2, 768, 7, 7), (37632, 1, 5376, 768), storage_offset=768, is_leaf=True)  # view_188
    buf14 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf14, (6912,), is_leaf=True)  # div_1
    buf15 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256,), is_leaf=True)  # div_2
    buf16 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf16, (), is_leaf=True)  # sum_6
    buf17 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256, 768, 3, 3), is_leaf=True)  # div_3
    buf18 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2, 256, 9, 9), (20736, 1, 2304, 256), is_leaf=True)  # constant_pad_nd_1
    buf19 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2, 256, 6, 6), (9216, 1, 1536, 256), is_leaf=True)  # convolution_2
    buf20 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1,), is_leaf=True)  # clamp_min_6
    buf21 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1,), is_leaf=True)  # sum_9
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1, 256, 1, 1), is_leaf=True)  # div_6
    buf23 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf23, (2, 1, 3, 3), (9, 1, 3, 1), is_leaf=True)  # convolution_3
    buf24 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf24, (6912,), is_leaf=True)  # div_7
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # div_8
    buf26 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf26, (), is_leaf=True)  # sum_18
    buf27 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256, 768, 3, 3), is_leaf=True)  # div_9
    buf28 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf28, (2, 256, 9, 9), (20736, 1, 2304, 256), is_leaf=True)  # constant_pad_nd_2
    buf29 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2, 256, 6, 6), (9216, 1, 1536, 256), is_leaf=True)  # convolution_5
    buf30 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1,), is_leaf=True)  # clamp_min_10
    buf31 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1,), is_leaf=True)  # sum_21
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1, 256, 1, 1), is_leaf=True)  # div_12
    buf33 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf33, (2, 1, 3, 3), (9, 1, 3, 1), is_leaf=True)  # convolution_6
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # div_13
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # div_14
    buf36 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf36, (), is_leaf=True)  # sum_30
    buf37 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf37, (2, 256), is_leaf=True)  # where_2
    buf38 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1,), is_leaf=True)  # clamp_min_14
    buf39 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1,), is_leaf=True)  # sum_33
    buf40 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2, 1), is_leaf=True)  # addmm_49
    buf41 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf41, (1, 256), is_leaf=True)  # permute_125
    buf42 = reader.storage(None, 25088, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf42, (2, 256, 7, 7), (12544, 1, 1792, 256), dtype=torch.bool, is_leaf=True)  # gt_4
    buf43 = reader.storage(None, 25088, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf43, (2, 256, 7, 7), (12544, 1, 1792, 256), dtype=torch.bool, is_leaf=True)  # gt_5
    buf44 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2, 1), is_leaf=True)  # tangents_1
    buf45 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256, 768, 3, 3), is_leaf=True)  # tangents_2
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1, 256, 1, 1), is_leaf=True)  # tangents_3
    buf47 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf47, (256, 768, 3, 3), is_leaf=True)  # tangents_4
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1, 256, 1, 1), is_leaf=True)  # tangents_5
    buf49 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256, 512), is_leaf=True)  # tangents_6
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1, 256), is_leaf=True)  # tangents_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)