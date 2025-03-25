
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

    
    
    def forward(self, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, div, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, convolution_12, convolution_25, sqrt, pow_3, sqrt_2, pow_6, sqrt_4, pow_9, sqrt_6, pow_12, sqrt_8, add_9, pow_15, mul_6, div_22, mul_12, div_26, mul_18, div_30, mul_24, div_34, tangents_1):
        expand = torch.ops.aten.expand.default(tangents_1, [2, 1, 16, 16])
        div_12 = torch.ops.aten.div.Scalar(expand, 256);  expand = None
        convolution_backward = torch.ops.aten.convolution_backward.default(div_12, pow_15, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_12 = pow_15 = primals_35 = None
        getitem_16 = convolution_backward[0];  convolution_backward = None
        expand_1 = torch.ops.aten.expand.default(tangents_1, [2, 1, 32, 32])
        div_13 = torch.ops.aten.div.Scalar(expand_1, 1024);  expand_1 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(div_13, pow_12, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_13 = pow_12 = primals_34 = None
        getitem_19 = convolution_backward_1[0];  convolution_backward_1 = None
        expand_2 = torch.ops.aten.expand.default(tangents_1, [2, 1, 64, 64])
        div_14 = torch.ops.aten.div.Scalar(expand_2, 4096);  expand_2 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(div_14, pow_9, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_14 = pow_9 = primals_33 = None
        getitem_22 = convolution_backward_2[0];  convolution_backward_2 = None
        expand_3 = torch.ops.aten.expand.default(tangents_1, [2, 1, 128, 128])
        div_15 = torch.ops.aten.div.Scalar(expand_3, 16384);  expand_3 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(div_15, pow_6, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_15 = pow_6 = primals_32 = None
        getitem_25 = convolution_backward_3[0];  convolution_backward_3 = None
        expand_4 = torch.ops.aten.expand.default(tangents_1, [2, 1, 256, 256]);  tangents_1 = None
        div_16 = torch.ops.aten.div.Scalar(expand_4, 65536);  expand_4 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(div_16, pow_3, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_16 = pow_3 = primals_31 = None
        getitem_28 = convolution_backward_4[0];  convolution_backward_4 = None
        relu_12 = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        relu_25 = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        add_8 = torch.ops.aten.add.Tensor(sqrt_8, 1e-10)
        div_10 = torch.ops.aten.div.Tensor(relu_12, add_8)
        div_11 = torch.ops.aten.div.Tensor(relu_25, add_9);  relu_25 = add_9 = None
        sub_6 = torch.ops.aten.sub.Tensor(div_10, div_11);  div_11 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(sub_6, 1.0);  sub_6 = None
        mul = torch.ops.aten.mul.Scalar(pow_16, 2.0);  pow_16 = None
        mul_1 = torch.ops.aten.mul.Tensor(getitem_16, mul);  getitem_16 = mul = None
        div_18 = torch.ops.aten.div.Tensor(div_10, add_8);  div_10 = None
        neg = torch.ops.aten.neg.default(mul_1)
        mul_2 = torch.ops.aten.mul.Tensor(neg, div_18);  neg = div_18 = None
        div_19 = torch.ops.aten.div.Tensor(mul_1, add_8);  mul_1 = add_8 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_2, [1], True);  mul_2 = None
        mul_3 = torch.ops.aten.mul.Scalar(sqrt_8, 2);  sqrt_8 = None
        div_20 = torch.ops.aten.div.Tensor(sum_11, mul_3);  sum_11 = mul_3 = None
        expand_5 = torch.ops.aten.expand.default(div_20, [2, 512, 16, 16]);  div_20 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(relu_12, 1.0)
        mul_4 = torch.ops.aten.mul.Scalar(pow_17, 2.0);  pow_17 = None
        mul_5 = torch.ops.aten.mul.Tensor(expand_5, mul_4);  expand_5 = mul_4 = None
        add_15 = torch.ops.aten.add.Tensor(div_19, mul_5);  div_19 = mul_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(getitem_19, mul_6);  getitem_19 = mul_6 = None
        neg_1 = torch.ops.aten.neg.default(mul_7)
        mul_8 = torch.ops.aten.mul.Tensor(neg_1, div_22);  neg_1 = div_22 = None
        add_6 = torch.ops.aten.add.Tensor(sqrt_6, 1e-10)
        div_23 = torch.ops.aten.div.Tensor(mul_7, add_6);  mul_7 = add_6 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_8, [1], True);  mul_8 = None
        mul_9 = torch.ops.aten.mul.Scalar(sqrt_6, 2);  sqrt_6 = None
        div_24 = torch.ops.aten.div.Tensor(sum_12, mul_9);  sum_12 = mul_9 = None
        expand_6 = torch.ops.aten.expand.default(div_24, [2, 512, 32, 32]);  div_24 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(relu_9, 1.0)
        mul_10 = torch.ops.aten.mul.Scalar(pow_19, 2.0);  pow_19 = None
        mul_11 = torch.ops.aten.mul.Tensor(expand_6, mul_10);  expand_6 = mul_10 = None
        add_16 = torch.ops.aten.add.Tensor(div_23, mul_11);  div_23 = mul_11 = None
        mul_13 = torch.ops.aten.mul.Tensor(getitem_22, mul_12);  getitem_22 = mul_12 = None
        neg_2 = torch.ops.aten.neg.default(mul_13)
        mul_14 = torch.ops.aten.mul.Tensor(neg_2, div_26);  neg_2 = div_26 = None
        add_4 = torch.ops.aten.add.Tensor(sqrt_4, 1e-10)
        div_27 = torch.ops.aten.div.Tensor(mul_13, add_4);  mul_13 = add_4 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_14, [1], True);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Scalar(sqrt_4, 2);  sqrt_4 = None
        div_28 = torch.ops.aten.div.Tensor(sum_13, mul_15);  sum_13 = mul_15 = None
        expand_7 = torch.ops.aten.expand.default(div_28, [2, 256, 64, 64]);  div_28 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(relu_6, 1.0)
        mul_16 = torch.ops.aten.mul.Scalar(pow_21, 2.0);  pow_21 = None
        mul_17 = torch.ops.aten.mul.Tensor(expand_7, mul_16);  expand_7 = mul_16 = None
        add_17 = torch.ops.aten.add.Tensor(div_27, mul_17);  div_27 = mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(getitem_25, mul_18);  getitem_25 = mul_18 = None
        neg_3 = torch.ops.aten.neg.default(mul_19)
        mul_20 = torch.ops.aten.mul.Tensor(neg_3, div_30);  neg_3 = div_30 = None
        add_2 = torch.ops.aten.add.Tensor(sqrt_2, 1e-10)
        div_31 = torch.ops.aten.div.Tensor(mul_19, add_2);  mul_19 = add_2 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_20, [1], True);  mul_20 = None
        mul_21 = torch.ops.aten.mul.Scalar(sqrt_2, 2);  sqrt_2 = None
        div_32 = torch.ops.aten.div.Tensor(sum_14, mul_21);  sum_14 = mul_21 = None
        expand_8 = torch.ops.aten.expand.default(div_32, [2, 128, 128, 128]);  div_32 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(relu_3, 1.0)
        mul_22 = torch.ops.aten.mul.Scalar(pow_23, 2.0);  pow_23 = None
        mul_23 = torch.ops.aten.mul.Tensor(expand_8, mul_22);  expand_8 = mul_22 = None
        add_18 = torch.ops.aten.add.Tensor(div_31, mul_23);  div_31 = mul_23 = None
        mul_25 = torch.ops.aten.mul.Tensor(getitem_28, mul_24);  getitem_28 = mul_24 = None
        neg_4 = torch.ops.aten.neg.default(mul_25)
        mul_26 = torch.ops.aten.mul.Tensor(neg_4, div_34);  neg_4 = div_34 = None
        add = torch.ops.aten.add.Tensor(sqrt, 1e-10)
        div_35 = torch.ops.aten.div.Tensor(mul_25, add);  mul_25 = add = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_26, [1], True);  mul_26 = None
        mul_27 = torch.ops.aten.mul.Scalar(sqrt, 2);  sqrt = None
        div_36 = torch.ops.aten.div.Tensor(sum_15, mul_27);  sum_15 = mul_27 = None
        expand_9 = torch.ops.aten.expand.default(div_36, [2, 64, 256, 256]);  div_36 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(relu_1, 1.0)
        mul_28 = torch.ops.aten.mul.Scalar(pow_25, 2.0);  pow_25 = None
        mul_29 = torch.ops.aten.mul.Tensor(expand_9, mul_28);  expand_9 = mul_28 = None
        add_19 = torch.ops.aten.add.Tensor(div_35, mul_29);  div_35 = mul_29 = None
        le = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, add_15);  le = add_15 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where, relu_11, primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where = primals_29 = None
        getitem_31 = convolution_backward_5[0];  convolution_backward_5 = None
        le_1 = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_1 = torch.ops.aten.where.self(le_1, full_default, getitem_31);  le_1 = getitem_31 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_1, relu_10, primals_27, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_1 = primals_27 = None
        getitem_34 = convolution_backward_6[0];  convolution_backward_6 = None
        le_2 = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_2 = torch.ops.aten.where.self(le_2, full_default, getitem_34);  le_2 = getitem_34 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_2, getitem_6, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_2 = getitem_6 = primals_25 = None
        getitem_37 = convolution_backward_7[0];  convolution_backward_7 = None
        _low_memory_max_pool2d_offsets_to_indices_3 = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_7, 2, 32, [2, 2], [0, 0]);  getitem_7 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_37, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_3);  getitem_37 = _low_memory_max_pool2d_offsets_to_indices_3 = None
        add_20 = torch.ops.aten.add.Tensor(add_16, max_pool2d_with_indices_backward);  add_16 = max_pool2d_with_indices_backward = None
        le_3 = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_3 = torch.ops.aten.where.self(le_3, full_default, add_20);  le_3 = add_20 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_3, relu_8, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_3 = primals_23 = None
        getitem_40 = convolution_backward_8[0];  convolution_backward_8 = None
        le_4 = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_4 = torch.ops.aten.where.self(le_4, full_default, getitem_40);  le_4 = getitem_40 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_4, relu_7, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_4 = primals_21 = None
        getitem_43 = convolution_backward_9[0];  convolution_backward_9 = None
        le_5 = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_5 = torch.ops.aten.where.self(le_5, full_default, getitem_43);  le_5 = getitem_43 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_5, getitem_4, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_5 = getitem_4 = primals_19 = None
        getitem_46 = convolution_backward_10[0];  convolution_backward_10 = None
        _low_memory_max_pool2d_offsets_to_indices_2 = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_5, 2, 64, [2, 2], [0, 0]);  getitem_5 = None
        max_pool2d_with_indices_backward_1 = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_46, relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_2);  getitem_46 = _low_memory_max_pool2d_offsets_to_indices_2 = None
        add_21 = torch.ops.aten.add.Tensor(add_17, max_pool2d_with_indices_backward_1);  add_17 = max_pool2d_with_indices_backward_1 = None
        le_6 = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_6 = torch.ops.aten.where.self(le_6, full_default, add_21);  le_6 = add_21 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_6, relu_5, primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_6 = primals_17 = None
        getitem_49 = convolution_backward_11[0];  convolution_backward_11 = None
        le_7 = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_7 = torch.ops.aten.where.self(le_7, full_default, getitem_49);  le_7 = getitem_49 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_7, relu_4, primals_15, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_7 = primals_15 = None
        getitem_52 = convolution_backward_12[0];  convolution_backward_12 = None
        le_8 = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_8 = torch.ops.aten.where.self(le_8, full_default, getitem_52);  le_8 = getitem_52 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_8, getitem_2, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_8 = getitem_2 = primals_13 = None
        getitem_55 = convolution_backward_13[0];  convolution_backward_13 = None
        _low_memory_max_pool2d_offsets_to_indices_1 = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_3, 2, 128, [2, 2], [0, 0]);  getitem_3 = None
        max_pool2d_with_indices_backward_2 = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_55, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_1);  getitem_55 = _low_memory_max_pool2d_offsets_to_indices_1 = None
        add_22 = torch.ops.aten.add.Tensor(add_18, max_pool2d_with_indices_backward_2);  add_18 = max_pool2d_with_indices_backward_2 = None
        le_9 = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_9 = torch.ops.aten.where.self(le_9, full_default, add_22);  le_9 = add_22 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_9, relu_2, primals_11, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_9 = primals_11 = None
        getitem_58 = convolution_backward_14[0];  convolution_backward_14 = None
        le_10 = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_10 = torch.ops.aten.where.self(le_10, full_default, getitem_58);  le_10 = getitem_58 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_10, getitem, primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_10 = getitem = primals_9 = None
        getitem_61 = convolution_backward_15[0];  convolution_backward_15 = None
        _low_memory_max_pool2d_offsets_to_indices = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_1, 2, 256, [2, 2], [0, 0]);  getitem_1 = None
        max_pool2d_with_indices_backward_3 = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_61, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices);  getitem_61 = _low_memory_max_pool2d_offsets_to_indices = None
        add_23 = torch.ops.aten.add.Tensor(add_19, max_pool2d_with_indices_backward_3);  add_19 = max_pool2d_with_indices_backward_3 = None
        le_11 = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_11 = torch.ops.aten.where.self(le_11, full_default, add_23);  le_11 = add_23 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_11, relu, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_11 = primals_7 = None
        getitem_64 = convolution_backward_16[0];  convolution_backward_16 = None
        le_12 = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_12 = torch.ops.aten.where.self(le_12, full_default, getitem_64);  le_12 = full_default = getitem_64 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_12, div, primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_12 = div = primals_5 = None
        getitem_67 = convolution_backward_17[0];  convolution_backward_17 = None
        div_37 = torch.ops.aten.div.Tensor(getitem_67, primals_3);  getitem_67 = primals_3 = None
        return (None, div_37, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 3, 1, 1), is_leaf=True)  # primals_3
    buf1 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64, 3, 3, 3), (27, 1, 9, 3), is_leaf=True)  # primals_5
    buf2 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 64, 3, 3), (576, 1, 192, 64), is_leaf=True)  # primals_9
    buf4 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_11
    buf5 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256, 128, 3, 3), (1152, 1, 384, 128), is_leaf=True)  # primals_13
    buf6 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (256, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_15
    buf7 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_17
    buf8 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf8, (512, 256, 3, 3), (2304, 1, 768, 256), is_leaf=True)  # primals_19
    buf9 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf9, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_21
    buf10 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf10, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_23
    buf11 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_25
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_27
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512, 512, 3, 3), (4608, 1, 1536, 512), is_leaf=True)  # primals_29
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1, 64, 1, 1), is_leaf=True)  # primals_31
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1, 128, 1, 1), is_leaf=True)  # primals_32
    buf16 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1, 256, 1, 1), is_leaf=True)  # primals_33
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1, 512, 1, 1), is_leaf=True)  # primals_34
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1, 512, 1, 1), is_leaf=True)  # primals_35
    buf19 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2, 3, 256, 256), (196608, 1, 768, 3), is_leaf=True)  # div
    buf20 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf20, (2, 64, 256, 256), (4194304, 1, 16384, 64), is_leaf=True)  # relu
    buf21 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf21, (2, 64, 256, 256), (4194304, 1, 16384, 64), is_leaf=True)  # relu_1
    buf22 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf22, (2, 64, 128, 128), (1048576, 1, 8192, 64), is_leaf=True)  # getitem
    buf23 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf23, (2, 64, 128, 128), (1048576, 1, 8192, 64), dtype=torch.int8, is_leaf=True)  # getitem_1
    buf24 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf24, (2, 128, 128, 128), (2097152, 1, 16384, 128), is_leaf=True)  # relu_2
    buf25 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf25, (2, 128, 128, 128), (2097152, 1, 16384, 128), is_leaf=True)  # relu_3
    buf26 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf26, (2, 128, 64, 64), (524288, 1, 8192, 128), is_leaf=True)  # getitem_2
    buf27 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf27, (2, 128, 64, 64), (524288, 1, 8192, 128), dtype=torch.int8, is_leaf=True)  # getitem_3
    buf28 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf28, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # relu_4
    buf29 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # relu_5
    buf30 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf30, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # relu_6
    buf31 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf31, (2, 256, 32, 32), (262144, 1, 8192, 256), is_leaf=True)  # getitem_4
    buf32 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf32, (2, 256, 32, 32), (262144, 1, 8192, 256), dtype=torch.int8, is_leaf=True)  # getitem_5
    buf33 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf33, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # relu_7
    buf34 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf34, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # relu_8
    buf35 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf35, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # relu_9
    buf36 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf36, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # getitem_6
    buf37 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf37, (2, 512, 16, 16), (131072, 1, 8192, 512), dtype=torch.int8, is_leaf=True)  # getitem_7
    buf38 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf38, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # relu_10
    buf39 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf39, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # relu_11
    buf40 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # convolution_12
    buf41 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf41, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # convolution_25
    buf42 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf42, (2, 1, 256, 256), (65536, 1, 256, 1), is_leaf=True)  # sqrt
    buf43 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf43, (2, 64, 256, 256), (4194304, 1, 16384, 64), is_leaf=True)  # pow_3
    buf44 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2, 1, 128, 128), (16384, 1, 128, 1), is_leaf=True)  # sqrt_2
    buf45 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2, 128, 128, 128), (2097152, 1, 16384, 128), is_leaf=True)  # pow_6
    buf46 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2, 1, 64, 64), (4096, 1, 64, 1), is_leaf=True)  # sqrt_4
    buf47 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf47, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # pow_9
    buf48 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf48, (2, 1, 32, 32), (1024, 1, 32, 1), is_leaf=True)  # sqrt_6
    buf49 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf49, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # pow_12
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (2, 1, 16, 16), (256, 1, 16, 1), is_leaf=True)  # sqrt_8
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (2, 1, 16, 16), (256, 1, 16, 1), is_leaf=True)  # add_9
    buf52 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf52, (2, 512, 16, 16), (131072, 1, 8192, 512), is_leaf=True)  # pow_15
    buf53 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf53, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # mul_6
    buf54 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2, 512, 32, 32), (524288, 1, 16384, 512), is_leaf=True)  # div_22
    buf55 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf55, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # mul_12
    buf56 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf56, (2, 256, 64, 64), (1048576, 1, 16384, 256), is_leaf=True)  # div_26
    buf57 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf57, (2, 128, 128, 128), (2097152, 1, 16384, 128), is_leaf=True)  # mul_18
    buf58 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf58, (2, 128, 128, 128), (2097152, 1, 16384, 128), is_leaf=True)  # div_30
    buf59 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf59, (2, 64, 256, 256), (4194304, 1, 16384, 64), is_leaf=True)  # mul_24
    buf60 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf60, (2, 64, 256, 256), (4194304, 1, 16384, 64), is_leaf=True)  # div_34
    buf61 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf61, (2, 1, 1, 1), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)