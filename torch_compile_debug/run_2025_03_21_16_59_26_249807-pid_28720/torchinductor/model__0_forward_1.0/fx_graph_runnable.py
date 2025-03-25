
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
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_level', 'repro_after', 'constant_functions', 'skipfiles_inline_module_allowlist'}
torch._dynamo.config.reorderable_logging_functions = set()
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35):
        sub = torch.ops.aten.sub.Tensor(primals_2, primals_1);  primals_2 = None
        div = torch.ops.aten.div.Tensor(sub, primals_3);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(primals_4, primals_1);  primals_4 = primals_1 = None
        div_1 = torch.ops.aten.div.Tensor(sub_1, primals_3);  sub_1 = None
        convolution = torch.ops.aten.convolution.default(div, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1 = torch.ops.aten.convolution.default(relu, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem = _low_memory_max_pool2d_with_offsets[0]
        getitem_1 = _low_memory_max_pool2d_with_offsets[1];  _low_memory_max_pool2d_with_offsets = None
        convolution_2 = torch.ops.aten.convolution.default(getitem, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, primals_11, primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_3 = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2 = _low_memory_max_pool2d_with_offsets_1[0]
        getitem_3 = _low_memory_max_pool2d_with_offsets_1[1];  _low_memory_max_pool2d_with_offsets_1 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_2, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_4 = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, primals_15, primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_5 = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, primals_17, primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_6 = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool2d_with_offsets_2 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4 = _low_memory_max_pool2d_with_offsets_2[0]
        getitem_5 = _low_memory_max_pool2d_with_offsets_2[1];  _low_memory_max_pool2d_with_offsets_2 = None
        convolution_7 = torch.ops.aten.convolution.default(getitem_4, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_7 = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_7, primals_21, primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_8 = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_8, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_9 = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6 = _low_memory_max_pool2d_with_offsets_3[0]
        getitem_7 = _low_memory_max_pool2d_with_offsets_3[1];  _low_memory_max_pool2d_with_offsets_3 = None
        convolution_10 = torch.ops.aten.convolution.default(getitem_6, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_10 = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, primals_27, primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_11 = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_11, primals_29, primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_12 = torch.ops.aten.relu.default(convolution_12)
        convolution_13 = torch.ops.aten.convolution.default(div_1, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div_1 = primals_6 = None
        relu_13 = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_13, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = primals_8 = None
        relu_14 = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        _low_memory_max_pool2d_with_offsets_4 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_8 = _low_memory_max_pool2d_with_offsets_4[0];  _low_memory_max_pool2d_with_offsets_4 = None
        convolution_15 = torch.ops.aten.convolution.default(getitem_8, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_8 = primals_10 = None
        relu_15 = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_15, primals_11, primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_15 = primals_12 = None
        relu_16 = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
        _low_memory_max_pool2d_with_offsets_5 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_10 = _low_memory_max_pool2d_with_offsets_5[0];  _low_memory_max_pool2d_with_offsets_5 = None
        convolution_17 = torch.ops.aten.convolution.default(getitem_10, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_10 = primals_14 = None
        relu_17 = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_17, primals_15, primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_17 = primals_16 = None
        relu_18 = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_18, primals_17, primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_18 = primals_18 = None
        relu_19 = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
        _low_memory_max_pool2d_with_offsets_6 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_12 = _low_memory_max_pool2d_with_offsets_6[0];  _low_memory_max_pool2d_with_offsets_6 = None
        convolution_20 = torch.ops.aten.convolution.default(getitem_12, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_12 = primals_20 = None
        relu_20 = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_20, primals_21, primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_20 = primals_22 = None
        relu_21 = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_21, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_21 = primals_24 = None
        relu_22 = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
        _low_memory_max_pool2d_with_offsets_7 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_14 = _low_memory_max_pool2d_with_offsets_7[0];  _low_memory_max_pool2d_with_offsets_7 = None
        convolution_23 = torch.ops.aten.convolution.default(getitem_14, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_14 = primals_26 = None
        relu_23 = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_23, primals_27, primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_23 = primals_28 = None
        relu_24 = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_24, primals_29, primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_24 = primals_30 = None
        relu_25 = torch.ops.aten.relu.default(convolution_25)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(relu_1, 2)
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        sqrt = torch.ops.aten.sqrt.default(sum_1);  sum_1 = None
        add = torch.ops.aten.add.Tensor(sqrt, 1e-10)
        div_2 = torch.ops.aten.div.Tensor(relu_1, add)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(relu_14, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_2, [1], True);  pow_2 = None
        sqrt_1 = torch.ops.aten.sqrt.default(sum_2);  sum_2 = None
        add_1 = torch.ops.aten.add.Tensor(sqrt_1, 1e-10);  sqrt_1 = None
        div_3 = torch.ops.aten.div.Tensor(relu_14, add_1);  relu_14 = add_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(div_2, div_3);  div_3 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sub_2, 2)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(relu_3, 2)
        sum_3 = torch.ops.aten.sum.dim_IntList(pow_4, [1], True);  pow_4 = None
        sqrt_2 = torch.ops.aten.sqrt.default(sum_3);  sum_3 = None
        add_2 = torch.ops.aten.add.Tensor(sqrt_2, 1e-10)
        div_4 = torch.ops.aten.div.Tensor(relu_3, add_2)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(relu_16, 2)
        sum_4 = torch.ops.aten.sum.dim_IntList(pow_5, [1], True);  pow_5 = None
        sqrt_3 = torch.ops.aten.sqrt.default(sum_4);  sum_4 = None
        add_3 = torch.ops.aten.add.Tensor(sqrt_3, 1e-10);  sqrt_3 = None
        div_5 = torch.ops.aten.div.Tensor(relu_16, add_3);  relu_16 = add_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(div_4, div_5);  div_5 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sub_3, 2)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(relu_6, 2)
        sum_5 = torch.ops.aten.sum.dim_IntList(pow_7, [1], True);  pow_7 = None
        sqrt_4 = torch.ops.aten.sqrt.default(sum_5);  sum_5 = None
        add_4 = torch.ops.aten.add.Tensor(sqrt_4, 1e-10)
        div_6 = torch.ops.aten.div.Tensor(relu_6, add_4)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(relu_19, 2)
        sum_6 = torch.ops.aten.sum.dim_IntList(pow_8, [1], True);  pow_8 = None
        sqrt_5 = torch.ops.aten.sqrt.default(sum_6);  sum_6 = None
        add_5 = torch.ops.aten.add.Tensor(sqrt_5, 1e-10);  sqrt_5 = None
        div_7 = torch.ops.aten.div.Tensor(relu_19, add_5);  relu_19 = add_5 = None
        sub_4 = torch.ops.aten.sub.Tensor(div_6, div_7);  div_7 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sub_4, 2)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(relu_9, 2)
        sum_7 = torch.ops.aten.sum.dim_IntList(pow_10, [1], True);  pow_10 = None
        sqrt_6 = torch.ops.aten.sqrt.default(sum_7);  sum_7 = None
        add_6 = torch.ops.aten.add.Tensor(sqrt_6, 1e-10)
        div_8 = torch.ops.aten.div.Tensor(relu_9, add_6)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(relu_22, 2)
        sum_8 = torch.ops.aten.sum.dim_IntList(pow_11, [1], True);  pow_11 = None
        sqrt_7 = torch.ops.aten.sqrt.default(sum_8);  sum_8 = None
        add_7 = torch.ops.aten.add.Tensor(sqrt_7, 1e-10);  sqrt_7 = None
        div_9 = torch.ops.aten.div.Tensor(relu_22, add_7);  relu_22 = add_7 = None
        sub_5 = torch.ops.aten.sub.Tensor(div_8, div_9);  div_9 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(sub_5, 2)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(relu_12, 2)
        sum_9 = torch.ops.aten.sum.dim_IntList(pow_13, [1], True);  pow_13 = None
        sqrt_8 = torch.ops.aten.sqrt.default(sum_9);  sum_9 = None
        add_8 = torch.ops.aten.add.Tensor(sqrt_8, 1e-10)
        div_10 = torch.ops.aten.div.Tensor(relu_12, add_8);  relu_12 = add_8 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(relu_25, 2)
        sum_10 = torch.ops.aten.sum.dim_IntList(pow_14, [1], True);  pow_14 = None
        sqrt_9 = torch.ops.aten.sqrt.default(sum_10);  sum_10 = None
        add_9 = torch.ops.aten.add.Tensor(sqrt_9, 1e-10);  sqrt_9 = None
        div_11 = torch.ops.aten.div.Tensor(relu_25, add_9);  relu_25 = None
        sub_6 = torch.ops.aten.sub.Tensor(div_10, div_11);  div_10 = div_11 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(sub_6, 2);  sub_6 = None
        convolution_26 = torch.ops.aten.convolution.default(pow_3, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean = torch.ops.aten.mean.dim(convolution_26, [2, 3], True);  convolution_26 = None
        convolution_27 = torch.ops.aten.convolution.default(pow_6, primals_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_1 = torch.ops.aten.mean.dim(convolution_27, [2, 3], True);  convolution_27 = None
        convolution_28 = torch.ops.aten.convolution.default(pow_9, primals_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_2 = torch.ops.aten.mean.dim(convolution_28, [2, 3], True);  convolution_28 = None
        convolution_29 = torch.ops.aten.convolution.default(pow_12, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_3 = torch.ops.aten.mean.dim(convolution_29, [2, 3], True);  convolution_29 = None
        convolution_30 = torch.ops.aten.convolution.default(pow_15, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mean_4 = torch.ops.aten.mean.dim(convolution_30, [2, 3], True);  convolution_30 = None
        add_10 = torch.ops.aten.add.Tensor(mean, 0);  mean = None
        add_11 = torch.ops.aten.add.Tensor(add_10, mean_1);  add_10 = mean_1 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, mean_2);  add_11 = mean_2 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, mean_3);  add_12 = mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(add_13, mean_4);  add_13 = mean_4 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(sub_5, 1.0);  sub_5 = None
        mul_6 = torch.ops.aten.mul.Scalar(pow_18, 2.0);  pow_18 = None
        div_22 = torch.ops.aten.div.Tensor(div_8, add_6);  div_8 = add_6 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(sub_4, 1.0);  sub_4 = None
        mul_12 = torch.ops.aten.mul.Scalar(pow_20, 2.0);  pow_20 = None
        div_26 = torch.ops.aten.div.Tensor(div_6, add_4);  div_6 = add_4 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(sub_3, 1.0);  sub_3 = None
        mul_18 = torch.ops.aten.mul.Scalar(pow_22, 2.0);  pow_22 = None
        div_30 = torch.ops.aten.div.Tensor(div_4, add_2);  div_4 = add_2 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(sub_2, 1.0);  sub_2 = None
        mul_24 = torch.ops.aten.mul.Scalar(pow_24, 2.0);  pow_24 = None
        div_34 = torch.ops.aten.div.Tensor(div_2, add);  div_2 = add = None
        return (add_14, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, div, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, convolution_12, convolution_25, sqrt, pow_3, sqrt_2, pow_6, sqrt_4, pow_9, sqrt_6, pow_12, sqrt_8, add_9, pow_15, mul_6, div_22, mul_12, div_26, mul_18, div_30, mul_24, div_34)
        
def load_args(reader):
    buf0 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 3, 1, 1), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2, 3, 256, 256), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 3, 1, 1), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2, 3, 256, 256), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64, 3, 3, 3), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 64, 3, 3), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 64, 3, 3), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128, 128, 3, 3), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256, 128, 3, 3), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256, 256, 3, 3), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256, 256, 3, 3), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512, 256, 3, 3), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512, 512, 3, 3), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512, 512, 3, 3), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (512, 512, 3, 3), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512, 512, 3, 3), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf27, (512,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (512, 512, 3, 3), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1, 64, 1, 1), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1, 128, 1, 1), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1, 256, 1, 1), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1, 512, 1, 1), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1, 512, 1, 1), is_leaf=True)  # primals_35
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)