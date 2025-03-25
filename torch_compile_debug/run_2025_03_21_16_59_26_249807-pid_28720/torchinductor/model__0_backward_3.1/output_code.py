# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_elicer/x5/cx5ijtqvcev7jpniifwk4qmg5qzpbhst6a23ouatc5jtx57kbu62.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 256), kwargs = {})
#   %convolution_backward : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%div_12, %pow_15, %primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_div_0 = async_compile.triton('triton_poi_fused_convolution_backward_div_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.00390625
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sd/csdkax2cy34j2elu67sxe4aw3s4rtmqeqxzf4o4u676kq52lkpmb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 1024), kwargs = {})
#   %convolution_backward_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%div_13, %pow_12, %primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_div_1 = async_compile.triton('triton_poi_fused_convolution_backward_div_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_div_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0009765625
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/yl/cylswke64if7ew3ndw5vpxaju3xzacdxy66blmwj6rbeygalol4r.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_2, 4096), kwargs = {})
#   %convolution_backward_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%div_14, %pow_9, %primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_div_2 = async_compile.triton('triton_poi_fused_convolution_backward_div_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_div_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.000244140625
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/zy/czy6fpio3ceszulbhw53y6ojugqzx3axaqfp5kwczwt3fvbsmvnq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_3, 16384), kwargs = {})
#   %convolution_backward_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%div_15, %pow_6, %primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_div_3 = async_compile.triton('triton_poi_fused_convolution_backward_div_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_div_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = 6.103515625e-05
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fc/cfcwzj2ws7epu7kfjogxdlbrinjgzcbdxrvg7rtd5pt26mhunix6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_4, 65536), kwargs = {})
#   %convolution_backward_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%div_16, %pow_3, %primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_div_4 = async_compile.triton('triton_poi_fused_convolution_backward_div_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_div_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 65536
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = 1.52587890625e-05
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/r5/cr5zpakgw4y3wyvpdxm2wdw3qmbpwupprfznm3sjn2uojisb3b25.py
# Topologically Sorted Source Nodes: [input_30, input_60, add_8, truediv_10, truediv_11, sub_6], Original ATen: [aten.relu, aten.add, aten.div, aten.sub, aten.pow, aten.mul, aten.neg, aten.sum, aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   add_8 => add_8
#   input_30 => relu_12
#   input_60 => relu_25
#   sub_6 => sub_6
#   truediv_10 => div_10
#   truediv_11 => div_11
# Graph fragment:
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_8, 1e-10), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_12, %add_8), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_25, %add_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %div_11), kwargs = {})
#   %pow_16 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_16, 2.0), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_16, %mul), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_10, %add_8), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %div_18), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, %add_8), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1], True), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_12, 1.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_17, 2.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_5, %mul_4), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_19, %mul_5), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_12, 0), kwargs = {})
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %add_15), kwargs = {})
#   %convolution_backward_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where, %relu_11, %primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_per_fused_add_convolution_backward_div_mul_neg_pow_relu_sub_sum_threshold_backward_5 = async_compile.triton('triton_per_fused_add_convolution_backward_div_mul_neg_pow_relu_sub_sum_threshold_backward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div_mul_neg_pow_relu_sub_sum_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div_mul_neg_pow_relu_sub_sum_threshold_backward_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_out_ptr1 + (r1 + 512*x0), None)
    tmp4 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (r1 + 512*x0), None)
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp5 = 1e-10
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 / tmp6
    tmp9 = triton_helpers.maximum(tmp2, tmp8)
    tmp11 = tmp9 / tmp10
    tmp12 = tmp7 - tmp11
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp0 * tmp14
    tmp16 = -tmp15
    tmp17 = tmp7 / tmp6
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = 0.0
    tmp23 = tmp3 <= tmp22
    tmp24 = tmp15 / tmp6
    tmp25 = tmp4 * tmp13
    tmp26 = tmp21 / tmp25
    tmp27 = tmp3 * tmp13
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.where(tmp23, tmp22, tmp29)
    tl.store(in_out_ptr1 + (r1 + 512*x0), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/yc/cyc5h2uvchozqszircdvwuz4frcmnwcqbsheq4slbjxew3mf3ws5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_11, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_31), kwargs = {})
#   %convolution_backward_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_1, %relu_10, %primals_27, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_threshold_backward_6 = async_compile.triton('triton_poi_fused_convolution_backward_threshold_backward_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fi/cfiga3cjmedutpzjx3pchfovao3wotluyr4oypmkrhpdiyzt6db5.py
# Topologically Sorted Source Nodes: [add_6, input_24], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   add_6 => add_6
#   input_24 => _low_memory_max_pool2d_offsets_to_indices_3
# Graph fragment:
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_19, %mul_6), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_7,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %div_22), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_6, 1e-10), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_7, %add_6), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [1], True), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_9, 1.0), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_19, 2.0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_6, %mul_10), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_23, %mul_11), kwargs = {})
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_low_memory_max_pool2d_offsets_to_indices_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_7, 2, 32, [2, 2], [0, 0]), kwargs = {})
#   %max_pool2d_with_indices_backward : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%getitem_37, %relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool2d_offsets_to_indices_3), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %max_pool2d_with_indices_backward), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %add_20), kwargs = {})
#   %convolution_backward_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_3, %relu_8, %primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_7 = async_compile.triton('triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i8', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 32)
    x3 = ((xindex // 32) % 32)
    x4 = xindex // 1024
    x6 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 512*x0), None)
    tmp4 = tl.load(in_ptr2 + (r1 + 512*x0), None)
    tmp9 = tl.load(in_ptr3 + (r1 + 512*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 131072*x4), None)
    tmp21 = tl.load(in_ptr4 + (r1 + 512*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 131072*x4), None)
    tmp26 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.where((tmp9 < 0) != (tmp10 < 0), tl.where(tmp9 % tmp10 != 0, tmp9 // tmp10 - 1, tmp9 // tmp10), tmp9 // tmp10)
    tmp12 = tmp11 * tmp10
    tmp13 = tmp9 - tmp12
    tmp14 = 2*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (16)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0))))))
    tmp15 = tmp14 + tmp11
    tmp16 = 2*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp17 = tmp16 + tmp13
    tmp18 = tl.full([1], 32, tl.int64)
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19 + tmp17
    tmp22 = x6
    tmp23 = tmp20 == tmp22
    tmp24 = 0.0
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp27 = tmp26 <= tmp24
    tmp29 = 1e-10
    tmp30 = tmp28 + tmp29
    tmp31 = tmp2 / tmp30
    tmp32 = 2.0
    tmp33 = tmp28 * tmp32
    tmp34 = tmp8 / tmp33
    tmp35 = tmp26 * tmp32
    tmp36 = tmp34 * tmp35
    tmp37 = tmp31 + tmp36
    tmp38 = tmp37 + tmp25
    tmp39 = tl.where(tmp27, tmp24, tmp38)
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gl/cglkwuttjgb2figm7xjujzuutu7idjdjjp2wvdjyyrby6zgffbmi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_4 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default, %getitem_40), kwargs = {})
#   %convolution_backward_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_4, %relu_7, %primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_threshold_backward_8 = async_compile.triton('triton_poi_fused_convolution_backward_threshold_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/k6/ck6dduzwwsblcuns3qc6kepigoxwkdbculfewj5n3mso6f4vi5g5.py
# Topologically Sorted Source Nodes: [add_4, input_17], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   add_4 => add_4
#   input_17 => _low_memory_max_pool2d_offsets_to_indices_2
# Graph fragment:
#   %mul_13 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_22, %mul_12), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_13,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_2, %div_26), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_4, 1e-10), kwargs = {})
#   %div_27 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_13, %add_4), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [1], True), kwargs = {})
#   %pow_21 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_6, 1.0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_21, 2.0), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_7, %mul_16), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_27, %mul_17), kwargs = {})
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_low_memory_max_pool2d_offsets_to_indices_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_5, 2, 64, [2, 2], [0, 0]), kwargs = {})
#   %max_pool2d_with_indices_backward_1 : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%getitem_46, %relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool2d_offsets_to_indices_2), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %max_pool2d_with_indices_backward_1), kwargs = {})
#   %le_6 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default, %add_21), kwargs = {})
#   %convolution_backward_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_6, %relu_5, %primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_9 = async_compile.triton('triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i8', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 64)
    x3 = ((xindex // 64) % 64)
    x4 = xindex // 4096
    x6 = (xindex % 4096)
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 256*x0), None)
    tmp4 = tl.load(in_ptr2 + (r1 + 256*x0), None)
    tmp9 = tl.load(in_ptr3 + (r1 + 256*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 262144*x4), None)
    tmp21 = tl.load(in_ptr4 + (r1 + 256*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 262144*x4), None)
    tmp26 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.where((tmp9 < 0) != (tmp10 < 0), tl.where(tmp9 % tmp10 != 0, tmp9 // tmp10 - 1, tmp9 // tmp10), tmp9 // tmp10)
    tmp12 = tmp11 * tmp10
    tmp13 = tmp9 - tmp12
    tmp14 = 2*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (32)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0))))))
    tmp15 = tmp14 + tmp11
    tmp16 = 2*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (32)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp17 = tmp16 + tmp13
    tmp18 = tl.full([1], 64, tl.int64)
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19 + tmp17
    tmp22 = x6
    tmp23 = tmp20 == tmp22
    tmp24 = 0.0
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp27 = tmp26 <= tmp24
    tmp29 = 1e-10
    tmp30 = tmp28 + tmp29
    tmp31 = tmp2 / tmp30
    tmp32 = 2.0
    tmp33 = tmp28 * tmp32
    tmp34 = tmp8 / tmp33
    tmp35 = tmp26 * tmp32
    tmp36 = tmp34 * tmp35
    tmp37 = tmp31 + tmp36
    tmp38 = tmp37 + tmp25
    tmp39 = tl.where(tmp27, tmp24, tmp38)
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/pa/cpaglbd7nawnncezrqyero3rz7n5edprxrucfqtlyutcpdbualll.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_7 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default, %getitem_49), kwargs = {})
#   %convolution_backward_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_7, %relu_4, %primals_15, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_threshold_backward_10 = async_compile.triton('triton_poi_fused_convolution_backward_threshold_backward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/mh/cmhq6qu4rlidt5htme6bbghou6w637xwrru7xd5ixkm5tmcvfjji.py
# Topologically Sorted Source Nodes: [add_2, input_10], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   add_2 => add_2
#   input_10 => _low_memory_max_pool2d_offsets_to_indices_1
# Graph fragment:
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_25, %mul_18), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_19,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_3, %div_30), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_2, 1e-10), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_19, %add_2), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [1], True), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_3, 1.0), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_23, 2.0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_8, %mul_22), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_31, %mul_23), kwargs = {})
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_low_memory_max_pool2d_offsets_to_indices_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_3, 2, 128, [2, 2], [0, 0]), kwargs = {})
#   %max_pool2d_with_indices_backward_2 : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%getitem_55, %relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool2d_offsets_to_indices_1), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %max_pool2d_with_indices_backward_2), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_22), kwargs = {})
#   %convolution_backward_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_9, %relu_2, %primals_11, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_11 = async_compile.triton('triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i8', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 128)
    x3 = ((xindex // 128) % 128)
    x4 = xindex // 16384
    x6 = (xindex % 16384)
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), None)
    tmp4 = tl.load(in_ptr2 + (r1 + 128*x0), None)
    tmp9 = tl.load(in_ptr3 + (r1 + 128*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 524288*x4), None)
    tmp21 = tl.load(in_ptr4 + (r1 + 128*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 524288*x4), None)
    tmp26 = tl.load(in_out_ptr0 + (r1 + 128*x0), None)
    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tl.full([1, 1], 2, tl.int32)
    tmp11 = tl.where((tmp9 < 0) != (tmp10 < 0), tl.where(tmp9 % tmp10 != 0, tmp9 // tmp10 - 1, tmp9 // tmp10), tmp9 // tmp10)
    tmp12 = tmp11 * tmp10
    tmp13 = tmp9 - tmp12
    tmp14 = 2*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (64)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0))))))
    tmp15 = tmp14 + tmp11
    tmp16 = 2*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64))))) + ((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) * (((-1) + ((64) * ((64) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (64)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp17 = tmp16 + tmp13
    tmp18 = tl.full([1, 1], 128, tl.int64)
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19 + tmp17
    tmp22 = x6
    tmp23 = tmp20 == tmp22
    tmp24 = 0.0
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp27 = tmp26 <= tmp24
    tmp29 = 1e-10
    tmp30 = tmp28 + tmp29
    tmp31 = tmp2 / tmp30
    tmp32 = 2.0
    tmp33 = tmp28 * tmp32
    tmp34 = tmp8 / tmp33
    tmp35 = tmp26 * tmp32
    tmp36 = tmp34 * tmp35
    tmp37 = tmp31 + tmp36
    tmp38 = tmp37 + tmp25
    tmp39 = tl.where(tmp27, tmp24, tmp38)
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/v5/cv56akewfu5v52knv5jl4d6tybzbowqitp6voe37l2v76b6mvntr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_10 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %full_default, %getitem_58), kwargs = {})
#   %convolution_backward_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_10, %getitem, %primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_threshold_backward_12 = async_compile.triton('triton_poi_fused_convolution_backward_threshold_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hd/chdyp66iw6ackvppegn4lwadhjfx35s375s3dshlshyr4wg5dc27.py
# Topologically Sorted Source Nodes: [add, input_5], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   add => add
#   input_5 => _low_memory_max_pool2d_offsets_to_indices
# Graph fragment:
#   %mul_25 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_28, %mul_24), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_25,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_4, %div_34), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, 1e-10), kwargs = {})
#   %div_35 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_25, %add), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_26, [1], True), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_1, 1.0), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_25, 2.0), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_9, %mul_28), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_35, %mul_29), kwargs = {})
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_low_memory_max_pool2d_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_1, 2, 256, [2, 2], [0, 0]), kwargs = {})
#   %max_pool2d_with_indices_backward_3 : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%getitem_61, %relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool2d_offsets_to_indices), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %max_pool2d_with_indices_backward_3), kwargs = {})
#   %le_11 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_11, %full_default, %add_23), kwargs = {})
#   %convolution_backward_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_11, %relu, %primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_13 = async_compile.triton('triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i8', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 256)
    x3 = ((xindex // 256) % 256)
    x4 = xindex // 65536
    x6 = (xindex % 65536)
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 64*x0), None)
    tmp4 = tl.load(in_ptr2 + (r1 + 64*x0), None)
    tmp9 = tl.load(in_ptr3 + (r1 + 64*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 1048576*x4), None)
    tmp21 = tl.load(in_ptr4 + (r1 + 64*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))))) + 1048576*x4), None)
    tmp26 = tl.load(in_out_ptr0 + (r1 + 64*x0), None)
    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tl.full([1, 1], 2, tl.int32)
    tmp11 = tl.where((tmp9 < 0) != (tmp10 < 0), tl.where(tmp9 % tmp10 != 0, tmp9 // tmp10 - 1, tmp9 // tmp10), tmp9 // tmp10)
    tmp12 = tmp11 * tmp10
    tmp13 = tmp9 - tmp12
    tmp14 = 2*((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) * ((((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x3 // 2))) + (1 + (x3 // 2)) * ((1 + (x3 // 2)) < (128)))) < (((0) * ((0) >= (x3 // 2)) + (x3 // 2) * ((x3 // 2) > (0))))))
    tmp15 = tmp14 + tmp11
    tmp16 = 2*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128))))) + ((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) * (((-1) + ((128) * ((128) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (128)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp17 = tmp16 + tmp13
    tmp18 = tl.full([1, 1], 256, tl.int64)
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19 + tmp17
    tmp22 = x6
    tmp23 = tmp20 == tmp22
    tmp24 = 0.0
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp27 = tmp26 <= tmp24
    tmp29 = 1e-10
    tmp30 = tmp28 + tmp29
    tmp31 = tmp2 / tmp30
    tmp32 = 2.0
    tmp33 = tmp28 * tmp32
    tmp34 = tmp8 / tmp33
    tmp35 = tmp26 * tmp32
    tmp36 = tmp34 * tmp35
    tmp37 = tmp31 + tmp36
    tmp38 = tmp37 + tmp25
    tmp39 = tl.where(tmp27, tmp24, tmp38)
    tl.store(in_out_ptr0 + (r1 + 64*x0), tmp39, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ib/cib6z6npguciuam24fu2fkrrxg6vzwajmafwsavvzov7yzseun6j.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_12 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_12, %full_default, %getitem_64), kwargs = {})
#   %convolution_backward_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%where_12, %div, %primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_threshold_backward_14 = async_compile.triton('triton_poi_fused_convolution_backward_threshold_backward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_threshold_backward_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/eu/ceubi5wrgyarklxlwllm32sng44nu4etioe7uwpiq7aw5jbth6rk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem_67, %primals_3), kwargs = {})
triton_poi_fused_div_15 = async_compile.triton('triton_poi_fused_div_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, div, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, convolution_12, convolution_25, sqrt, pow_3, sqrt_2, pow_6, sqrt_4, pow_9, sqrt_6, pow_12, sqrt_8, add_9, pow_15, mul_6, div_22, mul_12, div_26, mul_18, div_30, mul_24, div_34, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_5, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_11, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_13, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_15, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_17, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_19, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_27, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_29, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_31, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_33, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_34, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_35, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(div, (2, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(relu, (2, 64, 256, 256), (4194304, 1, 16384, 64))
    assert_size_stride(relu_1, (2, 64, 256, 256), (4194304, 1, 16384, 64))
    assert_size_stride(getitem, (2, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(getitem_1, (2, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(relu_2, (2, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(relu_3, (2, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(getitem_2, (2, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(getitem_3, (2, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(relu_4, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(relu_5, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(relu_6, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(getitem_4, (2, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(getitem_5, (2, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(relu_7, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(relu_8, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(relu_9, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(getitem_6, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(getitem_7, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(relu_10, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(relu_11, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_12, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_25, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(sqrt, (2, 1, 256, 256), (65536, 1, 256, 1))
    assert_size_stride(pow_3, (2, 64, 256, 256), (4194304, 1, 16384, 64))
    assert_size_stride(sqrt_2, (2, 1, 128, 128), (16384, 1, 128, 1))
    assert_size_stride(pow_6, (2, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(sqrt_4, (2, 1, 64, 64), (4096, 1, 64, 1))
    assert_size_stride(pow_9, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(sqrt_6, (2, 1, 32, 32), (1024, 1, 32, 1))
    assert_size_stride(pow_12, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(sqrt_8, (2, 1, 16, 16), (256, 1, 16, 1))
    assert_size_stride(add_9, (2, 1, 16, 16), (256, 1, 16, 1))
    assert_size_stride(pow_15, (2, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(mul_6, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(div_22, (2, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(mul_12, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(div_26, (2, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(mul_18, (2, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(div_30, (2, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(mul_24, (2, 64, 256, 256), (4194304, 1, 16384, 64))
    assert_size_stride(div_34, (2, 64, 256, 256), (4194304, 1, 16384, 64))
    assert_size_stride(tangents_1, (2, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 1, 16, 16), (256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_div_0.run(tangents_1, buf0, 512, grid=grid(512), stream=stream0)
        buf3 = empty_strided_cuda((2, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_div_1.run(tangents_1, buf3, 2048, grid=grid(2048), stream=stream0)
        buf6 = empty_strided_cuda((2, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_div_2.run(tangents_1, buf6, 8192, grid=grid(8192), stream=stream0)
        buf9 = empty_strided_cuda((2, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_div_3.run(tangents_1, buf9, 32768, grid=grid(32768), stream=stream0)
        buf12 = empty_strided_cuda((2, 1, 256, 256), (65536, 65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_div_4.run(tangents_1, buf12, 131072, grid=grid(131072), stream=stream0)
        del tangents_1
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        buf1 = torch.ops.aten.convolution_backward.default(buf0, pow_15, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf0
        del pow_15
        del primals_35
        buf2 = buf1[0]
        del buf1
        buf15 = buf2; del buf2  # reuse
        buf21 = convolution_12; del convolution_12  # reuse
        # Topologically Sorted Source Nodes: [input_30, input_60, add_8, truediv_10, truediv_11, sub_6], Original ATen: [aten.relu, aten.add, aten.div, aten.sub, aten.pow, aten.mul, aten.neg, aten.sum, aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_backward_div_mul_neg_pow_relu_sub_sum_threshold_backward_5.run(buf15, buf21, sqrt_8, convolution_25, add_9, 512, 512, grid=grid(512), stream=stream0)
        del add_9
        del buf15
        del convolution_25
        del sqrt_8
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        buf4 = torch.ops.aten.convolution_backward.default(buf3, pow_12, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf3
        del pow_12
        del primals_34
        buf5 = buf4[0]
        del buf4
        # Topologically Sorted Source Nodes: [input_30, add_8], Original ATen: [aten.relu, aten.add, aten.div, aten.pow, aten.mul, aten.threshold_backward, aten.convolution_backward]
        buf22 = torch.ops.aten.convolution_backward.default(buf21, relu_11, primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf21
        del primals_29
        buf23 = buf22[0]
        del buf22
        buf24 = relu_11; del relu_11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_6.run(buf24, buf23, 262144, grid=grid(262144), stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf25 = torch.ops.aten.convolution_backward.default(buf24, relu_10, primals_27, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf24
        del primals_27
        buf26 = buf25[0]
        del buf25
        buf27 = relu_10; del relu_10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_6.run(buf27, buf26, 262144, grid=grid(262144), stream=stream0)
        del buf26
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf28 = torch.ops.aten.convolution_backward.default(buf27, getitem_6, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf27
        del getitem_6
        del primals_25
        buf29 = buf28[0]
        del buf28
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        buf7 = torch.ops.aten.convolution_backward.default(buf6, pow_9, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf6
        del pow_9
        del primals_33
        buf31 = relu_9; del relu_9  # reuse
        # Topologically Sorted Source Nodes: [add_6, input_24], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_7.run(buf31, buf5, mul_6, div_22, getitem_7, buf29, sqrt_6, 2048, 512, grid=grid(2048), stream=stream0)
        del buf29
        del buf5
        del div_22
        del getitem_7
        del mul_6
        del sqrt_6
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        buf10 = torch.ops.aten.convolution_backward.default(buf9, pow_6, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf9
        del pow_6
        del primals_32
        buf8 = buf7[0]
        del buf7
        buf11 = buf10[0]
        del buf10
        # Topologically Sorted Source Nodes: [add_6], Original ATen: [aten.mul, aten.add, aten.div, aten.pow, aten.threshold_backward, aten.convolution_backward]
        buf32 = torch.ops.aten.convolution_backward.default(buf31, relu_8, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf31
        del primals_23
        buf33 = buf32[0]
        del buf32
        buf34 = relu_8; del relu_8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_8.run(buf34, buf33, 1048576, grid=grid(1048576), stream=stream0)
        del buf33
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf35 = torch.ops.aten.convolution_backward.default(buf34, relu_7, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf34
        del primals_21
        buf36 = buf35[0]
        del buf35
        buf37 = relu_7; del relu_7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_8.run(buf37, buf36, 1048576, grid=grid(1048576), stream=stream0)
        del buf36
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf38 = torch.ops.aten.convolution_backward.default(buf37, getitem_4, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf37
        del getitem_4
        del primals_19
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.convolution_backward]
        buf13 = torch.ops.aten.convolution_backward.default(buf12, pow_3, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf12
        del pow_3
        del primals_31
        buf14 = buf13[0]
        del buf13
        buf39 = buf38[0]
        del buf38
        buf41 = relu_6; del relu_6  # reuse
        # Topologically Sorted Source Nodes: [add_4, input_17], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_9.run(buf41, buf8, mul_12, div_26, getitem_5, buf39, sqrt_4, 8192, 256, grid=grid(8192), stream=stream0)
        del buf39
        del buf8
        del div_26
        del getitem_5
        del mul_12
        del sqrt_4
        # Topologically Sorted Source Nodes: [add_4], Original ATen: [aten.mul, aten.add, aten.div, aten.pow, aten.threshold_backward, aten.convolution_backward]
        buf42 = torch.ops.aten.convolution_backward.default(buf41, relu_5, primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf41
        del primals_17
        buf43 = buf42[0]
        del buf42
        buf44 = relu_5; del relu_5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_10.run(buf44, buf43, 2097152, grid=grid(2097152), stream=stream0)
        del buf43
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf45 = torch.ops.aten.convolution_backward.default(buf44, relu_4, primals_15, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf44
        del primals_15
        buf46 = buf45[0]
        del buf45
        buf47 = relu_4; del relu_4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_10.run(buf47, buf46, 2097152, grid=grid(2097152), stream=stream0)
        del buf46
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf48 = torch.ops.aten.convolution_backward.default(buf47, getitem_2, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf47
        del getitem_2
        del primals_13
        buf49 = buf48[0]
        del buf48
        buf51 = relu_3; del relu_3  # reuse
        # Topologically Sorted Source Nodes: [add_2, input_10], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_11.run(buf51, buf11, mul_18, div_30, getitem_3, buf49, sqrt_2, 32768, 128, grid=grid(32768), stream=stream0)
        del buf11
        del buf49
        del div_30
        del getitem_3
        del mul_18
        del sqrt_2
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.mul, aten.add, aten.div, aten.pow, aten.threshold_backward, aten.convolution_backward]
        buf52 = torch.ops.aten.convolution_backward.default(buf51, relu_2, primals_11, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf51
        del primals_11
        buf53 = buf52[0]
        del buf52
        buf54 = relu_2; del relu_2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_12.run(buf54, buf53, 4194304, grid=grid(4194304), stream=stream0)
        del buf53
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf55 = torch.ops.aten.convolution_backward.default(buf54, getitem, primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf54
        del getitem
        del primals_9
        buf56 = buf55[0]
        del buf55
        buf58 = relu_1; del relu_1  # reuse
        # Topologically Sorted Source Nodes: [add, input_5], Original ATen: [aten.mul, aten.neg, aten.add, aten.div, aten.sum, aten.pow, aten.threshold_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_backward_div_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_pow_sum_threshold_backward_13.run(buf58, buf14, mul_24, div_34, getitem_1, buf56, sqrt, 131072, 64, grid=grid(131072), stream=stream0)
        del buf14
        del buf56
        del div_34
        del getitem_1
        del mul_24
        del sqrt
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.mul, aten.add, aten.div, aten.pow, aten.threshold_backward, aten.convolution_backward]
        buf59 = torch.ops.aten.convolution_backward.default(buf58, relu, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf58
        del primals_7
        buf60 = buf59[0]
        del buf59
        buf61 = relu; del relu  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_threshold_backward_14.run(buf61, buf60, 8388608, grid=grid(8388608), stream=stream0)
        del buf60
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.convolution_backward]
        buf62 = torch.ops.aten.convolution_backward.default(buf61, div, primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf61
        del div
        del primals_5
        buf63 = buf62[0]
        del buf62
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_15.run(buf64, primals_3, 393216, grid=grid(393216), stream=stream0)
        del primals_3
    return (None, buf64, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((2, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((2, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((2, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((2, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((2, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda:0', dtype=torch.int8)
    relu_2 = rand_strided((2, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((2, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((2, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((2, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.int8)
    relu_4 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((2, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((2, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.int8)
    relu_7 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.int8)
    relu_10 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    sqrt = rand_strided((2, 1, 256, 256), (65536, 1, 256, 1), device='cuda:0', dtype=torch.float32)
    pow_3 = rand_strided((2, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    sqrt_2 = rand_strided((2, 1, 128, 128), (16384, 1, 128, 1), device='cuda:0', dtype=torch.float32)
    pow_6 = rand_strided((2, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    sqrt_4 = rand_strided((2, 1, 64, 64), (4096, 1, 64, 1), device='cuda:0', dtype=torch.float32)
    pow_9 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    sqrt_6 = rand_strided((2, 1, 32, 32), (1024, 1, 32, 1), device='cuda:0', dtype=torch.float32)
    pow_12 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    sqrt_8 = rand_strided((2, 1, 16, 16), (256, 1, 16, 1), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((2, 1, 16, 16), (256, 1, 16, 1), device='cuda:0', dtype=torch.float32)
    pow_15 = rand_strided((2, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    mul_6 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((2, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((2, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    mul_18 = rand_strided((2, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((2, 128, 128, 128), (2097152, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((2, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((2, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((2, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, div, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, convolution_12, convolution_25, sqrt, pow_3, sqrt_2, pow_6, sqrt_4, pow_9, sqrt_6, pow_12, sqrt_8, add_9, pow_15, mul_6, div_22, mul_12, div_26, mul_18, div_30, mul_24, div_34, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
