# AOT ID: ['12_forward']
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


# kernel path: /tmp/torchinductor_elicer/gw/cgw4pepe3sy5mjb2mkwr3f3bmksrynrlmooeuci4g4xzyonp7wex.py
# Topologically Sorted Source Nodes: [truediv], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   truediv => div
# Graph fragment:
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view, 0.18215), kwargs = {})
triton_poi_fused_div_0 = async_compile.triton('triton_poi_fused_div_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex // 1024
    x0 = (xindex % 1024)
    x1 = ((xindex // 1024) % 4)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1 + 4*x2)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-4) + x1 + 4*x2)), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-8) + x1 + 4*x2)), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 1024*((-12) + x1 + 4*x2)), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp23 = 5.489980785067252
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x4), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/lm/clmoeuzh36mylgesfabqznpk3isswhv65xqfbaunjkj62d6mxrjy.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/mm/cmmiaracriz54r2vx2i2vljfl7eyxs4s7mor4bj7cu7awr7w564w.py
# Topologically Sorted Source Nodes: [result, mul, result_1, hidden_states], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => add_1, rsqrt, var_mean
#   mul => mul
#   result => convolution_1
#   result_1 => add
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 2.0), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %mul), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
triton_red_fused_add_convolution_mul_native_group_norm_2 = async_compile.triton('triton_red_fused_add_convolution_mul_native_group_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mul_native_group_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mul_native_group_norm_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 16384*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 16384.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/mo/cmohij2fyd5jclmc2qe7kmbvb3q7c6q6uspow6kzis5bpwze4lpn.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add_2, mul_2
#   hidden_states_1 => mul_3, sigmoid
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_2), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid), kwargs = {})
triton_poi_fused_native_group_norm_silu_3 = async_compile.triton('triton_poi_fused_native_group_norm_silu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bo/cbo6ax7nahthwjcl7t6ozfyap3ajofqhg2ctoshz22vzddjkr5n5.py
# Topologically Sorted Source Nodes: [result_6, mul_2, result_7, add, output_tensor, group_norm_2], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add => add_7
#   group_norm_2 => add_8, rsqrt_2, var_mean_2
#   mul_2 => mul_8
#   output_tensor => div_1
#   result_6 => convolution_7
#   result_7 => add_6
# Graph fragment:
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_7, %primals_183, %primals_184, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, 2.0), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %mul_8), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %add_6), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_7, 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_6, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
triton_red_fused_add_convolution_div_mul_native_group_norm_4 = async_compile.triton('triton_red_fused_add_convolution_div_mul_native_group_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_mul_native_group_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_div_mul_native_group_norm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r5 + 16384*x4), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp12, xmask)
    tl.store(out_ptr1 + (x4), tmp13, xmask)
    tmp15 = 16384.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.store(out_ptr2 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/si/csicqwzfpu6v23lalf6poslcws4b7gcywaxwst7b7svwzrd4wbca.py
# Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_2 => add_9, mul_10
# Graph fragment:
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %unsqueeze_15), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_13), kwargs = {})
triton_poi_fused_native_group_norm_5 = async_compile.triton('triton_poi_fused_native_group_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 16384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gk/cgktel6wrmjum4hwetgw7fmk6yrpwgwznsxq2yogowwdsxqj2tne.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 512*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/is/cisphkzb463icqpq7nokcwn3qiaupzlazp6hvw7idvpdegtlvpen.py
# Topologically Sorted Source Nodes: [result_9, mul_3, result_10], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_3 => mul_11
#   result_10 => add_11
#   result_9 => add_10
# Graph fragment:
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm, %primals_190), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 2.0), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_11), kwargs = {})
triton_poi_fused_add_mul_7 = async_compile.triton('triton_poi_fused_add_mul_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ns/cnsyd3gvmazglorwkzq3m5yokhquhce7eed75e6m7owouj4b4n2n.py
# Topologically Sorted Source Nodes: [hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   hidden_states_12 => add_17
#   hidden_states_13 => div_2
#   hidden_states_14 => clone_5
# Graph fragment:
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_45, %div_1), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, 1), kwargs = {})
#   %clone_5 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_div_8 = async_compile.triton('triton_poi_fused_add_clone_div_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_div_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_div_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 512*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/7t/c7top5ds7kpf2jqu6wcoddwed3gv6ybxdrhx5wq7drykjhrr7pm2.py
# Topologically Sorted Source Nodes: [hidden_states_14], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_14 => add_18, rsqrt_3, var_mean_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_46, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
triton_red_fused_native_group_norm_9 = async_compile.triton('triton_red_fused_native_group_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_9(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 16384*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = 16384.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ub/cubwk4gwrsevuma5o3v7t7ytqya5rxclbsftqrsmhtsv54o24r2j.py
# Topologically Sorted Source Nodes: [mul_9], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_9 => mul_23
# Graph fragment:
#   %mul_23 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_217, 1), kwargs = {})
triton_poi_fused_mul_10 = async_compile.triton('triton_poi_fused_mul_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5r/c5rtpwr5nbpn6vdzewavlr4q4lymm2m33fx7htyvrn776holi6li.py
# Topologically Sorted Source Nodes: [hidden_states_12, hidden_states_13, result_24, mul_8, result_25, add_2, output_tensor_1, mul_10, result_28, sample_1, hidden_states_19], Original ATen: [aten.add, aten.div, aten.convolution, aten.mul, aten.clone]
# Source node to ATen node mapping:
#   add_2 => add_24
#   hidden_states_12 => add_17
#   hidden_states_13 => div_2
#   hidden_states_19 => clone_7
#   mul_10 => mul_24
#   mul_8 => mul_22
#   output_tensor_1 => div_3
#   result_24 => convolution_13
#   result_25 => add_23
#   result_28 => add_25
#   sample_1 => add_26
# Graph fragment:
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_45, %div_1), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_21, %primals_213, %primals_214, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 2.0), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_13, %mul_22), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, %add_23), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_24, 1), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_18, 2.0), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_16, %mul_24), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %add_25), kwargs = {})
#   %clone_7 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_convolution_div_mul_11 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_mul_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_mul_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 512*x2 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp14 * tmp4
    tmp16 = tmp13 + tmp15
    tmp17 = tmp10 + tmp16
    tmp18 = tmp17 * tmp9
    tmp21 = tmp20 * tmp4
    tmp22 = tmp19 + tmp21
    tmp23 = tmp18 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp23, xmask)
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/dg/cdgqde7l52ifgv6tlyjh3q6jrv7f5caj5auzlypgyfhg3obldvm2.py
# Topologically Sorted Source Nodes: [result_33, mul_12, result_34, add_4, output_tensor_2, hidden_states_24], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_33
#   hidden_states_24 => add_34, clone_9, rsqrt_7, var_mean_7
#   mul_12 => mul_32
#   output_tensor_2 => div_4
#   result_33 => convolution_22
#   result_34 => add_32
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_24, 2.0), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %mul_32), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %add_32), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_33, 1.0), kwargs = {})
#   %clone_9 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_54, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_12 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(out_ptr0 + (r5 + 16384*x4), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp12, xmask)
    tmp15 = 16384.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/nt/cntn235s7jan3del5wpwedokx366yafhuibq4xjqscr5ayun5dcf.py
# Topologically Sorted Source Nodes: [result_33, mul_12, result_34, add_4, output_tensor_2, result_39, mul_14, result_40, add_5, output_tensor_3, hidden_states_29], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_33
#   add_5 => add_40
#   hidden_states_29 => add_41, clone_11, rsqrt_9, var_mean_9
#   mul_12 => mul_32
#   mul_14 => mul_40
#   output_tensor_2 => div_4
#   output_tensor_3 => div_5
#   result_33 => convolution_22
#   result_34 => add_32
#   result_39 => convolution_28
#   result_40 => add_39
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_24, 2.0), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %mul_32), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %add_32), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_33, 1.0), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_39, %primals_31, %primals_32, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_30, 2.0), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_28, %mul_40), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %add_39), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_40, 1.0), kwargs = {})
#   %clone_11 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_5,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_58, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-06), kwargs = {})
#   %rsqrt_9 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_13 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (r5 + 16384*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp13 = tmp11 + tmp12
        tmp15 = tmp14 * tmp5
        tmp16 = tmp13 + tmp15
        tmp17 = tmp10 + tmp16
        tmp18 = tmp17 * tmp9
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
        tl.store(in_out_ptr0 + (r5 + 16384*x4), tmp18, rmask & xmask)
        tl.store(out_ptr0 + (r5 + 16384*x4), tmp18, rmask & xmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp20, xmask)
    tmp23 = 16384.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/pl/cplgxfukpuif3nbhm72gyyosxhmh7tsijirkzguxmpug5neimj3w.py
# Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   hidden_states_34 => add_48, add_49, convert_element_type, convert_element_type_1, iota, mul_49, mul_50
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_48, torch.float32), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_49, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_50, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_14 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/jz/cjz7wnzywaaru3hvqpvwmrr5pmc44sjusrnc2lr7lupjal2v7ohz.py
# Topologically Sorted Source Nodes: [result_45, mul_16, result_46, add_6, output_tensor_4, hidden_states_34], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_6 => add_47
#   hidden_states_34 => _unsafe_index, clone_13
#   mul_16 => mul_48
#   output_tensor_4 => div_6
#   result_45 => convolution_34
#   result_46 => add_46
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_47, %primals_43, %primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_36, 2.0), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_34, %mul_48), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_5, %add_46), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_47, 1.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_6, [None, None, %unsqueeze_64, %convert_element_type_1]), kwargs = {})
#   %clone_13 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_15 = async_compile.triton('triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 32768) % 64)
    x1 = ((xindex // 512) % 64)
    x0 = (xindex % 512)
    x3 = xindex // 2097152
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x0 + 524288*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x0 + 524288*x3), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr4 + (tmp8 + 32*tmp4 + 1024*x0 + 524288*x3), None, eviction_policy='evict_last')
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tmp9 + tmp16
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hh/chhtwzu7xvvdjdjwey72adeuftdioaqtt4qxr3jukvearhnnxpqc.py
# Topologically Sorted Source Nodes: [result_48], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   result_48 => convolution_37
# Graph fragment:
#   %convolution_37 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %primals_47, %primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_16 = async_compile.triton('triton_poi_fused_convolution_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/jq/cjqeuebucmkbudfj4ltr2u3zqkrzmbdle2lfoi23wyz5spftk7iq.py
# Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_38 => convolution_38
# Graph fragment:
#   %convolution_38 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_17 = async_compile.triton('triton_poi_fused_convolution_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/w5/cw5b2ojrvzzgqceltylijspaho2jvuvsbojx4qcll7sfqzqtcrft.py
# Topologically Sorted Source Nodes: [mul_18], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_18 => mul_54
# Graph fragment:
#   %mul_54 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_221, 1), kwargs = {})
triton_poi_fused_mul_18 = async_compile.triton('triton_poi_fused_mul_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/qm/cqmszouysqjv7q3xpnr65n7pw4ddab675c2qrwx6ivaxntokcrj6.py
# Topologically Sorted Source Nodes: [result_48, mul_17, result_49, mul_19, result_52, sample_2, hidden_states_35], Original ATen: [aten.convolution, aten.mul, aten.add, aten.clone]
# Source node to ATen node mapping:
#   hidden_states_35 => clone_14
#   mul_17 => mul_53
#   mul_19 => mul_55
#   result_48 => convolution_37
#   result_49 => add_52
#   result_52 => add_53
#   sample_2 => add_54
# Graph fragment:
#   %convolution_37 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %primals_47, %primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_39, 2.0), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_37, %mul_53), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_42, 2.0), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_40, %mul_55), kwargs = {})
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_52, %add_53), kwargs = {})
#   %clone_14 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_54,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_convolution_mul_19 = async_compile.triton('triton_poi_fused_add_clone_convolution_mul_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_mul_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_mul_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 2097152*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 512*x2 + 2097152*y1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + 4096*y3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp8 * tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 + tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 4096*y3), tmp11, None)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/tx/ctxo6iebdtzvfwkguoe5cgcuwpcjbnx3ef2xynmvqtmapskc27yi.py
# Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_35 => add_55, rsqrt_11, var_mean_11
# Graph fragment:
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_62, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
triton_red_fused_native_group_norm_20 = async_compile.triton('triton_red_fused_native_group_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_20(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 65536*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = 65536.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/n3/cn3mcwqpmgzk34yu2qzfurdv6ym3pgycxjaql7qsaemudbrmx4z2.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_35 => add_56, mul_57
#   hidden_states_36 => mul_58, sigmoid_10
# Graph fragment:
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_63, %unsqueeze_70), kwargs = {})
#   %add_56 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %unsqueeze_67), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_56,), kwargs = {})
#   %mul_58 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %sigmoid_10), kwargs = {})
triton_poi_fused_native_group_norm_silu_21 = async_compile.triton('triton_poi_fused_native_group_norm_silu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 4096
    x1 = ((xindex // 4096) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/b2/cb2vrdqznrnny3xigp5dq64ahgkolj5n4h5qvhqwhvnzzy4het5g.py
# Topologically Sorted Source Nodes: [result_54, mul_20, result_55, hidden_states_37], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_37 => add_58, rsqrt_12, var_mean_12
#   mul_20 => mul_59
#   result_54 => convolution_43
#   result_55 => add_57
# Graph fragment:
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_58, %primals_53, %primals_54, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_45, 2.0), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_43, %mul_59), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_64, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-06), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_58,), kwargs = {})
triton_red_fused_add_convolution_mul_native_group_norm_22 = async_compile.triton('triton_red_fused_add_convolution_mul_native_group_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mul_native_group_norm_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mul_native_group_norm_22(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 65536*x4), tmp6, xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 65536.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/eo/ceopfzdechc5zfnngqotsntcv7hp74wzylqaye7zwzzvbr5yxl4g.py
# Topologically Sorted Source Nodes: [result_57, mul_21, result_58, add_8, output_tensor_5, hidden_states_40], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_61
#   hidden_states_40 => add_62, clone_16, rsqrt_13, var_mean_13
#   mul_21 => mul_63
#   output_tensor_5 => div_7
#   result_57 => convolution_46
#   result_58 => add_60
# Graph fragment:
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_62, %primals_59, %primals_60, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_48, 2.0), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_46, %mul_63), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %add_60), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_61, 1.0), kwargs = {})
#   %clone_16 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_66, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-06), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_62,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_23 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_ptr0 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(xmask, tmp12_weight_next, tmp12_weight)
        tl.store(out_ptr0 + (r5 + 65536*x4), tmp10, xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp12, xmask)
    tmp15 = 65536.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bp/cbpdnjgtf7l2aobpmc2jeyulmswxqh63j2gfqxobscivhqq5omuw.py
# Topologically Sorted Source Nodes: [result_57, mul_21, result_58, add_8, output_tensor_5, result_63, mul_23, result_64, add_9, output_tensor_6, hidden_states_45], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_61
#   add_9 => add_68
#   hidden_states_45 => add_69, clone_18, rsqrt_15, var_mean_15
#   mul_21 => mul_63
#   mul_23 => mul_71
#   output_tensor_5 => div_7
#   output_tensor_6 => div_8
#   result_57 => convolution_46
#   result_58 => add_60
#   result_63 => convolution_52
#   result_64 => add_67
# Graph fragment:
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_62, %primals_59, %primals_60, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_48, 2.0), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_46, %mul_63), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %add_60), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_61, 1.0), kwargs = {})
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_70, %primals_71, %primals_72, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_54, 2.0), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_52, %mul_71), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %add_67), kwargs = {})
#   %div_8 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_68, 1.0), kwargs = {})
#   %clone_18 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_8,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_70, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-06), kwargs = {})
#   %rsqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_24 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_24(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r3 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (r5 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp13 = tmp11 + tmp12
        tmp15 = tmp14 * tmp5
        tmp16 = tmp13 + tmp15
        tmp17 = tmp10 + tmp16
        tmp18 = tmp17 * tmp9
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(xmask, tmp20_weight_next, tmp20_weight)
        tl.store(in_out_ptr0 + (r5 + 65536*x4), tmp18, xmask)
        tl.store(out_ptr0 + (r5 + 65536*x4), tmp18, xmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp20, xmask)
    tmp23 = 65536.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sa/csanuzgzzfjfz6lyzj6j3gflkl7xwowzzbtu6jkm5r6aolj3im4y.py
# Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   hidden_states_50 => add_76, add_77, convert_element_type_4, convert_element_type_5, iota_2, mul_80, mul_81
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_76, torch.float32), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, 0.5), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_81, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_25 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/t6/ct6kppsttekkeioklcfxqyziecnsmmuvtc7ih2g25q4z3l6wbtm7.py
# Topologically Sorted Source Nodes: [result_69, mul_25, result_70, add_10, output_tensor_7, hidden_states_50], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_10 => add_75
#   hidden_states_50 => _unsafe_index_1, clone_20
#   mul_25 => mul_79
#   output_tensor_7 => div_9
#   result_69 => convolution_58
#   result_70 => add_74
# Graph fragment:
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_78, %primals_83, %primals_84, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_60, 2.0), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_58, %mul_79), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_8, %add_74), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_75, 1.0), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_9, [None, None, %unsqueeze_101, %convert_element_type_5]), kwargs = {})
#   %clone_20 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_26 = async_compile.triton('triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 65536) % 128)
    x1 = ((xindex // 512) % 128)
    x0 = (xindex % 512)
    x3 = xindex // 8388608
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 64*tmp4 + 4096*x0 + 2097152*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 64*tmp4 + 4096*x0 + 2097152*x3), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr4 + (tmp8 + 64*tmp4 + 4096*x0 + 2097152*x3), None, eviction_policy='evict_last')
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tmp9 + tmp16
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ha/chaf67xk4t3dknwpvrwinh426t7ex2ch52nvpqcpjm4mrb7hfacq.py
# Topologically Sorted Source Nodes: [mul_27], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_27 => mul_85
# Graph fragment:
#   %mul_85 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_225, 1), kwargs = {})
triton_poi_fused_mul_27 = async_compile.triton('triton_poi_fused_mul_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/m4/cm4yalonbra5zifmzsmilrf6hapdwyrwhknkspisi2r5fzgkrw7e.py
# Topologically Sorted Source Nodes: [result_72, mul_26, result_73, mul_28, result_76, sample_3], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_26 => mul_84
#   mul_28 => mul_86
#   result_72 => convolution_61
#   result_73 => add_80
#   result_76 => add_81
#   sample_3 => add_82
# Graph fragment:
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %primals_87, %primals_88, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_63, 2.0), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_61, %mul_84), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_66, 2.0), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_64, %mul_86), kwargs = {})
#   %add_82 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %add_81), kwargs = {})
triton_poi_fused_add_convolution_mul_28 = async_compile.triton('triton_poi_fused_add_convolution_mul_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 512
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16384)
    y1 = yindex // 16384
    tmp0 = tl.load(in_out_ptr0 + (x2 + 512*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + 512*y3), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + 16384*x2 + 8388608*y1), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + 16384*x2 + 8388608*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp8 * tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 + tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 512*y3), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/uk/cukxkroso5oklrsj6b726wyivampbjbhb72dzz2ko6pw2bazwvas.py
# Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_51 => add_83, rsqrt_17, var_mean_17
# Graph fragment:
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_74, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-06), kwargs = {})
#   %rsqrt_17 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_83,), kwargs = {})
triton_red_fused_native_group_norm_29 = async_compile.triton('triton_red_fused_native_group_norm_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_29(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r2 = (rindex % 16)
        r3 = rindex // 16
        tmp0 = tl.load(in_ptr0 + (r2 + 16*x0 + 512*r3 + 8388608*x1), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tmp5 = 262144.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/qz/cqzadojykoo6sxnxsk347bit3eivw3fpyuhjcerurvaqhr2ts7ff.py
# Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_51 => add_84, mul_88
#   hidden_states_52 => mul_89, sigmoid_16
# Graph fragment:
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_75, %unsqueeze_107), kwargs = {})
#   %add_84 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_104), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_84,), kwargs = {})
#   %mul_89 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %sigmoid_16), kwargs = {})
triton_poi_fused_native_group_norm_silu_30 = async_compile.triton('triton_poi_fused_native_group_norm_silu_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 512
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 16384
    y0 = (yindex % 16384)
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (32*y1 + (x2 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*y1 + (x2 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (y0 + 16384*x2 + 8388608*y1), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/7v/c7vluucfexnk53up54fy2nrl337fe54ypgslatue4gsuiyl4duyh.py
# Topologically Sorted Source Nodes: [result_78, mul_29, result_79, hidden_states_53], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_53 => add_86, rsqrt_18, var_mean_18
#   mul_29 => mul_90
#   result_78 => convolution_67
#   result_79 => add_85
# Graph fragment:
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_89, %primals_93, %primals_94, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_69, 2.0), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_67, %mul_90), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-06), kwargs = {})
#   %rsqrt_18 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_86,), kwargs = {})
triton_red_fused_add_convolution_mul_native_group_norm_31 = async_compile.triton('triton_red_fused_add_convolution_mul_native_group_norm_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mul_native_group_norm_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mul_native_group_norm_31(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 16384
        tmp0 = tl.load(in_out_ptr0 + (r5 + 131072*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 131072*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 131072*x4), tmp6, xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 131072.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bd/cbd2vcygmc3q3rufsxcjbi7gjwk2ob7fxcf4rm5grjcqxj4ghmih.py
# Topologically Sorted Source Nodes: [hidden_states_53, hidden_states_54], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_53 => add_87, mul_92
#   hidden_states_54 => mul_93, sigmoid_17
# Graph fragment:
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_77, %unsqueeze_113), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_110), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_87,), kwargs = {})
#   %mul_93 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, %sigmoid_17), kwargs = {})
triton_poi_fused_native_group_norm_silu_32 = async_compile.triton('triton_poi_fused_native_group_norm_silu_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16384
    x1 = ((xindex // 16384) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/yv/cyvn2fneuqwi5duwhkbzgeode5wk36fa6poxaqk6fnqozdj32mqm.py
# Topologically Sorted Source Nodes: [result_81, mul_30, result_82, result_84, mul_31, result_85, add_12, output_tensor_8, hidden_states_56], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_12 => add_90
#   hidden_states_56 => clone_23
#   mul_30 => mul_94
#   mul_31 => mul_95
#   output_tensor_8 => div_10
#   result_81 => convolution_70
#   result_82 => add_88
#   result_84 => convolution_73
#   result_85 => add_89
# Graph fragment:
#   %convolution_70 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_93, %primals_99, %primals_100, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_72, 2.0), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_70, %mul_94), kwargs = {})
#   %convolution_73 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_82, %primals_103, %primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_75, 2.0), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_73, %mul_95), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %add_88), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_90, 1.0), kwargs = {})
#   %clone_23 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_10,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_convolution_div_mul_33 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_mul_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_mul_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_mul_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 4194304*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 256*x2 + 4194304*y1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + 16384*y3), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + 16384*y3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = tmp6 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 16384*y3), tmp15, None)
    tl.store(out_ptr0 + (x2 + 16384*y3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gn/cgnk4dx2i2elpdbhtarx4mpttzh3zfkhhfc3zwvkqkw5u6etyawa.py
# Topologically Sorted Source Nodes: [hidden_states_56], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_56 => add_91, rsqrt_19, var_mean_19
# Graph fragment:
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_78, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-06), kwargs = {})
#   %rsqrt_19 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_91,), kwargs = {})
triton_red_fused_native_group_norm_34 = async_compile.triton('triton_red_fused_native_group_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_34(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = 131072.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/su/csu4vc4of7pdcbazbqpquc5jzb2huo4gq2ltmjn3ckm3iletafkc.py
# Topologically Sorted Source Nodes: [result_90, mul_33, result_91, add_13, output_tensor_9, hidden_states_61], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_13 => add_97
#   hidden_states_61 => add_98, clone_25, rsqrt_21, var_mean_21
#   mul_33 => mul_103
#   output_tensor_9 => div_11
#   result_90 => convolution_79
#   result_91 => add_96
# Graph fragment:
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_102, %primals_115, %primals_116, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 2.0), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_79, %mul_103), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %add_96), kwargs = {})
#   %div_11 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_97, 1.0), kwargs = {})
#   %clone_25 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_11,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_82, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-06), kwargs = {})
#   %rsqrt_21 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_35 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 16384
        tmp0 = tl.load(in_ptr0 + (r5 + 131072*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 131072*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + 8*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r5 + 131072*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(xmask, tmp12_weight_next, tmp12_weight)
        tl.store(out_ptr0 + (r5 + 131072*x4), tmp10, xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp12, xmask)
    tmp15 = 131072.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gz/cgza3qus7bqyffm4ipfy4co5tflxutc7h6gp26kvp3k72xn4ue6l.py
# Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   hidden_states_66 => add_105, add_106, convert_element_type_8, convert_element_type_9, iota_4, mul_112, mul_113
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, 0), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.float32), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, 0.0), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, 0.5), kwargs = {})
#   %convert_element_type_9 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_113, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_36 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_36(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/2m/c2mdnwopkeqoqxbrquuiidi4tlsqyf3ktlhbzdpd3adhgnne5jt7.py
# Topologically Sorted Source Nodes: [result_90, mul_33, result_91, add_13, output_tensor_9, result_96, mul_35, result_97, add_14, output_tensor_10, hidden_states_66], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_13 => add_97
#   add_14 => add_104
#   hidden_states_66 => _unsafe_index_2, clone_27
#   mul_33 => mul_103
#   mul_35 => mul_111
#   output_tensor_10 => div_12
#   output_tensor_9 => div_11
#   result_90 => convolution_79
#   result_91 => add_96
#   result_96 => convolution_85
#   result_97 => add_103
# Graph fragment:
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_102, %primals_115, %primals_116, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 2.0), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_79, %mul_103), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %add_96), kwargs = {})
#   %div_11 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_97, 1.0), kwargs = {})
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_110, %primals_127, %primals_128, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 2.0), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_85, %mul_111), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_11, %add_103), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_104, 1.0), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_12, [None, None, %unsqueeze_138, %convert_element_type_9]), kwargs = {})
#   %clone_27 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_2,), kwargs = {memory_format: torch.channels_last})
triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_37 = async_compile.triton('triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex // 256
    x2 = (xindex % 256)
    y5 = yindex
    y0 = (yindex % 256)
    x4 = xindex
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 128*tmp4 + 16384*y5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 128*tmp4 + 16384*y5), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr4 + (tmp8 + 128*tmp4 + 16384*y5), None, eviction_policy='evict_last')
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tmp9 + tmp16
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (tmp8 + 128*tmp4 + 16384*y5), None, eviction_policy='evict_last')
    tmp22 = tmp20 + tmp21
    tmp23 = tl.load(in_ptr7 + (tmp8 + 128*tmp4 + 16384*y5), None, eviction_policy='evict_last')
    tmp24 = tmp23 * tmp14
    tmp25 = tmp22 + tmp24
    tmp26 = tmp19 + tmp25
    tmp27 = tmp26 * tmp18
    tl.store(out_ptr1 + (y0 + 256*x4 + 16777216*y1), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gn/cgnhnqgrfpsllisu5iljkbtbuu226ji55gtv6jwfgqykfucukolg.py
# Topologically Sorted Source Nodes: [result_99], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   result_99 => convolution_88
# Graph fragment:
#   %convolution_88 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_27, %primals_131, %primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/a6/ca6s3uhgkq5nmy57wcxyzn5n5no5bfo4jvavjd3owksndzi5hsqr.py
# Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_89 => convolution_89
# Graph fragment:
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_27, %primals_133, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_39 = async_compile.triton('triton_poi_fused_convolution_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3z/c3z7r6lkhnc2pbmzor5v2nqggjohkgfp73fyzazft4vawxcs54ev.py
# Topologically Sorted Source Nodes: [mul_37], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_37 => mul_117
# Graph fragment:
#   %mul_117 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_229, 1), kwargs = {})
triton_poi_fused_mul_40 = async_compile.triton('triton_poi_fused_mul_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bn/cbnsuwjgtrhljhmsnbadnrr4yjahvlu65e4wbgikzx7kw6sqmemk.py
# Topologically Sorted Source Nodes: [result_99, mul_36, result_100, mul_38, result_103, sample_4], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_36 => mul_116
#   mul_38 => mul_118
#   result_100 => add_109
#   result_103 => add_110
#   result_99 => convolution_88
#   sample_4 => add_111
# Graph fragment:
#   %convolution_88 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_27, %primals_131, %primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_90, 2.0), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_88, %mul_116), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 2.0), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_91, %mul_118), kwargs = {})
#   %add_111 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %add_110), kwargs = {})
triton_poi_fused_add_convolution_mul_41 = async_compile.triton('triton_poi_fused_add_convolution_mul_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 65536)
    y1 = yindex // 65536
    tmp0 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + 65536*x2 + 16777216*y1), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + 65536*x2 + 16777216*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp8 * tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 + tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hg/chgehcqi44lgd453wbysnmq4qxb7dnsu7da665mbs74arxidj3cv.py
# Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_67 => add_112, rsqrt_23, var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_86, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-06), kwargs = {})
#   %rsqrt_23 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_112,), kwargs = {})
triton_red_fused_native_group_norm_42 = async_compile.triton('triton_red_fused_native_group_norm_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_42(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r2 = (rindex % 8)
        r3 = rindex // 8
        tmp0 = tl.load(in_ptr0 + (r2 + 8*x0 + 256*r3 + 16777216*x1), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tmp5 = 524288.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/pi/cpieq23z6lygni2sgjz2d7mxvpwnjsw2i5xt4pwp35e4kclzmlg4.py
# Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_67 => add_113, mul_120
#   hidden_states_68 => mul_121, sigmoid_22
# Graph fragment:
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_87, %unsqueeze_144), kwargs = {})
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_141), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_113,), kwargs = {})
#   %mul_121 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_113, %sigmoid_22), kwargs = {})
triton_poi_fused_native_group_norm_silu_43 = async_compile.triton('triton_poi_fused_native_group_norm_silu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 65536
    y0 = (yindex % 65536)
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (32*y1 + (x2 // 8)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*y1 + (x2 // 8)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (y0 + 65536*x2 + 16777216*y1), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gd/cgdcghqrkvwkt6dpiapvjjixlgct3wgkboekmgol3gwagy4epy4g.py
# Topologically Sorted Source Nodes: [result_105, mul_39, result_106, hidden_states_69], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_69 => add_115, rsqrt_24, var_mean_24
#   mul_39 => mul_122
#   result_105 => convolution_94
#   result_106 => add_114
# Graph fragment:
#   %convolution_94 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_121, %primals_137, %primals_138, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_96, 2.0), kwargs = {})
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_94, %mul_122), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_88, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-06), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_115,), kwargs = {})
triton_red_fused_add_convolution_mul_native_group_norm_44 = async_compile.triton('triton_red_fused_add_convolution_mul_native_group_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mul_native_group_norm_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mul_native_group_norm_44(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 65536
        tmp0 = tl.load(in_out_ptr0 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 4*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 262144*x4), tmp6, xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 262144.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bh/cbhbkglqscmmem7bdezbgkbj755dmvqmghrquqfjpqh7jghj6l3s.py
# Topologically Sorted Source Nodes: [hidden_states_69, hidden_states_70], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_69 => add_116, mul_124
#   hidden_states_70 => mul_125, sigmoid_23
# Graph fragment:
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_89, %unsqueeze_150), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %unsqueeze_147), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_125 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_23), kwargs = {})
triton_poi_fused_native_group_norm_silu_45 = async_compile.triton('triton_poi_fused_native_group_norm_silu_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 65536
    x1 = ((xindex // 65536) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/pu/cpuxrwdmvbra5qoexh7hyuyxnad4vrbfcubucijacteu4qgmmapo.py
# Topologically Sorted Source Nodes: [result_108, mul_40, result_109, result_111, mul_41, result_112, add_16, output_tensor_11, hidden_states_72], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone]
# Source node to ATen node mapping:
#   add_16 => add_119
#   hidden_states_72 => clone_30
#   mul_40 => mul_126
#   mul_41 => mul_127
#   output_tensor_11 => div_13
#   result_108 => convolution_97
#   result_109 => add_117
#   result_111 => convolution_100
#   result_112 => add_118
# Graph fragment:
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_125, %primals_143, %primals_144, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_99, 2.0), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_97, %mul_126), kwargs = {})
#   %convolution_100 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_147, %primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_102, 2.0), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_100, %mul_127), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %add_117), kwargs = {})
#   %div_13 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_119, 1.0), kwargs = {})
#   %clone_30 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_13,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_convolution_div_mul_46 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_mul_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_mul_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_mul_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8388608*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 128*x2 + 8388608*y1), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + 65536*y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + 65536*y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = tmp6 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 65536*y3), tmp15, ymask)
    tl.store(out_ptr0 + (x2 + 65536*y3), tmp15, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/i5/ci52vr5gehhrk2hjedrn6j4eo5lbgbb3xpdnyfebjxlhphbk2nej.py
# Topologically Sorted Source Nodes: [hidden_states_72], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_72 => add_120, rsqrt_25, var_mean_25
# Graph fragment:
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_90, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_120,), kwargs = {})
triton_red_fused_native_group_norm_47 = async_compile.triton('triton_red_fused_native_group_norm_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_47(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 262144*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = 262144.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ps/cpsnir6kv2zvk24qjlacwcpduh4lohk5s4lqphpztwz35k5nmgcw.py
# Topologically Sorted Source Nodes: [result_117, mul_43, result_118, add_17, output_tensor_12, hidden_states_77], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_17 => add_126
#   hidden_states_77 => add_127, clone_32, rsqrt_27, var_mean_27
#   mul_43 => mul_135
#   output_tensor_12 => div_14
#   result_117 => convolution_106
#   result_118 => add_125
# Graph fragment:
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_134, %primals_159, %primals_160, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 2.0), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_106, %mul_135), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %add_125), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_126, 1.0), kwargs = {})
#   %clone_32 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_14,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_94, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_127,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_48 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 65536
        tmp0 = tl.load(in_ptr0 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + 4*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(xmask, tmp12_weight_next, tmp12_weight)
        tl.store(out_ptr0 + (r5 + 262144*x4), tmp10, xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr1 + (x4), tmp12, xmask)
    tmp15 = 262144.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cg/ccgvqlwrecmff4gz5ifg3urdqmf7sfnx6wy3lievl2n5725xq7zm.py
# Topologically Sorted Source Nodes: [result_117, mul_43, result_118, add_17, output_tensor_12, result_123, mul_45, result_124, add_18, output_tensor_13, sample_5], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_17 => add_126
#   add_18 => add_133
#   mul_43 => mul_135
#   mul_45 => mul_143
#   output_tensor_12 => div_14
#   output_tensor_13 => div_15
#   result_117 => convolution_106
#   result_118 => add_125
#   result_123 => convolution_112
#   result_124 => add_132
#   sample_5 => add_134, clone_34, rsqrt_29, var_mean_29
# Graph fragment:
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_134, %primals_159, %primals_160, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_108, 2.0), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_106, %mul_135), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %add_125), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_126, 1.0), kwargs = {})
#   %convolution_112 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_142, %primals_171, %primals_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_114, 2.0), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_112, %mul_143), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_14, %add_132), kwargs = {})
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_133, 1.0), kwargs = {})
#   %clone_34 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%div_15,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_98, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-06), kwargs = {})
#   %rsqrt_29 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_134,), kwargs = {})
triton_red_fused_add_clone_convolution_div_mul_native_group_norm_49 = async_compile.triton('triton_red_fused_add_clone_convolution_div_mul_native_group_norm_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_mul_native_group_norm_49', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_mul_native_group_norm_49(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex
        r3 = rindex // 65536
        tmp0 = tl.load(in_out_ptr0 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 4*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r3 + 4*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (r5 + 262144*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 1.0
        tmp10 = tmp8 * tmp9
        tmp13 = tmp11 + tmp12
        tmp15 = tmp14 * tmp5
        tmp16 = tmp13 + tmp15
        tmp17 = tmp10 + tmp16
        tmp18 = tmp17 * tmp9
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(xmask, tmp20_weight_next, tmp20_weight)
        tl.store(in_out_ptr0 + (r5 + 262144*x4), tmp18, xmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp20, xmask)
    tmp23 = 262144.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gl/cglwqppaktrb2ozmk4c7um2ssll5h6ghr54tfnm75q5566y3ax4o.py
# Topologically Sorted Source Nodes: [result_126, mul_46, result_127, x_decoded], Original ATen: [aten.convolution, aten.mul, aten.add, aten.clamp, aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
#   mul_46 => mul_147
#   result_126 => convolution_115
#   result_127 => add_136
#   x_decoded => clamp_max, clamp_min
# Graph fragment:
#   %convolution_115 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_146, %primals_235, %primals_236, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_117, 2.0), kwargs = {})
#   %add_136 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_115, %mul_147), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_136, -1), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_136, -1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_136, 1), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
triton_poi_fused_add_clamp_convolution_ge_le_logical_and_mul_50 = async_compile.triton('triton_poi_fused_add_clamp_convolution_ge_le_logical_and_mul_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_convolution_ge_le_logical_and_mul_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_convolution_ge_le_logical_and_mul_50(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 65536) % 3)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = -1.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1.0
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp6 >= tmp7
    tmp12 = tmp6 <= tmp9
    tmp13 = tmp11 & tmp12
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238 = args
    args.clear()
    assert_size_stride(primals_1, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_2, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_3, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_4, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_5, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (512, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_10, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_16, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_28, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_34, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_40, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_46, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_47, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_50, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_56, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_62, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_68, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_74, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_80, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_86, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_87, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_90, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_96, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_102, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_103, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (4, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_106, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_112, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_118, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_124, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_130, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_131, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_132, (256, ), (1, ))
    assert_size_stride(primals_133, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_134, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_140, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_146, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_147, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_150, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_156, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_162, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_168, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_174, (128, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_180, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_183, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_186, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (512, 512), (512, 1))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (4, 512), (512, 1))
    assert_size_stride(primals_192, (512, 4), (4, 1))
    assert_size_stride(primals_193, (512, 512), (512, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (4, 512), (512, 1))
    assert_size_stride(primals_196, (512, 4), (4, 1))
    assert_size_stride(primals_197, (512, 512), (512, 1))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (4, 512), (512, 1))
    assert_size_stride(primals_200, (512, 4), (4, 1))
    assert_size_stride(primals_201, (512, 512), (512, 1))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (4, 512), (512, 1))
    assert_size_stride(primals_204, (512, 4), (4, 1))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_210, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (512, ), (1, ))
    assert_size_stride(primals_213, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_216, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_217, (4, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(primals_218, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_219, (4, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_220, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_221, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_222, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_223, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_224, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_225, (4, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(primals_226, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_227, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_228, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_229, (4, 128, 256, 256), (8388608, 65536, 256, 1))
    assert_size_stride(primals_230, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_231, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_232, (256, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (3, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_236, (3, ), (1, ))
    assert_size_stride(primals_237, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_238, (3, 4, 1, 1), (4, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_0.run(primals_4, primals_3, primals_2, primals_1, buf0, 16384, grid=grid(16384), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_4
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf2, primals_6, 16384, grid=grid(16384), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [result], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf6 = buf3; del buf3  # reuse
        buf7 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf10 = reinterpret_tensor(buf8, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [result, mul, result_1, hidden_states], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf6, buf10, primals_8, buf5, buf7, 128, 16384, grid=grid(128), stream=stream0)
        del primals_8
        buf11 = buf5; del buf5  # reuse
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf12, buf6, buf7, buf10, primals_175, primals_176, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_3], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf12, primals_179, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf16 = buf13; del buf13  # reuse
        buf17 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf20 = reinterpret_tensor(buf18, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [result_3, mul_1, result_4, hidden_states_2], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf16, buf20, primals_178, buf15, buf17, 128, 16384, grid=grid(128), stream=stream0)
        del primals_178
        buf21 = buf15; del buf15  # reuse
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_2, hidden_states_3], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf22, buf16, buf17, buf20, primals_181, primals_182, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_6], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf22, primals_185, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf26 = buf23; del buf23  # reuse
        buf27 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf28 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf30 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [result_6, mul_2, result_7, add, output_tensor, group_norm_2], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mul_native_group_norm_4.run(buf26, buf6, primals_184, buf25, buf27, buf28, buf30, 128, 16384, grid=grid(128), stream=stream0)
        del primals_184
        buf31 = reinterpret_tensor(buf25, (4, 512, 1024), (524288, 1024, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_5.run(buf26, buf27, buf28, primals_187, primals_188, buf31, 2097152, grid=grid(2097152), stream=stream0)
        del primals_188
        buf32 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [result_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (4, 1024, 512), (524288, 1, 1024), 0), reinterpret_tensor(primals_189, (4, 512, 512), (0, 1, 512), 0), out=buf32)
        buf33 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf31, buf33, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf34 = empty_strided_cuda((4096, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_191, (512, 4), (1, 512), 0), out=buf34)
        buf35 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_192, (4, 512), (1, 4), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [result_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (4, 1024, 512), (524288, 1, 1024), 0), reinterpret_tensor(primals_193, (4, 512, 512), (0, 1, 512), 0), out=buf36)
        buf37 = empty_strided_cuda((4096, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_195, (512, 4), (1, 512), 0), out=buf37)
        buf38 = empty_strided_cuda((4096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf37, reinterpret_tensor(primals_196, (4, 512), (1, 4), 0), out=buf38)
        buf39 = empty_strided_cuda((4, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [result_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (4, 1024, 512), (524288, 1, 1024), 0), reinterpret_tensor(primals_197, (4, 512, 512), (0, 1, 512), 0), out=buf39)
        buf40 = empty_strided_cuda((4096, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_199, (512, 4), (1, 512), 0), out=buf40)
        buf41 = reinterpret_tensor(buf31, (4096, 512), (512, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_200, (4, 512), (1, 4), 0), out=buf41)
        buf42 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [result_9, mul_3, result_10], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_7.run(buf42, primals_190, buf35, 2097152, grid=grid(2097152), stream=stream0)
        del primals_190
        buf43 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [result_12, mul_4, result_13], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_7.run(buf43, primals_194, buf38, 2097152, grid=grid(2097152), stream=stream0)
        del primals_194
        buf44 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [result_15, mul_5, result_16], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_7.run(buf44, primals_198, buf41, 2097152, grid=grid(2097152), stream=stream0)
        del primals_198
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf45 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf42, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), reinterpret_tensor(buf43, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), reinterpret_tensor(buf44, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), None, True)
        buf46 = buf45[0]
        buf47 = buf45[1]
        buf48 = buf45[2]
        buf49 = buf45[3]
        del buf45
        buf50 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [result_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf46, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_201, (512, 512), (1, 512), 0), out=buf50)
        buf51 = empty_strided_cuda((4096, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_203, (512, 4), (1, 512), 0), out=buf51)
        buf52 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_204, (4, 512), (1, 4), 0), out=buf52)
        buf53 = reinterpret_tensor(buf35, (4, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_div_8.run(buf50, primals_202, buf52, buf26, buf53, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        buf54 = reinterpret_tensor(buf28, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf28  # reuse
        buf55 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf57 = reinterpret_tensor(buf55, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf57, buf53, buf54, 128, 16384, grid=grid(128), stream=stream0)
        buf58 = empty_strided_cuda((4, 512, 32, 32), (524288, 1024, 32, 1), torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14, hidden_states_15], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf59, buf53, buf54, buf57, primals_205, primals_206, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_21], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf59, primals_209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf63 = buf60; del buf60  # reuse
        buf64 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf65 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf67 = reinterpret_tensor(buf65, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [result_21, mul_7, result_22, hidden_states_16], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf63, buf67, primals_208, buf62, buf64, 128, 16384, grid=grid(128), stream=stream0)
        del primals_208
        buf68 = buf62; del buf62  # reuse
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf69, buf63, buf64, buf67, primals_211, primals_212, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_24], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf69, primals_215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf73 = empty_strided_cuda((4, 512, 32, 32), (524288, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_9], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_10.run(primals_217, buf73, 2097152, grid=grid(2097152), stream=stream0)
        del primals_217
        # Topologically Sorted Source Nodes: [result_27], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf73, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf77 = buf70; del buf70  # reuse
        buf78 = empty_strided_cuda((4, 512, 32, 32), (524288, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_12, hidden_states_13, result_24, mul_8, result_25, add_2, output_tensor_1, mul_10, result_28, sample_1, hidden_states_19], Original ATen: [aten.add, aten.div, aten.convolution, aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_mul_11.run(buf77, buf50, primals_202, buf52, buf26, primals_214, buf72, buf74, buf76, buf78, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del buf50
        del primals_202
        del primals_214
        buf79 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf80 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf82 = reinterpret_tensor(buf80, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf82, buf78, buf79, 128, 16384, grid=grid(128), stream=stream0)
        buf83 = buf76; del buf76  # reuse
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf84, buf78, buf79, buf82, primals_11, primals_12, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_30], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf84, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf88 = buf85; del buf85  # reuse
        buf89 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf90 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf92 = reinterpret_tensor(buf90, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [result_30, mul_11, result_31, hidden_states_21], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf88, buf92, primals_14, buf87, buf89, 128, 16384, grid=grid(128), stream=stream0)
        del primals_14
        buf93 = buf87; del buf87  # reuse
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_22], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf94, buf88, buf89, buf92, primals_17, primals_18, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_33], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf94, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf98 = buf74; del buf74  # reuse
        buf99 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf100 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf102 = reinterpret_tensor(buf100, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [result_33, mul_12, result_34, add_4, output_tensor_2, hidden_states_24], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_12.run(buf102, buf77, buf95, primals_20, buf97, buf98, buf99, 128, 16384, grid=grid(128), stream=stream0)
        buf103 = buf72; del buf72  # reuse
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf104, buf98, buf99, buf102, primals_23, primals_24, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_36], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf104, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf108 = buf105; del buf105  # reuse
        buf109 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf110 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf112 = reinterpret_tensor(buf110, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [result_36, mul_13, result_37, hidden_states_26], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf108, buf112, primals_26, buf107, buf109, 128, 16384, grid=grid(128), stream=stream0)
        del primals_26
        buf113 = buf107; del buf107  # reuse
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf114, buf108, buf109, buf112, primals_29, primals_30, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_39], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf114, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf118 = buf77; del buf77  # reuse
        buf119 = reinterpret_tensor(buf52, (4, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf52  # reuse
        buf120 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf121 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf123 = reinterpret_tensor(buf121, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [result_33, mul_12, result_34, add_4, output_tensor_2, result_39, mul_14, result_40, add_5, output_tensor_3, hidden_states_29], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_13.run(buf118, buf123, buf95, primals_20, buf97, buf115, primals_32, buf117, buf119, buf120, 128, 16384, grid=grid(128), stream=stream0)
        del buf115
        del buf117
        del buf95
        del primals_20
        del primals_32
        buf124 = buf97; del buf97  # reuse
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29, hidden_states_30], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf125, buf119, buf120, buf123, primals_35, primals_36, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_42], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf125, primals_39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf129 = buf126; del buf126  # reuse
        buf130 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf131 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf133 = reinterpret_tensor(buf131, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [result_42, mul_15, result_43, hidden_states_31], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_2.run(buf129, buf133, primals_38, buf128, buf130, 128, 16384, grid=grid(128), stream=stream0)
        del primals_38
        buf134 = buf128; del buf128  # reuse
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_3.run(buf135, buf129, buf130, buf133, primals_41, primals_42, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [result_45], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 512, 32, 32), (524288, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_35], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf135, primals_45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 4, 32, 32), (4096, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 512, 32, 32), (524288, 1024, 32, 1))
        buf139 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hidden_states_34], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_14.run(buf139, 64, grid=grid(64), stream=stream0)
        buf140 = empty_strided_cuda((4, 512, 64, 64), (2097152, 1, 32768, 512), torch.float32)
        # Topologically Sorted Source Nodes: [result_45, mul_16, result_46, add_6, output_tensor_4, hidden_states_34], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_15.run(buf139, buf118, buf136, primals_44, buf138, buf140, 8388608, grid=grid(8388608), stream=stream0)
        del buf118
        del buf136
        del buf138
        del primals_44
        buf141 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [result_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(primals_47, buf141, 262144, 9, grid=grid(262144, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [result_48], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf140, buf141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf143 = empty_strided_cuda((4, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_17.run(primals_49, buf143, 2048, 9, grid=grid(2048, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf140, buf143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 4, 64, 64), (16384, 1, 256, 4))
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf146 = empty_strided_cuda((4, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_18], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_18.run(primals_221, buf146, 4194304, grid=grid(4194304), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [result_51], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf146, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf150 = buf147; del buf147  # reuse
        buf151 = empty_strided_cuda((4, 512, 64, 64), (2097152, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [result_48, mul_17, result_49, mul_19, result_52, sample_2, hidden_states_35], Original ATen: [aten.convolution, aten.mul, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_mul_19.run(buf150, buf142, primals_48, buf145, buf149, buf151, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del primals_48
        buf152 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf153 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf155 = reinterpret_tensor(buf153, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_20.run(buf155, buf151, buf152, 128, 65536, grid=grid(128), stream=stream0)
        buf156 = buf149; del buf149  # reuse
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf157, buf151, buf152, buf155, primals_51, primals_52, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_54], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf157, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf161 = buf158; del buf158  # reuse
        buf162 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf163 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf165 = reinterpret_tensor(buf163, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [result_54, mul_20, result_55, hidden_states_37], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_22.run(buf161, buf165, primals_54, buf160, buf162, 128, 65536, grid=grid(128), stream=stream0)
        del primals_54
        buf166 = buf160; del buf160  # reuse
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf167, buf161, buf162, buf165, primals_57, primals_58, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_57], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf167, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf171 = reinterpret_tensor(buf145, (4, 512, 64, 64), (2097152, 4096, 64, 1), 0); del buf145  # reuse
        buf172 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf173 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf175 = reinterpret_tensor(buf173, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [result_57, mul_21, result_58, add_8, output_tensor_5, hidden_states_40], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_23.run(buf175, buf150, buf168, primals_60, buf170, buf171, buf172, 128, 65536, grid=grid(128), stream=stream0)
        buf176 = reinterpret_tensor(buf142, (4, 512, 64, 64), (2097152, 4096, 64, 1), 0); del buf142  # reuse
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf177, buf171, buf172, buf175, primals_63, primals_64, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_60], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf177, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf181 = buf178; del buf178  # reuse
        buf182 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf183 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf185 = reinterpret_tensor(buf183, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [result_60, mul_22, result_61, hidden_states_42], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_22.run(buf181, buf185, primals_66, buf180, buf182, 128, 65536, grid=grid(128), stream=stream0)
        del primals_66
        buf186 = buf180; del buf180  # reuse
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_43], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf187, buf181, buf182, buf185, primals_69, primals_70, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_63], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf187, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf191 = buf150; del buf150  # reuse
        buf192 = empty_strided_cuda((4, 512, 64, 64), (2097152, 4096, 64, 1), torch.float32)
        buf193 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf194 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf196 = reinterpret_tensor(buf194, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [result_57, mul_21, result_58, add_8, output_tensor_5, result_63, mul_23, result_64, add_9, output_tensor_6, hidden_states_45], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_24.run(buf191, buf196, buf168, primals_60, buf170, buf188, primals_72, buf190, buf192, buf193, 128, 65536, grid=grid(128), stream=stream0)
        del buf168
        del buf170
        del buf188
        del primals_60
        del primals_72
        buf197 = buf190; del buf190  # reuse
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf198, buf192, buf193, buf196, primals_75, primals_76, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_66], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf198, primals_79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf202 = buf199; del buf199  # reuse
        buf203 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf204 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf206 = reinterpret_tensor(buf204, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [result_66, mul_24, result_67, hidden_states_47], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_22.run(buf202, buf206, primals_78, buf201, buf203, 128, 65536, grid=grid(128), stream=stream0)
        del primals_78
        buf207 = buf201; del buf201  # reuse
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_21.run(buf208, buf202, buf203, buf206, primals_81, primals_82, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [result_69], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf208, primals_85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 4, 64, 64), (16384, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 512, 64, 64), (2097152, 4096, 64, 1))
        buf212 = empty_strided_cuda((128, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_25.run(buf212, 128, grid=grid(128), stream=stream0)
        buf213 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [result_69, mul_25, result_70, add_10, output_tensor_7, hidden_states_50], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_26.run(buf212, buf191, buf209, primals_84, buf211, buf213, 33554432, grid=grid(33554432), stream=stream0)
        del buf191
        del buf209
        del primals_84
        buf214 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [result_72], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_16.run(primals_87, buf214, 262144, 9, grid=grid(262144, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [result_72], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf213, buf214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 512, 128, 128), (8388608, 1, 65536, 512))
        del buf214
        buf216 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_17.run(primals_89, buf216, 2048, 9, grid=grid(2048, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf213, buf216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 4, 128, 128), (65536, 1, 512, 4))
        del buf216
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 512, 128, 128), (8388608, 1, 65536, 512))
        buf219 = reinterpret_tensor(buf211, (4, 128, 128, 128), (2097152, 16384, 128, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [mul_27], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_27.run(primals_225, buf219, 8388608, grid=grid(8388608), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [result_75], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 512, 128, 128), (8388608, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf219, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 512, 128, 128), (8388608, 16384, 128, 1))
        buf223 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [result_72, mul_26, result_73, mul_28, result_76, sample_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_28.run(buf223, primals_88, buf218, buf220, buf222, 65536, 512, grid=grid(65536, 512), stream=stream0)
        del primals_88
        buf224 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf225 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf227 = reinterpret_tensor(buf225, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_29.run(buf227, buf223, buf224, 128, 262144, grid=grid(128), stream=stream0)
        buf229 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_30.run(buf223, buf224, buf227, primals_91, primals_92, buf229, 65536, 512, grid=grid(65536, 512), stream=stream0)
        # Topologically Sorted Source Nodes: [result_78], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf229, primals_95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        buf233 = buf230; del buf230  # reuse
        buf234 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf235 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf237 = reinterpret_tensor(buf235, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [result_78, mul_29, result_79, hidden_states_53], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_31.run(buf233, buf237, primals_94, buf232, buf234, 128, 131072, grid=grid(128), stream=stream0)
        del primals_94
        buf238 = buf232; del buf232  # reuse
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53, hidden_states_54], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_32.run(buf239, buf233, buf234, buf237, primals_97, primals_98, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [result_81], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf239, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [result_84], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf223, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        # Topologically Sorted Source Nodes: [conv2d_74], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf223, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 4, 128, 128), (65536, 1, 512, 4))
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 256, 128, 128), (4194304, 1, 32768, 256))
        buf246 = buf240; del buf240  # reuse
        buf247 = empty_strided_cuda((4, 256, 128, 128), (4194304, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [result_81, mul_30, result_82, result_84, mul_31, result_85, add_12, output_tensor_8, hidden_states_56], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_mul_33.run(buf246, buf243, primals_104, buf245, primals_100, buf242, buf247, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        del primals_100
        del primals_104
        buf248 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf249 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf251 = reinterpret_tensor(buf249, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_34.run(buf251, buf247, buf248, 128, 131072, grid=grid(128), stream=stream0)
        buf252 = reinterpret_tensor(buf245, (4, 256, 128, 128), (4194304, 16384, 128, 1), 0); del buf245  # reuse
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_32.run(buf253, buf247, buf248, buf251, primals_107, primals_108, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [result_87], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_77], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf253, primals_111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        buf257 = buf254; del buf254  # reuse
        buf258 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf259 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf261 = reinterpret_tensor(buf259, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [result_87, mul_32, result_88, hidden_states_58], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_31.run(buf257, buf261, primals_110, buf256, buf258, 128, 131072, grid=grid(128), stream=stream0)
        del primals_110
        buf262 = buf256; del buf256  # reuse
        buf263 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_32.run(buf263, buf257, buf258, buf261, primals_113, primals_114, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [result_90], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_80], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf263, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        buf267 = reinterpret_tensor(buf243, (4, 256, 128, 128), (4194304, 16384, 128, 1), 0); del buf243  # reuse
        buf268 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf269 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf271 = reinterpret_tensor(buf269, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [result_90, mul_33, result_91, add_13, output_tensor_9, hidden_states_61], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_35.run(buf271, buf246, buf264, primals_116, buf266, buf267, buf268, 128, 131072, grid=grid(128), stream=stream0)
        buf272 = buf242; del buf242  # reuse
        buf273 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_32.run(buf273, buf267, buf268, buf271, primals_119, primals_120, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [result_93], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_83], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf273, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        buf277 = buf274; del buf274  # reuse
        buf278 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf279 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf281 = reinterpret_tensor(buf279, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [result_93, mul_34, result_94, hidden_states_63], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_31.run(buf277, buf281, primals_122, buf276, buf278, 128, 131072, grid=grid(128), stream=stream0)
        del primals_122
        buf282 = buf276; del buf276  # reuse
        buf283 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63, hidden_states_64], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_32.run(buf283, buf277, buf278, buf281, primals_125, primals_126, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [result_96], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_86], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf283, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 4, 128, 128), (65536, 16384, 128, 1))
        # Topologically Sorted Source Nodes: [conv2d_87], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 256, 128, 128), (4194304, 16384, 128, 1))
        buf287 = empty_strided_cuda((256, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [hidden_states_66], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_36.run(buf287, 256, grid=grid(256), stream=stream0)
        buf289 = empty_strided_cuda((4, 256, 256, 256), (16777216, 1, 65536, 256), torch.float32)
        # Topologically Sorted Source Nodes: [result_90, mul_33, result_91, add_13, output_tensor_9, result_96, mul_35, result_97, add_14, output_tensor_10, hidden_states_66], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten._unsafe_index, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_clone_convolution_div_mul_37.run(buf287, buf246, buf264, primals_116, buf266, buf284, primals_128, buf286, buf289, 1024, 65536, grid=grid(1024, 65536), stream=stream0)
        del buf246
        del buf264
        del buf266
        del buf284
        del buf286
        del primals_116
        del primals_128
        buf290 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [result_99], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(primals_131, buf290, 65536, 9, grid=grid(65536, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [result_99], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf289, buf290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 256, 256, 256), (16777216, 1, 65536, 256))
        del buf290
        buf292 = empty_strided_cuda((4, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(primals_133, buf292, 1024, 9, grid=grid(1024, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf289, buf292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 4, 256, 256), (262144, 1, 1024, 4))
        del buf292
        # Topologically Sorted Source Nodes: [conv2d_90], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 256, 256, 256), (16777216, 1, 65536, 256))
        buf295 = reinterpret_tensor(buf220, (4, 128, 256, 256), (8388608, 65536, 256, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [mul_37], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_40.run(primals_229, buf295, 33554432, grid=grid(33554432), stream=stream0)
        del primals_229
        # Topologically Sorted Source Nodes: [result_102], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 256, 256, 256), (16777216, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf295, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_93], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 256, 256, 256), (16777216, 65536, 256, 1))
        buf299 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [result_99, mul_36, result_100, mul_38, result_103, sample_4], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_41.run(buf299, primals_132, buf294, buf296, buf298, 262144, 256, grid=grid(262144, 256), stream=stream0)
        del buf294
        del buf296
        del primals_132
        buf300 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf301 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf303 = reinterpret_tensor(buf301, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_42.run(buf303, buf299, buf300, 128, 524288, grid=grid(128), stream=stream0)
        buf305 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_43.run(buf299, buf300, buf303, primals_135, primals_136, buf305, 262144, 256, grid=grid(262144, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [result_105], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_95], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf305, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_96], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        buf309 = buf306; del buf306  # reuse
        buf310 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf311 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf313 = reinterpret_tensor(buf311, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [result_105, mul_39, result_106, hidden_states_69], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_44.run(buf309, buf313, primals_138, buf308, buf310, 128, 262144, grid=grid(128), stream=stream0)
        del primals_138
        buf314 = buf308; del buf308  # reuse
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69, hidden_states_70], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf315, buf309, buf310, buf313, primals_141, primals_142, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_108], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_98], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf315, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_99], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [result_111], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf299, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        # Topologically Sorted Source Nodes: [conv2d_101], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf299, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 4, 256, 256), (262144, 1, 1024, 4))
        # Topologically Sorted Source Nodes: [conv2d_102], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (4, 128, 256, 256), (8388608, 1, 32768, 128))
        buf322 = buf316; del buf316  # reuse
        buf323 = reinterpret_tensor(buf218, (4, 128, 256, 256), (8388608, 65536, 256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [result_108, mul_40, result_109, result_111, mul_41, result_112, add_16, output_tensor_11, hidden_states_72], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_mul_46.run(buf322, buf319, primals_148, buf321, primals_144, buf318, buf323, 512, 65536, grid=grid(512, 65536), stream=stream0)
        del primals_144
        del primals_148
        buf324 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf325 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf327 = reinterpret_tensor(buf325, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_72], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_47.run(buf327, buf323, buf324, 128, 262144, grid=grid(128), stream=stream0)
        buf328 = reinterpret_tensor(buf321, (4, 128, 256, 256), (8388608, 65536, 256, 1), 0); del buf321  # reuse
        buf329 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_73], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf329, buf323, buf324, buf327, primals_151, primals_152, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_114], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_104], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf329, primals_155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_105], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        buf333 = buf330; del buf330  # reuse
        buf334 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf335 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf337 = reinterpret_tensor(buf335, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [result_114, mul_42, result_115, hidden_states_74], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_44.run(buf333, buf337, primals_154, buf332, buf334, 128, 262144, grid=grid(128), stream=stream0)
        del primals_154
        buf338 = buf332; del buf332  # reuse
        buf339 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_75], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf339, buf333, buf334, buf337, primals_157, primals_158, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_117], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_107], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf339, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_108], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        buf343 = reinterpret_tensor(buf319, (4, 128, 256, 256), (8388608, 65536, 256, 1), 0); del buf319  # reuse
        buf344 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf345 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf347 = reinterpret_tensor(buf345, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [result_117, mul_43, result_118, add_17, output_tensor_12, hidden_states_77], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_48.run(buf347, buf322, buf340, primals_160, buf342, buf343, buf344, 128, 262144, grid=grid(128), stream=stream0)
        buf348 = buf318; del buf318  # reuse
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf349, buf343, buf344, buf347, primals_163, primals_164, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_120], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_110], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf349, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_111], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        buf353 = buf350; del buf350  # reuse
        buf354 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf355 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf357 = reinterpret_tensor(buf355, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [result_120, mul_44, result_121, hidden_states_79], Original ATen: [aten.convolution, aten.mul, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mul_native_group_norm_44.run(buf353, buf357, primals_166, buf352, buf354, 128, 262144, grid=grid(128), stream=stream0)
        del primals_166
        buf358 = buf352; del buf352  # reuse
        buf359 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79, hidden_states_80], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf359, buf353, buf354, buf357, primals_169, primals_170, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_123], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_113], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf359, primals_173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_114], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 128, 256, 256), (8388608, 65536, 256, 1))
        buf363 = buf322; del buf322  # reuse
        buf364 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf365 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf367 = reinterpret_tensor(buf365, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [result_117, mul_43, result_118, add_17, output_tensor_12, result_123, mul_45, result_124, add_18, output_tensor_13, sample_5], Original ATen: [aten.convolution, aten.mul, aten.add, aten.div, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_mul_native_group_norm_49.run(buf363, buf367, buf340, primals_160, buf342, buf360, primals_172, buf362, buf364, 128, 262144, grid=grid(128), stream=stream0)
        del buf340
        del buf342
        del buf360
        del primals_160
        del primals_172
        buf368 = buf362; del buf362  # reuse
        buf369 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [sample_5, sample_6], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_45.run(buf369, buf363, buf364, buf367, primals_233, primals_234, 33554432, grid=grid(33554432), stream=stream0)
        # Topologically Sorted Source Nodes: [result_126], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, primals_235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 3, 256, 256), (196608, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_116], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf369, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 4, 256, 256), (262144, 65536, 256, 1))
        # Topologically Sorted Source Nodes: [conv2d_117], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 3, 256, 256), (196608, 65536, 256, 1))
        buf373 = empty_strided_cuda((4, 3, 256, 256), (196608, 65536, 256, 1), torch.float32)
        buf374 = empty_strided_cuda((4, 3, 256, 256), (196608, 65536, 256, 1), torch.bool)
        # Topologically Sorted Source Nodes: [result_126, mul_46, result_127, x_decoded], Original ATen: [aten.convolution, aten.mul, aten.add, aten.clamp, aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_convolution_ge_le_logical_and_mul_50.run(buf370, primals_236, buf372, buf373, buf374, 786432, grid=grid(786432), stream=stream0)
        del buf370
        del buf372
        del primals_236
    return (buf373, primals_5, primals_7, primals_9, primals_10, primals_11, primals_12, primals_13, primals_15, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_73, primals_74, primals_75, primals_76, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_173, primals_174, primals_175, primals_176, primals_177, primals_179, primals_180, primals_181, primals_182, primals_183, primals_185, primals_186, primals_187, primals_189, primals_193, primals_197, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, buf0, buf2, buf4, buf6, buf7, buf10, buf12, buf14, buf16, buf17, buf20, buf22, buf24, reinterpret_tensor(buf26, (4, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(buf27, (4, 32), (32, 1), 0), reinterpret_tensor(buf30, (4, 32), (32, 1), 0), reinterpret_tensor(buf33, (4096, 512), (512, 1), 0), buf34, buf37, buf40, reinterpret_tensor(buf42, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), reinterpret_tensor(buf43, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), reinterpret_tensor(buf44, (4, 1, 1024, 512), (524288, 512, 512, 1), 0), buf46, buf47, buf48, buf49, buf51, buf53, buf54, buf57, buf59, buf61, buf63, buf64, buf67, buf69, buf71, buf73, buf75, buf78, buf79, buf82, buf84, buf86, buf88, buf89, buf92, buf94, buf96, buf98, buf99, buf102, buf104, buf106, buf108, buf109, buf112, buf114, buf116, buf119, buf120, buf123, buf125, buf127, buf129, buf130, buf133, buf135, buf137, buf139, buf140, buf144, buf146, buf148, buf151, buf152, buf155, buf157, buf159, buf161, buf162, buf165, buf167, buf169, buf171, buf172, buf175, buf177, buf179, buf181, buf182, buf185, buf187, buf189, buf192, buf193, buf196, buf198, buf200, buf202, buf203, buf206, buf208, buf210, buf212, buf213, buf217, buf219, buf221, buf223, buf224, buf227, buf229, buf231, buf233, buf234, buf237, buf239, buf241, buf244, buf247, buf248, buf251, buf253, buf255, buf257, buf258, buf261, buf263, buf265, buf267, buf268, buf271, buf273, buf275, buf277, buf278, buf281, buf283, buf285, buf287, buf289, buf293, buf295, buf297, buf299, buf300, buf303, buf305, buf307, buf309, buf310, buf313, buf315, buf317, buf320, buf323, buf324, buf327, buf329, buf331, buf333, buf334, buf337, buf339, buf341, buf343, buf344, buf347, buf349, buf351, buf353, buf354, buf357, buf359, buf361, buf363, buf364, buf367, buf369, buf371, buf374, primals_204, primals_203, primals_201, primals_200, primals_199, primals_196, primals_195, primals_192, primals_191, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((4, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((4, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((4, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((4, 128, 256, 256), (8388608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((3, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((3, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
