# AOT ID: ['2_backward']
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


# kernel path: /tmp/torchinductor_elicer/cz/ccznv3isd7uizj7zfquk5vo2lvvd25qm6nmazwesjkojgtx7em7s.py
# Topologically Sorted Source Nodes: [target__2], Original ATen: [aten._to_copy, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   target__2 => full_default_9
# Graph fragment:
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 1], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_49,), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_12, %full_default_9), kwargs = {})
#   %mul_112 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %tangents_1), kwargs = {})
triton_poi_fused__to_copy_mul_sigmoid_sub_0 = async_compile.triton('triton_poi_fused__to_copy_mul_sigmoid_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_sigmoid_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp4 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/g3/cg3gq6g6ntvpklgkm6yvvjltolbw4427ylluy7w6t3mfk5e2kory.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_112, [0], True), kwargs = {})
triton_poi_fused_sum_1 = async_compile.triton('triton_poi_fused_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ip/cipa3yblph7jf355wmpmhholan2dc3eunsss7pvrkh2bk3td523g.py
# Topologically Sorted Source Nodes: [target_], Original ATen: [aten.div, aten.sigmoid, aten._to_copy, aten.sub, aten.mul, aten.convolution_backward]
# Source node to ATen node mapping:
#   target_ => full_default_3
# Graph fragment:
#   %div_19 : [num_users=2] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_33, 9), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%squeeze_13,), kwargs = {})
#   %full_default_3 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 3, 3], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_13, %full_default_3), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %div_19), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%unsqueeze_22, [0, 2, 3]), kwargs = {})
triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2 = async_compile.triton('triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    r1 = rindex // 9
    tmp0 = tl.load(in_out_ptr0 + (r2), rmask, other=0.0)
    tmp4 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 - tmp2
    tmp5 = 0.1111111111111111
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tl.store(in_out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp7, rmask)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/m4/cm4x2jtkodpvviyjecvyakzkxrbjxgwr2fpluu55y3oxjtmrqpae.py
# Topologically Sorted Source Nodes: [u_10, sigma_5, mv_15, v_10], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum, aten.mv]
# Source node to ATen node mapping:
#   mv_15 => mul_104, sum_31
#   sigma_5 => mul_107, sum_36
#   u_10 => clamp_min_15, div_17, pow_23, pow_24, sum_34
#   v_10 => div_16
# Graph fragment:
#   %add_93 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_7, %mm_2), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_33, 2.0), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_23, [0], True), kwargs = {})
#   %pow_24 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_34, 0.5), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_24, 1e-12), kwargs = {})
#   %div_17 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_33, %expand_29), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_17, %sum_33), kwargs = {})
#   %sum_36 : [num_users=3] = call_function[target=torch.ops.aten.sum.default](args = (%mul_107,), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_178, %sum_36), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_21, %sum_36), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_93,), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_3, %div_22), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_93, %sum_36), kwargs = {})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_115,), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_123, %primals_179), kwargs = {})
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_104, [1]), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_31, %expand_27), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_200, %div_16), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_23, %mul_117), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_179, %div_17), kwargs = {})
triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3 = async_compile.triton('triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3', 'mutated_arg_names': ['in_out_ptr0', 'in_ptr3', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_out_ptr0 + (r0), None)
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.load(in_ptr4 + (0))
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp29 = tl.broadcast_to(tmp5, [1])
    tmp2 = tmp0 + tmp1
    tmp3 = -tmp2
    tmp7 = tmp6 * tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = 1e-12
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tmp6 / tmp10
    tmp12 = tmp11 * tmp6
    tmp13 = tmp4 / tmp12
    tmp14 = tmp13 / tmp12
    tmp15 = tmp3 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp2 / tmp12
    tmp20 = tmp18 * tmp11
    tmp23 = tmp4 * tmp22
    tmp26 = tmp23 / tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp19 + tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = libdevice.sqrt(tmp30)
    tmp32 = triton_helpers.maximum(tmp31, tmp9)
    tmp33 = tmp29 / tmp32
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp28, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/4w/c4w64nhcoilbegcvee5xmbuf7s5d4n3c4vckx7q4esexandmidre.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.leaky_relu_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_2, 0), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_1, 0.2), kwargs = {})
#   %where_3 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mm_1, %mul_118), kwargs = {})
triton_poi_fused_leaky_relu_backward_4 = async_compile.triton('triton_poi_fused_leaky_relu_backward_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_backward_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.where(tmp2, tmp3, tmp5)
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/eh/cehjvhr36zyd4tkq3lv5vdv45gb4mct6lmajghykc33thwfmkexo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_137 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_5, %primals_153), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, 768), kwargs = {})
#   %sum_49 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_137, [1], True), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %mul_79), kwargs = {})
#   %sum_50 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_139, [1], True), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %sum_50), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_138, %sum_49), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_45, %mul_140), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, %sub_46), kwargs = {})
triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_native_layer_norm_backward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    xnumel = 2
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp19, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5j/c5jwyums6uhegzsfyoksf4wd4ypg7fgpvfmx536hniytwud3m2ii.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_103, [-1, -1, -1, -1]), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd_3, 0.2), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %constant_pad_nd_3, %mul_125), kwargs = {})
triton_poi_fused_constant_pad_nd_leaky_relu_backward_6 = async_compile.triton('triton_poi_fused_constant_pad_nd_leaky_relu_backward_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_leaky_relu_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_leaky_relu_backward_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = ((xindex // 1792) % 7)
    x1 = ((xindex // 256) % 7)
    x3 = xindex // 12544
    x5 = (xindex % 1792)
    tmp0 = tl.load(in_ptr0 + (x4), xmask).to(tl.int1)
    tmp1 = 1 + x2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 9, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = 1 + x1
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp3 & tmp5
    tmp10 = tmp9 & tmp7
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr1 + (2560 + x5 + 2304*x2 + 20736*x3), tmp11 & xmask, other=0.0)
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp0, tmp12, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/zd/czdocnxrdqbu3dhsnvjnjfm523pah4nfjaxjwb2sisqo4elfjkxb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_3, [0], True), kwargs = {})
triton_poi_fused_sum_7 = async_compile.triton('triton_poi_fused_sum_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/az/cazdopuju4fnnkkuhmkyj7rmx67q4sqztkoaefzecnnr2mn4epvw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_4, [0, 2, 3]), kwargs = {})
triton_red_fused_convolution_backward_8 = async_compile.triton('triton_red_fused_convolution_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_backward_8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ms/cmsoveuo6kv4awh47yaec6q7ojrguizh6vy7n2u3kh5ff4pul5xk.py
# Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_1 => clone_2, mul_7, sub_7
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_2, %getitem_3), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_1), kwargs = {})
triton_poi_fused_native_layer_norm_9 = async_compile.triton('triton_poi_fused_native_layer_norm_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex % 38400)
    x5 = xindex // 768
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 50)
    x2 = xindex // 38400
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + 2*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2 + 2*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3t/c3tsef6qwalh3ruyiii6mnokbdlaukywkmqbvx7yb62cnidd47y7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_64 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_136,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1536
    x0 = (xindex % 1536)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xx/cxxaqgphnh6s5r4oqgyuh4mgyhnpto6n4xixzqpqs74vfuhsljbp.py
# Topologically Sorted Source Nodes: [mul_27, sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
# Source node to ATen node mapping:
#   mul_27 => mul_77
#   sigmoid_11 => sigmoid_11
# Graph fragment:
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_216, %view_184), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_184, 1.702), kwargs = {})
#   %sigmoid_11 : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_77,), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_216, %sigmoid_11), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_11), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_11, %sub_47), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %mul_144), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_145, 1.702), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %mul_146), kwargs = {})
triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 307200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 1.702
    tmp3 = tmp1 * tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = tmp0 * tmp1
    tmp7 = 1.0
    tmp8 = tmp7 - tmp4
    tmp9 = tmp4 * tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 * tmp2
    tmp12 = tmp5 + tmp11
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hy/chyf4cl3bapj6xthehng4ownlgbpxekdnj7azpk2oyqhqbvzidwm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_148 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_218, %primals_147), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, 768), kwargs = {})
#   %sum_51 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_148, [2], True), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %mul_75), kwargs = {})
#   %sum_52 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_150, [2], True), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %sum_52), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_149, %sum_51), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_49, %mul_151), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_40, %sub_50), kwargs = {})
#   %add_106 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_136, %mul_152), kwargs = {})
#   %clone_66 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_106,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_native_layer_norm_backward_12 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 100
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = xindex // 2
    x2 = (xindex % 2)
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + (r1 + 768*x2), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = x3
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp13 == tmp14
    tmp17 = 0.0
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp20 = 768.0
    tmp21 = tmp2 * tmp20
    tmp22 = tmp21 - tmp6
    tmp23 = tmp7 * tmp12
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp18 + tmp25
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp26, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp26, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/7o/c7ovho7xnebqdxcisgoogxl4dwrfiq5jmkfxwyttb4tg3antptba.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_73 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_14,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 768) % 3)
    x0 = (xindex % 768)
    x2 = ((xindex // 2304) % 50)
    x3 = xindex // 115200
    x4 = (xindex % 2304)
    tmp3 = tl.load(in_ptr0 + (x0 + 768*x2 + 768*((x0 + 768*x3) // 1536) + 38400*x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 768*x2 + 768*((x0 + 768*x3) // 1536) + 38400*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0 + 768*x2 + 768*((x0 + 768*x3) // 1536) + 38400*x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x4 + 2304*x3 + 4608*x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ws/cwsjeasdjonw6vlhjmvw6fit56qx3dfmkirvfxjkw2azmogrhrhi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_154 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_229, %primals_141), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, 768), kwargs = {})
#   %sum_53 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_154, [2], True), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %mul_73), kwargs = {})
#   %sum_54 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_156, [2], True), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %sum_54), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_155, %sum_53), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_52, %mul_157), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_41, %sub_53), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %mul_158), kwargs = {})
#   %clone_75 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_109,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_native_layer_norm_backward_14 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_backward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_backward_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 100
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + 768*x0), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp21, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp21, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sl/cslq2adcfjncojylzx3br6p24xnilprj2ri7h2wccjui2uozyths.py
# Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.clone]
# Source node to ATen node mapping:
#   ret_17 => clone_34, mul_55, sub_23
# Graph fragment:
#   %mul_205 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_274, %primals_105), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, 768), kwargs = {})
#   %sum_65 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_205, [2], True), kwargs = {})
#   %clone_34 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_62,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_34, %getitem_67), kwargs = {})
#   %mul_55 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_17), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %mul_55), kwargs = {})
#   %sum_66 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_207, [2], True), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %sum_66), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_206, %sum_65), kwargs = {})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_73, %mul_208), kwargs = {})
#   %div_47 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_17, 768), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_47, %sub_74), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %mul_209), kwargs = {})
#   %add_125 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_124, %permute_173), kwargs = {})
#   %clone_108 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_125,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 100
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 2)
    x3 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + 768*x3 + 38400*x2), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 0.0013020833333333333
    tmp19 = tmp10 * tmp18
    tmp20 = 768.0
    tmp21 = tmp2 * tmp20
    tmp22 = tmp21 - tmp6
    tmp23 = tmp11 * tmp16
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp17 + tmp25
    tmp27 = x3
    tmp28 = tl.full([1], 1, tl.int64)
    tmp29 = tmp27 >= tmp28
    tmp30 = tl.load(in_ptr5 + (r1 + 768*((((-1) + x3) % 49)) + 37632*x2), rmask & tmp29, other=0.0)
    tmp31 = 0.0
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp26 + tmp32
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp33, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/rh/crhvorks47vmc6jw4oelfmve4ufe2i7mpxi7s72wuonw7fpscsvn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_165, %sum_18), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_30, %sum_18), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_6, %div_31), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_126,), kwargs = {})
triton_red_fused_add_div_mul_neg_sum_16 = async_compile.triton('triton_red_fused_add_div_mul_neg_sum_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_neg_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_mul_neg_sum_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (768*(((r1 + 8192*x0) % 9)) + 6912*((r1 + 8192*x0) // 6912) + ((((r1 + 8192*x0) // 9) % 768))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = -tmp2
        tmp7 = tmp4 / tmp6
        tmp8 = tmp7 / tmp6
        tmp9 = tmp3 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/b7/cb7u45l7zmh3gnwy6kx2sdh572bx5d37widt4vlk7anocuevwioo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_165, %sum_18), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_30, %sum_18), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_6, %div_31), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_126,), kwargs = {})
triton_per_fused_add_div_mul_neg_sum_17 = async_compile.triton('triton_per_fused_add_div_mul_neg_sum_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_sum_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_neg_sum_17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 216
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sm/csmuxejscwhbkrbkl5xek7n54nk4giyo2b6x5hlvpj3a5ohsfqfq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %sum_18), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_32, %view_208), kwargs = {})
triton_poi_fused_add_div_18 = async_compile.triton('triton_poi_fused_add_div_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196608
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 768*x2 + 6912*y1), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, YBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, YBLOCK])
    tmp8 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2 + 9*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 / tmp4
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 + tmp11
    tl.store(out_ptr0 + (x2 + 9*y3), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bq/cbqfehtjprih5rfr3lwpqkkhmv6paja3xd3kme5abfwlugxryggc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_341 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_394, %primals_9), kwargs = {})
#   %sum_97 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_341, [2], True), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %mul_7), kwargs = {})
#   %sum_98 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_343, [2], True), kwargs = {})
#   %mul_347 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_247, %primals_7), kwargs = {})
#   %sum_99 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_347, [2], True), kwargs = {})
triton_per_fused_native_layer_norm_backward_19 = async_compile.triton('triton_per_fused_native_layer_norm_backward_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 3, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 100
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 2)
    x3 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + 768*x3 + 38400*x2), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 0.0013020833333333333
    tmp16 = tmp14 * tmp15
    tmp17 = 768.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp6
    tmp20 = tmp7 * tmp12
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp13 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp25, rmask)
    tl.store(out_ptr2 + (x0), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xf/cxfkh65k77yi6gvjjwe5oscowdssmponnnwujcoh4yomnhhhafzy.py
# Topologically Sorted Source Nodes: [x_13, ret], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret => mul_5, sub_6
#   x_13 => add_12
# Graph fragment:
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_6), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_1), kwargs = {})
#   %mul_5 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_347, %mul_5), kwargs = {})
#   %sum_100 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_349, [2], True), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 100
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 50)
    x1 = xindex // 50
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x1 + 1536*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 768*x3), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + 768*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ox/coxsfwdhgh3ocftqpekyfv2n7wmjyv4kovls24tbiufzkf446xd6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convolution_backward_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_395, %div, %primals_4, [0], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]), kwargs = {})
triton_poi_fused_convolution_backward_21 = async_compile.triton('triton_poi_fused_convolution_backward_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 768) % 49)
    x2 = xindex // 37632
    x0 = (xindex % 768)
    x3 = (xindex % 37632)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x1 + 50*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (1536 + x0 + 768*x2 + 1536*x1), xmask)
    tmp6 = tl.load(in_ptr2 + (2 + x2 + 2*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (768 + x3 + 38400*x2), xmask)
    tmp9 = tl.load(in_ptr4 + (768 + x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (1 + x1 + 50*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (1 + x1 + 50*x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0013020833333333333
    tmp2 = tmp0 * tmp1
    tmp4 = 768.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 - tmp15
    tmp17 = tmp2 * tmp16
    tl.store(out_ptr0 + (x4), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hh/chhg4a32ormofuvaz2lwadpikbxpumips5c3aw3ekkgn5u4gtgfb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten._adaptive_avg_pool2d_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_65 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem_166, %device_put_1), kwargs = {})
#   %_adaptive_avg_pool2d_backward : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d_backward.default](args = (%div_65, %add_10), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_backward_div_22 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_backward_div_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_backward_div_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_backward_div_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 50176) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ay/cayzbcvaymdnhfzteipek57awlukj72aajgnljmqyj42pkrrmifj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_4), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_24, %sum_30), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_95,), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_4, %div_25), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_119,), kwargs = {})
triton_red_fused_add_div_mul_neg_sum_23 = async_compile.triton('triton_red_fused_add_div_mul_neg_sum_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_neg_sum_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_mul_neg_sum_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = -tmp2
        tmp7 = tmp4 / tmp6
        tmp8 = tmp7 / tmp6
        tmp9 = tmp3 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/6x/c6xwnpw6a6dxtqwq24dcjbeeq7w32mfis3seumm2ja73jciogwa2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_4), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_24, %sum_30), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_95,), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_4, %div_25), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_119,), kwargs = {})
triton_per_fused_add_div_mul_neg_sum_24 = async_compile.triton('triton_per_fused_add_div_mul_neg_sum_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_sum_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_neg_sum_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ok/cok5putn37xoxatofm2ztzwuk5ssucnjndeer62i47oosww7utbk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_4), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_95, %sum_30), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_203, %div_13), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_26, %mul_121), kwargs = {})
triton_poi_fused_add_div_mul_25 = async_compile.triton('triton_poi_fused_add_div_mul_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 512
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 / tmp4
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/f2/cf2mnucyn464gtwh7wbg3y6xps7fa4w5m34juqdctghxp52ww7dd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_57 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 258, 258, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_26 = async_compile.triton('triton_poi_fused_new_zeros_26', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_26(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 199692)
    x1 = xindex // 199692
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + 199712*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/wy/cwyhnief32schvmgod5fs3eswxpgckqvqtakpjklplhlpo3a4ba4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros, aten.index_put]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_57 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 258, 258, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_57, [%expand, %clamp_max, %clamp_max_1], %permute_249, True), kwargs = {})
triton_poi_fused_index_put_new_zeros_27 = async_compile.triton('triton_poi_fused_index_put_new_zeros_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_27', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_new_zeros_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 196608
    x0 = (xindex % 65536)
    x3 = xindex
    x1 = ((xindex // 65536) % 3)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x3), None)
    tmp20 = tl.load(in_ptr4 + (x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2), "index out of bounds: 0 <= tmp4 < 2")
    tmp7 = tl.full([XBLOCK], 258, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 258), "index out of bounds: 0 <= tmp10 < 258")
    tmp13 = tmp12 + tmp7
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 258), "index out of bounds: 0 <= tmp15 < 258")
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 * tmp20
    tl.atomic_add(out_ptr0 + (x1 + 3*tmp15 + 774*tmp10 + 199712*tmp4), tmp21, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/r3/cr3b562rlpp6pqf2tngqviae7ocxdjext6fkuj6sqmygf2qezgww.py
# Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.sum, aten.add, aten.mul, aten.neg]
# Source node to ATen node mapping:
#   add_2 => add_2
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]), kwargs = {})
#   %sum_101 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%constant_pad_nd_5, [1, 2, 3], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%inductor_random_default, 0.5), kwargs = {})
#   %mul_354 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd_5, %add_2), kwargs = {})
#   %neg_9 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_354,), kwargs = {})
#   %sum_102 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg_9, [1, 2, 3], True), kwargs = {})
triton_per_fused_add_constant_pad_nd_mul_neg_sum_28 = async_compile.triton('triton_per_fused_add_constant_pad_nd_mul_neg_sum_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_constant_pad_nd_mul_neg_sum_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_constant_pad_nd_mul_neg_sum_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 24)
    x2 = xindex // 1536
    x5 = xindex
    tmp16 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = 1 + ((((r3 + 128*x0 + 8192*x1) // 256) % 256))
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 258, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (((r3 + 128*x0) % 256))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (777 + 3*(((r3 + 128*x0) % 256)) + 774*((((r3 + 128*x0 + 8192*x1) // 256) % 256)) + 199712*x2 + ((r3 + 128*x0 + 8192*x1) // 65536)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 0.5
    tmp18 = tmp16 + tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = -tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/vs/cvsjxtnuyu5t5qfjtuxpjduei6ipcehmwgt3u5x3zrdx676lo4xz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]), kwargs = {})
#   %sum_101 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%constant_pad_nd_5, [1, 2, 3], True), kwargs = {})
triton_per_fused_constant_pad_nd_sum_29 = async_compile.triton('triton_per_fused_constant_pad_nd_sum_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_sum_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_sum_29(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/eu/ceudldqz6dgn2efg4pj642y7gjmlgkmctjwd2rohvjkcxrvbou5u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]), kwargs = {})
#   %sum_101 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%constant_pad_nd_5, [1, 2, 3], True), kwargs = {})
triton_per_fused_constant_pad_nd_sum_30 = async_compile.triton('triton_per_fused_constant_pad_nd_sum_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_sum_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_sum_30(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 24*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/mw/cmwqa43bsgh5hn4goy2sowdkbtxlaxmgrdmi2emvthkzbtd2lzyn.py
# Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.div, aten.sum]
# Source node to ATen node mapping:
#   add_2 => add_2
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%inductor_random_default, 0.5), kwargs = {})
#   %mul_354 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd_5, %add_2), kwargs = {})
#   %div_66 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_35, 196608), kwargs = {})
#   %add_168 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_354, %div_66), kwargs = {})
#   %sum_103 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_168, [1], True), kwargs = {})
triton_poi_fused_add_constant_pad_nd_div_mul_sum_31 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_div_mul_sum_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_div_mul_sum_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_div_mul_sum_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x2 = xindex // 65536
    x4 = xindex
    tmp12 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp0 = 1 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 258, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (777 + 3*x0 + 774*x1 + 199712*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = 5.086263020833333e-06
    tmp20 = tmp18 * tmp19
    tmp21 = tmp15 + tmp20
    tmp22 = tl.load(in_ptr0 + (778 + 3*x0 + 774*x1 + 199712*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp14
    tmp24 = tmp23 + tmp20
    tmp25 = tmp21 + tmp24
    tmp26 = tl.load(in_ptr0 + (779 + 3*x0 + 774*x1 + 199712*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 * tmp14
    tmp28 = tmp27 + tmp20
    tmp29 = tmp25 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/on/con6n7ljqha6r7ngvz6z5m6vbkenyqby7ouze257i2of7ltgpadt.py
# Topologically Sorted Source Nodes: [add_2, mul], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_2 => add_2
#   mul => mul
# Graph fragment:
#   %constant_pad_nd_5 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_250, [-1, -1, -1, -1, 0, 0, 0, 0]), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%inductor_random_default, 0.5), kwargs = {})
#   %mul_354 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd_5, %add_2), kwargs = {})
#   %div_66 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_35, 196608), kwargs = {})
#   %add_168 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_354, %div_66), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default_1, 2), kwargs = {})
#   %mul_355 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %mul), kwargs = {})
triton_poi_fused_add_constant_pad_nd_div_mul_32 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_div_mul_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_div_mul_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_div_mul_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 768) % 256)
    x1 = ((xindex // 3) % 256)
    x3 = xindex // 196608
    x4 = (xindex % 768)
    x6 = xindex
    tmp12 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 258, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (777 + x4 + 774*x2 + 199712*x3), tmp10, other=0.0)
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = 5.086263020833333e-06
    tmp20 = tmp18 * tmp19
    tmp21 = tmp15 + tmp20
    tmp23 = 2.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp21 * tmp24
    tl.store(out_ptr0 + (x6), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xq/cxqlphwfdzepjnv2eglzygt43opyh2fip3uxghcf4n6ggt6l4aam.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_67 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_36, 3), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_355, %div_67), kwargs = {})
triton_poi_fused_add_div_33 = async_compile.triton('triton_poi_fused_add_div_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_33(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (3*x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 3*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 3*x1), None, eviction_policy='evict_last')
    tmp3 = -tmp2
    tmp5 = -tmp4
    tmp6 = tmp3 + tmp5
    tmp8 = -tmp7
    tmp9 = tmp6 + tmp8
    tmp10 = tmp1 + tmp9
    tmp11 = 0.3333333333333333
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_6, primals_7, primals_8, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, inductor_random_default_1, inductor_random_default, view, clamp_max, clamp_max_1, index_put, add_10, device_put_1, div, cat, getitem_1, rsqrt, getitem_3, rsqrt_1, view_13, view_14, view_15, getitem_4, getitem_5, getitem_6, getitem_7, mul_9, addmm_2, mul_13, view_28, view_29, view_30, getitem_12, getitem_13, getitem_14, getitem_15, mul_15, addmm_6, mul_19, view_43, view_44, view_45, getitem_20, getitem_21, getitem_22, getitem_23, mul_21, addmm_10, mul_25, view_58, view_59, view_60, getitem_28, getitem_29, getitem_30, getitem_31, mul_27, addmm_14, add_38, getitem_35, rsqrt_9, view_73, view_74, view_75, getitem_36, getitem_37, getitem_38, getitem_39, mul_33, addmm_18, mul_37, view_88, view_89, view_90, getitem_44, getitem_45, getitem_46, getitem_47, mul_39, addmm_22, mul_43, view_103, view_104, view_105, getitem_52, getitem_53, getitem_54, getitem_55, mul_45, addmm_26, mul_49, view_118, view_119, view_120, getitem_60, getitem_61, getitem_62, getitem_63, mul_51, addmm_30, add_62, getitem_67, rsqrt_17, view_133, view_134, view_135, getitem_68, getitem_69, getitem_70, getitem_71, mul_57, addmm_34, mul_61, view_148, view_149, view_150, getitem_76, getitem_77, getitem_78, getitem_79, mul_63, addmm_38, mul_67, view_163, view_164, view_165, getitem_84, getitem_85, getitem_86, getitem_87, mul_69, addmm_42, mul_73, view_178, view_179, view_180, getitem_92, getitem_93, getitem_94, getitem_95, mul_75, addmm_46, mul_79, mm, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, permute_129, gt_4, gt_5, permute_135, div_39, permute_137, permute_138, div_40, permute_139, permute_145, div_41, permute_146, permute_147, div_42, permute_148, permute_154, div_43, permute_155, permute_156, div_44, permute_157, permute_163, div_45, permute_164, permute_165, div_46, permute_166, permute_172, permute_174, permute_175, div_48, permute_176, permute_182, div_49, permute_183, permute_184, div_50, permute_185, permute_191, div_51, permute_192, permute_193, div_52, permute_194, permute_200, div_53, permute_201, permute_202, div_54, permute_203, permute_209, permute_211, permute_212, div_56, permute_213, permute_219, div_57, permute_220, permute_221, div_58, permute_222, permute_228, div_59, permute_229, permute_230, div_60, permute_231, permute_237, div_61, permute_238, permute_239, div_62, permute_240, permute_246, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7 = args
    args.clear()
    assert_size_stride(primals_4, (768, 3, 32, 32), (3072, 1024, 32, 1))
    assert_size_stride(primals_6, (50, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_156, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_160, (256, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(primals_161, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_162, (1, ), (1, ))
    assert_size_stride(primals_165, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_169, (256, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(primals_170, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_171, (1, ), (1, ))
    assert_size_stride(primals_174, (256, 512), (512, 1))
    assert_size_stride(primals_178, (1, 256), (256, 1))
    assert_size_stride(primals_179, (1, ), (1, ))
    assert_size_stride(inductor_random_default_1, (2, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(inductor_random_default, (2, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(view, (2, 1, 1), (1, 1, 1))
    assert_size_stride(clamp_max, (2, 256, 256), (65536, 256, 1))
    assert_size_stride(clamp_max_1, (2, 256, 256), (65536, 256, 1))
    assert_size_stride(index_put, (2, 256, 256), (65536, 256, 1))
    assert_size_stride(add_10, (2, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(device_put_1, (3, 1, 1), (1, 1, 1))
    assert_size_stride(div, (2, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(cat, (2, 50, 768), (38400, 768, 1))
    assert_size_stride(getitem_1, (2, 50, 1), (50, 1, 1))
    assert_size_stride(rsqrt, (2, 50, 1), (50, 1, 1))
    assert_size_stride(getitem_3, (50, 2, 1), (2, 1, 1))
    assert_size_stride(rsqrt_1, (50, 2, 1), (2, 1, 1))
    assert_size_stride(view_13, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_14, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_15, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_4, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_5, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(mul_9, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_2, (100, 3072), (3072, 1))
    assert_size_stride(mul_13, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_28, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_29, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_30, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_12, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_13, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_14, (), ())
    assert_size_stride(getitem_15, (), ())
    assert_size_stride(mul_15, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_6, (100, 3072), (3072, 1))
    assert_size_stride(mul_19, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_43, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_44, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_45, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_20, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_21, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_22, (), ())
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(mul_21, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_10, (100, 3072), (3072, 1))
    assert_size_stride(mul_25, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_58, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_59, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_60, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_28, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_29, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(getitem_31, (), ())
    assert_size_stride(mul_27, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_14, (100, 3072), (3072, 1))
    assert_size_stride(add_38, (50, 2, 768), (768, 38400, 1))
    assert_size_stride(getitem_35, (50, 2, 1), (2, 1, 1))
    assert_size_stride(rsqrt_9, (50, 2, 1), (2, 1, 1))
    assert_size_stride(view_73, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_74, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_75, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_36, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_37, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_38, (), ())
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(mul_33, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_18, (100, 3072), (3072, 1))
    assert_size_stride(mul_37, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_88, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_89, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_90, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_44, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_45, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_46, (), ())
    assert_size_stride(getitem_47, (), ())
    assert_size_stride(mul_39, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_22, (100, 3072), (3072, 1))
    assert_size_stride(mul_43, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_103, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_104, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_105, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_52, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_53, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_54, (), ())
    assert_size_stride(getitem_55, (), ())
    assert_size_stride(mul_45, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_26, (100, 3072), (3072, 1))
    assert_size_stride(mul_49, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_118, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_119, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_120, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_60, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_61, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(mul_51, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_30, (100, 3072), (3072, 1))
    assert_size_stride(add_62, (50, 2, 768), (768, 38400, 1))
    assert_size_stride(getitem_67, (50, 2, 1), (2, 1, 1))
    assert_size_stride(rsqrt_17, (50, 2, 1), (2, 1, 1))
    assert_size_stride(view_133, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_134, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_135, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_68, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_69, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_70, (), ())
    assert_size_stride(getitem_71, (), ())
    assert_size_stride(mul_57, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_34, (100, 3072), (3072, 1))
    assert_size_stride(mul_61, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_148, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_149, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_150, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_76, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_77, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_78, (), ())
    assert_size_stride(getitem_79, (), ())
    assert_size_stride(mul_63, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_38, (100, 3072), (3072, 1))
    assert_size_stride(mul_67, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_163, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_164, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_165, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_84, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_85, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_86, (), ())
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(mul_69, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_42, (100, 3072), (3072, 1))
    assert_size_stride(mul_73, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(view_178, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_179, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(view_180, (2, 12, 50, 64), (768, 64, 1536, 1))
    assert_size_stride(getitem_92, (2, 12, 50, 64), (38400, 64, 768, 1))
    assert_size_stride(getitem_93, (2, 12, 64), (768, 64, 1))
    assert_size_stride(getitem_94, (), ())
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(mul_75, (50, 2, 768), (1536, 768, 1))
    assert_size_stride(addmm_46, (100, 3072), (3072, 1))
    assert_size_stride(mul_79, (2, 768), (768, 1))
    assert_size_stride(mm, (2, 512), (512, 1))
    assert_size_stride(div_1, (6912, ), (1, ))
    assert_size_stride(div_2, (256, ), (1, ))
    assert_size_stride(sum_6, (), ())
    assert_size_stride(div_3, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(constant_pad_nd_1, (2, 256, 9, 9), (20736, 1, 2304, 256))
    assert_size_stride(convolution_2, (2, 256, 6, 6), (9216, 1, 1536, 256))
    assert_size_stride(clamp_min_6, (1, ), (1, ))
    assert_size_stride(sum_9, (1, ), (1, ))
    assert_size_stride(div_6, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_3, (2, 1, 3, 3), (9, 1, 3, 1))
    assert_size_stride(div_7, (6912, ), (1, ))
    assert_size_stride(div_8, (256, ), (1, ))
    assert_size_stride(sum_18, (), ())
    assert_size_stride(div_9, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(constant_pad_nd_2, (2, 256, 9, 9), (20736, 1, 2304, 256))
    assert_size_stride(convolution_5, (2, 256, 6, 6), (9216, 1, 1536, 256))
    assert_size_stride(clamp_min_10, (1, ), (1, ))
    assert_size_stride(sum_21, (1, ), (1, ))
    assert_size_stride(div_12, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_6, (2, 1, 3, 3), (9, 1, 3, 1))
    assert_size_stride(div_13, (512, ), (1, ))
    assert_size_stride(div_14, (256, ), (1, ))
    assert_size_stride(sum_30, (), ())
    assert_size_stride(where_2, (2, 256), (256, 1))
    assert_size_stride(clamp_min_14, (1, ), (1, ))
    assert_size_stride(sum_33, (1, ), (1, ))
    assert_size_stride(addmm_49, (2, 1), (1, 1))
    assert_size_stride(permute_125, (1, 256), (256, 1))
    assert_size_stride(permute_129, (256, 512), (512, 1))
    assert_size_stride(gt_4, (2, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(gt_5, (2, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(permute_135, (512, 768), (1, 512))
    assert_size_stride(div_39, (2, 1), (1, 1))
    assert_size_stride(permute_137, (768, 3072), (3072, 1))
    assert_size_stride(permute_138, (3072, 768), (768, 1))
    assert_size_stride(div_40, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_139, (768, 768), (768, 1))
    assert_size_stride(permute_145, (2304, 768), (768, 1))
    assert_size_stride(div_41, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_146, (768, 3072), (3072, 1))
    assert_size_stride(permute_147, (3072, 768), (768, 1))
    assert_size_stride(div_42, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_148, (768, 768), (768, 1))
    assert_size_stride(permute_154, (2304, 768), (768, 1))
    assert_size_stride(div_43, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_155, (768, 3072), (3072, 1))
    assert_size_stride(permute_156, (3072, 768), (768, 1))
    assert_size_stride(div_44, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_157, (768, 768), (768, 1))
    assert_size_stride(permute_163, (2304, 768), (768, 1))
    assert_size_stride(div_45, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_164, (768, 3072), (3072, 1))
    assert_size_stride(permute_165, (3072, 768), (768, 1))
    assert_size_stride(div_46, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(permute_172, (2304, 768), (768, 1))
    assert_size_stride(permute_174, (768, 3072), (3072, 1))
    assert_size_stride(permute_175, (3072, 768), (768, 1))
    assert_size_stride(div_48, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_176, (768, 768), (768, 1))
    assert_size_stride(permute_182, (2304, 768), (768, 1))
    assert_size_stride(div_49, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_183, (768, 3072), (3072, 1))
    assert_size_stride(permute_184, (3072, 768), (768, 1))
    assert_size_stride(div_50, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_185, (768, 768), (768, 1))
    assert_size_stride(permute_191, (2304, 768), (768, 1))
    assert_size_stride(div_51, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_192, (768, 3072), (3072, 1))
    assert_size_stride(permute_193, (3072, 768), (768, 1))
    assert_size_stride(div_52, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_200, (2304, 768), (768, 1))
    assert_size_stride(div_53, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_201, (768, 3072), (3072, 1))
    assert_size_stride(permute_202, (3072, 768), (768, 1))
    assert_size_stride(div_54, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_203, (768, 768), (768, 1))
    assert_size_stride(permute_209, (2304, 768), (768, 1))
    assert_size_stride(permute_211, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_56, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_213, (768, 768), (768, 1))
    assert_size_stride(permute_219, (2304, 768), (768, 1))
    assert_size_stride(div_57, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_220, (768, 3072), (3072, 1))
    assert_size_stride(permute_221, (3072, 768), (768, 1))
    assert_size_stride(div_58, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_222, (768, 768), (768, 1))
    assert_size_stride(permute_228, (2304, 768), (768, 1))
    assert_size_stride(div_59, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_229, (768, 3072), (3072, 1))
    assert_size_stride(permute_230, (3072, 768), (768, 1))
    assert_size_stride(div_60, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_231, (768, 768), (768, 1))
    assert_size_stride(permute_237, (2304, 768), (768, 1))
    assert_size_stride(div_61, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_238, (768, 3072), (3072, 1))
    assert_size_stride(permute_239, (3072, 768), (768, 1))
    assert_size_stride(div_62, (50, 2, 1), (2, 1, 1))
    assert_size_stride(permute_240, (768, 768), (768, 1))
    assert_size_stride(permute_246, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (2, 1), (1, 1))
    assert_size_stride(tangents_2, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(tangents_3, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(tangents_4, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(tangents_5, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(tangents_6, (256, 512), (512, 1))
    assert_size_stride(tangents_7, (1, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = addmm_49; del addmm_49  # reuse
        # Topologically Sorted Source Nodes: [target__2], Original ATen: [aten._to_copy, aten.sigmoid, aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_sigmoid_sub_0.run(buf0, tangents_1, 2, grid=grid(2), stream=stream0)
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_1.run(buf0, buf3, 1, grid=grid(1), stream=stream0)
        buf13 = reinterpret_tensor(convolution_6, (2, 3, 3), (9, 3, 1), 0); del convolution_6  # reuse
        buf14 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [target_], Original ATen: [aten.div, aten.sigmoid, aten._to_copy, aten.sub, aten.mul, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2.run(buf13, tangents_1, buf14, 1, 18, grid=grid(1), stream=stream0)
        buf30 = reinterpret_tensor(convolution_3, (2, 3, 3), (9, 3, 1), 0); del convolution_3  # reuse
        buf31 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [target_], Original ATen: [aten.div, aten._to_copy, aten.sigmoid, aten.sub, aten.mul, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_convolution_backward_div_mul_sigmoid_sub_2.run(buf30, tangents_1, buf31, 1, 18, grid=grid(1), stream=stream0)
        del tangents_1
        buf2 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 2), (1, 1), 0), where_2, out=buf2)
        buf5 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [u_10, sigma_5, mv_15, v_10], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum, aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3.run(buf5, tangents_7, primals_178, sum_33, primals_179, clamp_min_14, primals_179, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_14
        del primals_178
        del primals_179
        del sum_33
        del tangents_7
        buf1 = empty_strided_cuda((2, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_125, out=buf1)
        del permute_125
        buf6 = where_2; del where_2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_backward_4.run(buf6, buf1, 512, grid=grid(512), stream=stream0)
        del buf1
        buf7 = empty_strided_cuda((2, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, permute_129, out=buf7)
        del permute_129
        buf47 = empty_strided_cuda((2, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, permute_135, out=buf47)
        del buf7
        del permute_135
        buf50 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_backward_5.run(buf50, primals_153, mul_79, div_39, 2, 768, grid=grid(2), stream=stream0)
        del div_39
        del mul_79
        del primals_153
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf15 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf13, (2, 1, 3, 3), (9, 0, 3, 1), 0), convolution_5, div_12, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf13
        del convolution_5
        del div_12
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf32 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf30, (2, 1, 3, 3), (9, 0, 3, 1), 0), convolution_2, div_6, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf30
        del convolution_2
        del div_6
        buf16 = buf15[0]
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = torch.ops.aten.convolution_backward.default(buf16, constant_pad_nd_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False])
        del buf16
        del constant_pad_nd_2
        del primals_169
        buf17 = buf15[1]
        del buf15
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [u_6, sigma_3], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3.run(buf19, tangents_5, primals_170, sum_21, primals_171, clamp_min_10, primals_171, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_10
        del primals_170
        del primals_171
        del sum_21
        del tangents_5
        buf21 = buf20[0]
        del buf20
        buf22 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_leaky_relu_backward_6.run(gt_4, buf21, buf22, 25088, grid=grid(25088), stream=stream0)
        del buf21
        del gt_4
        buf33 = buf32[0]
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf37 = torch.ops.aten.convolution_backward.default(buf33, constant_pad_nd_1, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False])
        del buf33
        del constant_pad_nd_1
        del primals_160
        buf34 = buf32[1]
        del buf32
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [u_2, sigma_1], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_3.run(buf36, tangents_3, primals_161, sum_9, primals_162, clamp_min_6, primals_162, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_6
        del primals_161
        del primals_162
        del sum_9
        del tangents_3
        buf38 = buf37[0]
        del buf37
        buf39 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_leaky_relu_backward_6.run(gt_5, buf38, buf39, 25088, grid=grid(25088), stream=stream0)
        del buf38
        del gt_5
        buf9 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_7.run(buf6, buf9, 256, grid=grid(256), stream=stream0)
        buf23 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_backward_8.run(buf22, buf23, 256, 98, grid=grid(256), stream=stream0)
        buf40 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_backward_8.run(buf39, buf40, 256, 98, grid=grid(256), stream=stream0)
        buf265 = empty_strided_cuda((50, 2, 768), (768, 38400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_9.run(cat, primals_6, getitem_1, rsqrt, primals_7, primals_8, getitem_3, rsqrt_1, buf265, 76800, grid=grid(76800), stream=stream0)
        del getitem_3
        del primals_8
        buf51 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_10.run(buf50, buf51, 76800, grid=grid(76800), stream=stream0)
        buf52 = empty_strided_cuda((100, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (100, 768), (768, 1), 0), permute_137, out=buf52)
        del permute_137
        buf53 = reinterpret_tensor(buf52, (50, 2, 3072), (6144, 3072, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [mul_27, sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf53, addmm_46, 307200, grid=grid(307200), stream=stream0)
        del addmm_46
        buf54 = reinterpret_tensor(buf51, (100, 768), (768, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (100, 3072), (3072, 1), 0), permute_138, out=buf54)
        del permute_138
        buf57 = reinterpret_tensor(buf54, (50, 2, 768), (1536, 768, 1), 0); del buf54  # reuse
        buf58 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_12.run(buf57, primals_147, mul_75, buf50, div_40, buf58, 100, 768, grid=grid(100), stream=stream0)
        del buf50
        del div_40
        del mul_75
        del primals_147
        buf59 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (100, 768), (768, 1), 0), permute_139, out=buf59)
        del buf58
        del permute_139
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf59, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_178, view_179, view_180, None, getitem_92, getitem_93, getitem_94, getitem_95, 0.0, [True, True, True, False])
        del buf59
        del getitem_92
        del getitem_93
        del getitem_94
        del getitem_95
        del view_178
        del view_179
        del view_180
        buf61 = buf60[0]
        buf62 = buf60[1]
        buf63 = buf60[2]
        del buf60
        buf64 = empty_strided_cuda((50, 2, 3, 768), (4608, 2304, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf63, buf62, buf61, buf64, 230400, grid=grid(230400), stream=stream0)
        del buf61
        buf65 = reinterpret_tensor(buf63, (100, 768), (768, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (100, 2304), (2304, 1), 0), permute_145, out=buf65)
        del permute_145
        buf68 = buf57; del buf57  # reuse
        buf69 = reinterpret_tensor(buf62, (50, 2, 768), (1536, 768, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf68, buf65, primals_141, mul_73, div_41, buf69, 100, 768, grid=grid(100), stream=stream0)
        del div_41
        del mul_73
        del primals_141
        buf70 = reinterpret_tensor(buf53, (100, 3072), (3072, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (100, 768), (768, 1), 0), permute_146, out=buf70)
        del permute_146
        buf71 = reinterpret_tensor(buf70, (50, 2, 3072), (6144, 3072, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [mul_25, sigmoid_10], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf71, addmm_42, 307200, grid=grid(307200), stream=stream0)
        del addmm_42
        buf72 = reinterpret_tensor(buf69, (100, 768), (768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (100, 3072), (3072, 1), 0), permute_147, out=buf72)
        del permute_147
        buf75 = buf68; del buf68  # reuse
        buf76 = reinterpret_tensor(buf65, (50, 2, 768), (1536, 768, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf75, buf72, primals_135, mul_69, div_42, buf76, 100, 768, grid=grid(100), stream=stream0)
        del div_42
        del mul_69
        del primals_135
        buf77 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (100, 768), (768, 1), 0), permute_148, out=buf77)
        del buf76
        del permute_148
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf78 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf77, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_163, view_164, view_165, None, getitem_84, getitem_85, getitem_86, getitem_87, 0.0, [True, True, True, False])
        del buf77
        del getitem_84
        del getitem_85
        del getitem_86
        del getitem_87
        del view_163
        del view_164
        del view_165
        buf79 = buf78[0]
        buf80 = buf78[1]
        buf81 = buf78[2]
        del buf78
        buf82 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf81, buf80, buf79, buf82, 230400, grid=grid(230400), stream=stream0)
        del buf79
        buf83 = reinterpret_tensor(buf81, (100, 768), (768, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (100, 2304), (2304, 1), 0), permute_154, out=buf83)
        del permute_154
        buf86 = buf75; del buf75  # reuse
        buf87 = reinterpret_tensor(buf80, (50, 2, 768), (1536, 768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf86, buf83, primals_129, mul_67, div_43, buf87, 100, 768, grid=grid(100), stream=stream0)
        del div_43
        del mul_67
        del primals_129
        buf88 = reinterpret_tensor(buf71, (100, 3072), (3072, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (100, 768), (768, 1), 0), permute_155, out=buf88)
        del permute_155
        buf89 = reinterpret_tensor(buf88, (50, 2, 3072), (6144, 3072, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [mul_23, sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf89, addmm_38, 307200, grid=grid(307200), stream=stream0)
        del addmm_38
        buf90 = reinterpret_tensor(buf87, (100, 768), (768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (100, 3072), (3072, 1), 0), permute_156, out=buf90)
        del permute_156
        buf93 = buf86; del buf86  # reuse
        buf94 = reinterpret_tensor(buf83, (50, 2, 768), (1536, 768, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf93, buf90, primals_123, mul_63, div_44, buf94, 100, 768, grid=grid(100), stream=stream0)
        del div_44
        del mul_63
        del primals_123
        buf95 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (100, 768), (768, 1), 0), permute_157, out=buf95)
        del buf94
        del permute_157
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf96 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf95, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_148, view_149, view_150, None, getitem_76, getitem_77, getitem_78, getitem_79, 0.0, [True, True, True, False])
        del buf95
        del getitem_76
        del getitem_77
        del getitem_78
        del getitem_79
        del view_148
        del view_149
        del view_150
        buf97 = buf96[0]
        buf98 = buf96[1]
        buf99 = buf96[2]
        del buf96
        buf100 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf99, buf98, buf97, buf100, 230400, grid=grid(230400), stream=stream0)
        del buf97
        buf101 = reinterpret_tensor(buf99, (100, 768), (768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (100, 2304), (2304, 1), 0), permute_163, out=buf101)
        del permute_163
        buf104 = buf93; del buf93  # reuse
        buf105 = reinterpret_tensor(buf98, (50, 2, 768), (1536, 768, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf104, buf101, primals_117, mul_61, div_45, buf105, 100, 768, grid=grid(100), stream=stream0)
        del div_45
        del mul_61
        del primals_117
        buf106 = reinterpret_tensor(buf89, (100, 3072), (3072, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (100, 768), (768, 1), 0), permute_164, out=buf106)
        del permute_164
        buf107 = reinterpret_tensor(buf106, (50, 2, 3072), (6144, 3072, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [mul_21, sigmoid_8], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf107, addmm_34, 307200, grid=grid(307200), stream=stream0)
        del addmm_34
        buf108 = reinterpret_tensor(buf105, (100, 768), (768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (100, 3072), (3072, 1), 0), permute_165, out=buf108)
        del permute_165
        buf111 = buf104; del buf104  # reuse
        buf112 = reinterpret_tensor(buf101, (50, 2, 768), (1536, 768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf111, buf108, primals_111, mul_57, div_46, buf112, 100, 768, grid=grid(100), stream=stream0)
        del div_46
        del mul_57
        del primals_111
        buf113 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (100, 768), (768, 1), 0), permute_166, out=buf113)
        del buf112
        del permute_166
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf114 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf113, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_133, view_134, view_135, None, getitem_68, getitem_69, getitem_70, getitem_71, 0.0, [True, True, True, False])
        del buf113
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_71
        del view_133
        del view_134
        del view_135
        buf115 = buf114[0]
        buf116 = buf114[1]
        buf117 = buf114[2]
        del buf114
        buf118 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf117, buf116, buf115, buf118, 230400, grid=grid(230400), stream=stream0)
        del buf115
        buf119 = reinterpret_tensor(buf117, (100, 768), (768, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (100, 2304), (2304, 1), 0), permute_172, out=buf119)
        del permute_172
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf24 = torch.ops.aten.convolution_backward.default(buf22, reinterpret_tensor(add_62, (2, 768, 7, 7), (38400, 1, 5376, 768), 768), div_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf22
        del div_9
        buf25 = buf24[0]
        buf122 = buf111; del buf111  # reuse
        buf123 = reinterpret_tensor(buf116, (50, 2, 768), (1536, 768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15.run(buf122, buf119, primals_105, add_62, getitem_67, rsqrt_17, buf25, buf123, 100, 768, grid=grid(100), stream=stream0)
        del add_62
        del buf25
        del getitem_67
        del primals_105
        del rsqrt_17
        buf124 = reinterpret_tensor(buf107, (100, 3072), (3072, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (100, 768), (768, 1), 0), permute_174, out=buf124)
        del permute_174
        buf125 = reinterpret_tensor(buf124, (50, 2, 3072), (6144, 3072, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [mul_19, sigmoid_7], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf125, addmm_30, 307200, grid=grid(307200), stream=stream0)
        del addmm_30
        buf126 = reinterpret_tensor(buf123, (100, 768), (768, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (100, 3072), (3072, 1), 0), permute_175, out=buf126)
        del permute_175
        buf129 = buf122; del buf122  # reuse
        buf130 = reinterpret_tensor(buf119, (50, 2, 768), (1536, 768, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf129, buf126, primals_99, mul_51, div_48, buf130, 100, 768, grid=grid(100), stream=stream0)
        del div_48
        del mul_51
        del primals_99
        buf131 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (100, 768), (768, 1), 0), permute_176, out=buf131)
        del buf130
        del permute_176
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf131, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_118, view_119, view_120, None, getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False])
        del buf131
        del getitem_60
        del getitem_61
        del getitem_62
        del getitem_63
        del view_118
        del view_119
        del view_120
        buf26 = buf24[1]
        del buf24
        buf133 = buf132[0]
        buf134 = buf132[1]
        buf135 = buf132[2]
        del buf132
        buf136 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf135, buf134, buf133, buf136, 230400, grid=grid(230400), stream=stream0)
        del buf133
        buf137 = reinterpret_tensor(buf135, (100, 768), (768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (100, 2304), (2304, 1), 0), permute_182, out=buf137)
        del permute_182
        buf140 = buf129; del buf129  # reuse
        buf141 = reinterpret_tensor(buf134, (50, 2, 768), (1536, 768, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf140, buf137, primals_93, mul_49, div_49, buf141, 100, 768, grid=grid(100), stream=stream0)
        del div_49
        del mul_49
        del primals_93
        buf142 = reinterpret_tensor(buf125, (100, 3072), (3072, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (100, 768), (768, 1), 0), permute_183, out=buf142)
        del permute_183
        buf143 = reinterpret_tensor(buf142, (50, 2, 3072), (6144, 3072, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [mul_17, sigmoid_6], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf143, addmm_26, 307200, grid=grid(307200), stream=stream0)
        del addmm_26
        buf144 = reinterpret_tensor(buf141, (100, 768), (768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (100, 3072), (3072, 1), 0), permute_184, out=buf144)
        del permute_184
        buf147 = buf140; del buf140  # reuse
        buf148 = reinterpret_tensor(buf137, (50, 2, 768), (1536, 768, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf147, buf144, primals_87, mul_45, div_50, buf148, 100, 768, grid=grid(100), stream=stream0)
        del div_50
        del mul_45
        del primals_87
        buf149 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (100, 768), (768, 1), 0), permute_185, out=buf149)
        del buf148
        del permute_185
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf150 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf149, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_103, view_104, view_105, None, getitem_52, getitem_53, getitem_54, getitem_55, 0.0, [True, True, True, False])
        del buf149
        del getitem_52
        del getitem_53
        del getitem_54
        del getitem_55
        del view_103
        del view_104
        del view_105
        buf151 = buf150[0]
        buf152 = buf150[1]
        buf153 = buf150[2]
        del buf150
        buf154 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf153, buf152, buf151, buf154, 230400, grid=grid(230400), stream=stream0)
        del buf151
        buf155 = reinterpret_tensor(buf153, (100, 768), (768, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (100, 2304), (2304, 1), 0), permute_191, out=buf155)
        del permute_191
        buf158 = buf147; del buf147  # reuse
        buf159 = reinterpret_tensor(buf152, (50, 2, 768), (1536, 768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf158, buf155, primals_81, mul_43, div_51, buf159, 100, 768, grid=grid(100), stream=stream0)
        del div_51
        del mul_43
        del primals_81
        buf160 = reinterpret_tensor(buf143, (100, 3072), (3072, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (100, 768), (768, 1), 0), permute_192, out=buf160)
        del permute_192
        buf161 = reinterpret_tensor(buf160, (50, 2, 3072), (6144, 3072, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [mul_15, sigmoid_5], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf161, addmm_22, 307200, grid=grid(307200), stream=stream0)
        del addmm_22
        buf162 = reinterpret_tensor(buf159, (100, 768), (768, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (100, 3072), (3072, 1), 0), permute_193, out=buf162)
        del permute_193
        buf165 = buf158; del buf158  # reuse
        buf166 = reinterpret_tensor(buf155, (50, 2, 768), (1536, 768, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf165, buf162, primals_75, mul_39, div_52, buf166, 100, 768, grid=grid(100), stream=stream0)
        del div_52
        del mul_39
        del primals_75
        buf167 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (100, 768), (768, 1), 0), permute_194, out=buf167)
        del buf166
        del permute_194
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf167, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_88, view_89, view_90, None, getitem_44, getitem_45, getitem_46, getitem_47, 0.0, [True, True, True, False])
        del buf167
        del getitem_44
        del getitem_45
        del getitem_46
        del getitem_47
        del view_88
        del view_89
        del view_90
        buf169 = buf168[0]
        buf170 = buf168[1]
        buf171 = buf168[2]
        del buf168
        buf172 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf171, buf170, buf169, buf172, 230400, grid=grid(230400), stream=stream0)
        del buf169
        buf173 = reinterpret_tensor(buf171, (100, 768), (768, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (100, 2304), (2304, 1), 0), permute_200, out=buf173)
        del permute_200
        buf176 = buf165; del buf165  # reuse
        buf177 = reinterpret_tensor(buf170, (50, 2, 768), (1536, 768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf176, buf173, primals_69, mul_37, div_53, buf177, 100, 768, grid=grid(100), stream=stream0)
        del div_53
        del mul_37
        del primals_69
        buf178 = reinterpret_tensor(buf161, (100, 3072), (3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (100, 768), (768, 1), 0), permute_201, out=buf178)
        del permute_201
        buf179 = reinterpret_tensor(buf178, (50, 2, 3072), (6144, 3072, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [mul_13, sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf179, addmm_18, 307200, grid=grid(307200), stream=stream0)
        del addmm_18
        buf180 = reinterpret_tensor(buf177, (100, 768), (768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (100, 3072), (3072, 1), 0), permute_202, out=buf180)
        del permute_202
        buf183 = buf176; del buf176  # reuse
        buf184 = reinterpret_tensor(buf173, (50, 2, 768), (1536, 768, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf183, buf180, primals_63, mul_33, div_54, buf184, 100, 768, grid=grid(100), stream=stream0)
        del div_54
        del mul_33
        del primals_63
        buf185 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (100, 768), (768, 1), 0), permute_203, out=buf185)
        del buf184
        del permute_203
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf186 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf185, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_73, view_74, view_75, None, getitem_36, getitem_37, getitem_38, getitem_39, 0.0, [True, True, True, False])
        del buf185
        del getitem_36
        del getitem_37
        del getitem_38
        del getitem_39
        del view_73
        del view_74
        del view_75
        buf187 = buf186[0]
        buf188 = buf186[1]
        buf189 = buf186[2]
        del buf186
        buf190 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf189, buf188, buf187, buf190, 230400, grid=grid(230400), stream=stream0)
        del buf187
        buf191 = reinterpret_tensor(buf189, (100, 768), (768, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (100, 2304), (2304, 1), 0), permute_209, out=buf191)
        del permute_209
        buf27 = empty_strided_cuda((216, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_16.run(tangents_4, buf26, primals_165, sum_18, buf27, 216, 8192, grid=grid(216), stream=stream0)
        del primals_165
        buf28 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_17.run(buf27, buf28, 1, 216, grid=grid(1), stream=stream0)
        buf29 = empty_strided_cuda((256, 768, 3, 3), (6912, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_18.run(tangents_4, buf26, sum_18, buf28, div_8, div_7, buf29, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del div_7
        del div_8
        del sum_18
        del tangents_4
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf41 = torch.ops.aten.convolution_backward.default(buf39, reinterpret_tensor(add_38, (2, 768, 7, 7), (38400, 1, 5376, 768), 768), div_3, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf39
        del div_3
        buf42 = buf41[0]
        buf194 = buf183; del buf183  # reuse
        buf195 = reinterpret_tensor(buf188, (50, 2, 768), (1536, 768, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_15.run(buf194, buf191, primals_57, add_38, getitem_35, rsqrt_9, buf42, buf195, 100, 768, grid=grid(100), stream=stream0)
        del add_38
        del getitem_35
        del primals_57
        del rsqrt_9
        buf196 = reinterpret_tensor(buf179, (100, 3072), (3072, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (100, 768), (768, 1), 0), permute_211, out=buf196)
        del permute_211
        buf197 = reinterpret_tensor(buf196, (50, 2, 3072), (6144, 3072, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [mul_11, sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf197, addmm_14, 307200, grid=grid(307200), stream=stream0)
        del addmm_14
        buf198 = reinterpret_tensor(buf195, (100, 768), (768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (100, 3072), (3072, 1), 0), permute_212, out=buf198)
        del permute_212
        buf201 = buf194; del buf194  # reuse
        buf202 = reinterpret_tensor(buf191, (50, 2, 768), (1536, 768, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf201, buf198, primals_51, mul_27, div_56, buf202, 100, 768, grid=grid(100), stream=stream0)
        del div_56
        del mul_27
        del primals_51
        buf203 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (100, 768), (768, 1), 0), permute_213, out=buf203)
        del buf202
        del permute_213
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf204 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf203, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_58, view_59, view_60, None, getitem_28, getitem_29, getitem_30, getitem_31, 0.0, [True, True, True, False])
        del buf203
        del getitem_28
        del getitem_29
        del getitem_30
        del getitem_31
        del view_58
        del view_59
        del view_60
        buf43 = buf41[1]
        del buf41
        buf205 = buf204[0]
        buf206 = buf204[1]
        buf207 = buf204[2]
        del buf204
        buf208 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf207, buf206, buf205, buf208, 230400, grid=grid(230400), stream=stream0)
        del buf205
        buf209 = reinterpret_tensor(buf207, (100, 768), (768, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (100, 2304), (2304, 1), 0), permute_219, out=buf209)
        del permute_219
        buf212 = buf201; del buf201  # reuse
        buf213 = reinterpret_tensor(buf206, (50, 2, 768), (1536, 768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf212, buf209, primals_45, mul_25, div_57, buf213, 100, 768, grid=grid(100), stream=stream0)
        del div_57
        del mul_25
        del primals_45
        buf214 = reinterpret_tensor(buf197, (100, 3072), (3072, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (100, 768), (768, 1), 0), permute_220, out=buf214)
        del permute_220
        buf215 = reinterpret_tensor(buf214, (50, 2, 3072), (6144, 3072, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [mul_9, sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf215, addmm_10, 307200, grid=grid(307200), stream=stream0)
        del addmm_10
        buf216 = reinterpret_tensor(buf213, (100, 768), (768, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (100, 3072), (3072, 1), 0), permute_221, out=buf216)
        del permute_221
        buf219 = buf212; del buf212  # reuse
        buf220 = reinterpret_tensor(buf209, (50, 2, 768), (1536, 768, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf219, buf216, primals_39, mul_21, div_58, buf220, 100, 768, grid=grid(100), stream=stream0)
        del div_58
        del mul_21
        del primals_39
        buf221 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (100, 768), (768, 1), 0), permute_222, out=buf221)
        del buf220
        del permute_222
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf222 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf221, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_43, view_44, view_45, None, getitem_20, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False])
        del buf221
        del getitem_20
        del getitem_21
        del getitem_22
        del getitem_23
        del view_43
        del view_44
        del view_45
        buf223 = buf222[0]
        buf224 = buf222[1]
        buf225 = buf222[2]
        del buf222
        buf226 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf225, buf224, buf223, buf226, 230400, grid=grid(230400), stream=stream0)
        del buf223
        buf227 = reinterpret_tensor(buf225, (100, 768), (768, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (100, 2304), (2304, 1), 0), permute_228, out=buf227)
        del permute_228
        buf230 = buf219; del buf219  # reuse
        buf231 = reinterpret_tensor(buf224, (50, 2, 768), (1536, 768, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf230, buf227, primals_33, mul_19, div_59, buf231, 100, 768, grid=grid(100), stream=stream0)
        del div_59
        del mul_19
        del primals_33
        buf232 = reinterpret_tensor(buf215, (100, 3072), (3072, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (100, 768), (768, 1), 0), permute_229, out=buf232)
        del permute_229
        buf233 = reinterpret_tensor(buf232, (50, 2, 3072), (6144, 3072, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [mul_7, sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf233, addmm_6, 307200, grid=grid(307200), stream=stream0)
        del addmm_6
        buf234 = reinterpret_tensor(buf231, (100, 768), (768, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (100, 3072), (3072, 1), 0), permute_230, out=buf234)
        del permute_230
        buf237 = buf230; del buf230  # reuse
        buf238 = reinterpret_tensor(buf227, (50, 2, 768), (1536, 768, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf237, buf234, primals_27, mul_15, div_60, buf238, 100, 768, grid=grid(100), stream=stream0)
        del div_60
        del mul_15
        del primals_27
        buf239 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (100, 768), (768, 1), 0), permute_231, out=buf239)
        del buf238
        del permute_231
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf240 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf239, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_28, view_29, view_30, None, getitem_12, getitem_13, getitem_14, getitem_15, 0.0, [True, True, True, False])
        del buf239
        del getitem_12
        del getitem_13
        del getitem_14
        del getitem_15
        del view_28
        del view_29
        del view_30
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        del buf240
        buf244 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf243, buf242, buf241, buf244, 230400, grid=grid(230400), stream=stream0)
        del buf241
        buf245 = reinterpret_tensor(buf243, (100, 768), (768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (100, 2304), (2304, 1), 0), permute_237, out=buf245)
        del permute_237
        buf248 = buf237; del buf237  # reuse
        buf249 = reinterpret_tensor(buf242, (50, 2, 768), (1536, 768, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf248, buf245, primals_21, mul_13, div_61, buf249, 100, 768, grid=grid(100), stream=stream0)
        del div_61
        del mul_13
        del primals_21
        buf250 = reinterpret_tensor(buf233, (100, 3072), (3072, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (100, 768), (768, 1), 0), permute_238, out=buf250)
        del permute_238
        buf251 = reinterpret_tensor(buf250, (50, 2, 3072), (6144, 3072, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [mul_5, sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sigmoid_backward_11.run(buf251, addmm_2, 307200, grid=grid(307200), stream=stream0)
        del addmm_2
        buf252 = reinterpret_tensor(buf249, (100, 768), (768, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (100, 3072), (3072, 1), 0), permute_239, out=buf252)
        del buf251
        del permute_239
        buf255 = buf248; del buf248  # reuse
        buf256 = reinterpret_tensor(buf245, (50, 2, 768), (1536, 768, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_backward_14.run(buf255, buf252, primals_15, mul_9, div_62, buf256, 100, 768, grid=grid(100), stream=stream0)
        del div_62
        del mul_9
        del primals_15
        buf257 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (100, 768), (768, 1), 0), permute_240, out=buf257)
        del buf256
        del permute_240
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf258 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf257, (2, 12, 50, 64), (768, 64, 1536, 1), 0), view_13, view_14, view_15, None, getitem_4, getitem_5, getitem_6, getitem_7, 0.0, [True, True, True, False])
        del buf257
        del getitem_4
        del getitem_5
        del getitem_6
        del getitem_7
        del view_13
        del view_14
        del view_15
        buf259 = buf258[0]
        buf260 = buf258[1]
        buf261 = buf258[2]
        del buf258
        buf262 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf261, buf260, buf259, buf262, 230400, grid=grid(230400), stream=stream0)
        del buf259
        del buf260
        buf263 = reinterpret_tensor(buf261, (100, 768), (768, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (100, 2304), (2304, 1), 0), permute_246, out=buf263)
        del buf262
        del permute_246
        buf267 = reinterpret_tensor(buf255, (2, 50, 768), (768, 1536, 1), 0); del buf255  # reuse
        buf268 = empty_strided_cuda((2, 50, 1), (1, 2, 100), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_backward_19.run(buf267, buf263, primals_9, buf265, rsqrt_1, primals_7, buf268, 100, 768, grid=grid(100), stream=stream0)
        del buf263
        del buf265
        del primals_7
        del primals_9
        del rsqrt_1
        buf269 = empty_strided_cuda((2, 50, 1), (50, 1, 100), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, ret], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf267, cat, primals_6, getitem_1, rsqrt, buf269, 100, 768, grid=grid(100), stream=stream0)
        buf270 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_backward_21.run(rsqrt, buf267, buf268, cat, primals_6, getitem_1, buf269, buf270, 75264, grid=grid(75264), stream=stream0)
        del buf267
        del buf268
        del buf269
        del cat
        del getitem_1
        del primals_6
        del rsqrt
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf271 = torch.ops.aten.convolution_backward.default(buf270, div, primals_4, [0], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False])
        del buf270
        del div
        del primals_4
        buf272 = buf271[0]
        del buf271
        buf273 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten._adaptive_avg_pool2d_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_backward_div_22.run(buf273, device_put_1, 301056, grid=grid(301056), stream=stream0)
        del device_put_1
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten._adaptive_avg_pool2d_backward]
        buf274 = torch.ops.aten._adaptive_avg_pool2d_backward.default(buf273, add_10)
        del add_10
        del buf273
        buf275 = buf274
        del buf274
        buf44 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_16.run(tangents_2, buf43, primals_156, sum_6, buf44, 216, 8192, grid=grid(216), stream=stream0)
        del primals_156
        buf45 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_17.run(buf44, buf45, 1, 216, grid=grid(1), stream=stream0)
        del buf44
        buf46 = reinterpret_tensor(buf26, (256, 768, 3, 3), (6912, 9, 3, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_18.run(tangents_2, buf43, sum_6, buf45, div_2, div_1, buf46, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del buf43
        del div_1
        del div_2
        del sum_6
        del tangents_2
        buf8 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (256, 2), (1, 256), 0), mm, out=buf8)
        del buf6
        del mm
        buf10 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_23.run(tangents_6, buf8, primals_174, sum_30, buf10, 16, 8192, grid=grid(16), stream=stream0)
        del primals_174
        buf11 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_24.run(buf10, buf11, 1, 16, grid=grid(1), stream=stream0)
        del buf10
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_25.run(buf12, tangents_6, sum_30, buf11, div_14, div_13, 131072, grid=grid(131072), stream=stream0)
        del buf11
        del div_13
        del div_14
        del sum_30
        del tangents_6
        buf276 = empty_strided_cuda((2, 258, 258, 3), (199712, 774, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_26.run(buf276, 399384, grid=grid(399384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_new_zeros_27.run(view, clamp_max, clamp_max_1, buf275, index_put, buf276, 393216, grid=grid(393216), stream=stream0)
        del clamp_max
        del clamp_max_1
        del index_put
        del view
        buf278 = empty_strided_cuda((2, 1, 1, 1, 24, 64), (1536, 3072, 3072, 3072, 64, 1), torch.float32)
        buf281 = empty_strided_cuda((2, 1, 1, 1, 24, 64), (1536, 3072, 3072, 3072, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.sum, aten.add, aten.mul, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_constant_pad_nd_mul_neg_sum_28.run(buf276, inductor_random_default, buf278, buf281, 3072, 128, grid=grid(3072), stream=stream0)
        buf279 = empty_strided_cuda((2, 1, 1, 1, 24), (24, 48, 48, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_constant_pad_nd_sum_29.run(buf278, buf279, 48, 64, grid=grid(48), stream=stream0)
        del buf278
        buf282 = empty_strided_cuda((2, 1, 1, 1, 24), (24, 48, 48, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.neg, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_constant_pad_nd_sum_29.run(buf281, buf282, 48, 64, grid=grid(48), stream=stream0)
        del buf281
        buf280 = reinterpret_tensor(buf0, (2, 1, 1, 1), (1, 2, 2, 2), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_constant_pad_nd_sum_30.run(buf279, buf280, 2, 24, grid=grid(2), stream=stream0)
        del buf279
        buf283 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.neg, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_constant_pad_nd_sum_30.run(buf282, buf283, 2, 24, grid=grid(2), stream=stream0)
        del buf282
        buf284 = empty_strided_cuda((2, 1, 256, 256), (65536, 131072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.div, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_div_mul_sum_31.run(buf276, inductor_random_default, buf280, buf283, buf284, 131072, grid=grid(131072), stream=stream0)
        buf285 = reinterpret_tensor(buf275, (2, 3, 256, 256), (196608, 1, 768, 3), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [add_2, mul], Original ATen: [aten.constant_pad_nd, aten.add, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_div_mul_32.run(buf276, inductor_random_default, buf280, buf283, inductor_random_default_1, buf285, 393216, grid=grid(393216), stream=stream0)
        del buf276
        del buf280
        del buf283
        del inductor_random_default
        del inductor_random_default_1
        buf286 = empty_strided_cuda((2, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_33.run(buf285, buf284, buf286, 393216, grid=grid(393216), stream=stream0)
        del buf284
        del buf285
    return (buf286, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, buf46, None, None, buf40, None, buf36, None, None, buf31, buf29, None, None, buf23, None, buf19, None, None, buf14, buf12, None, None, reinterpret_tensor(buf9, (256, ), (1, ), 0), buf5, None, None, reinterpret_tensor(buf3, (1, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((50, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    inductor_random_default_1 = rand_strided((2, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    inductor_random_default = rand_strided((2, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int64)
    clamp_max = rand_strided((2, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.int64)
    clamp_max_1 = rand_strided((2, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.int64)
    index_put = rand_strided((2, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    add_10 = rand_strided((2, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    device_put_1 = rand_strided((3, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((2, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((2, 50, 768), (38400, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((2, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((2, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_9 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_13 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_12 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_15 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_19 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_20 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_23 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_21 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_58 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_31 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_27 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    add_38 = rand_strided((50, 2, 768), (768, 38400, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_33 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_44 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_47 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_39 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_55 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_45 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_118 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_60 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_51 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    add_62 = rand_strided((50, 2, 768), (768, 38400, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_71 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_57 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_76 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_79 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_63 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_69 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((2, 12, 50, 64), (768, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((2, 12, 50, 64), (38400, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((2, 12, 64), (768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_75 = rand_strided((50, 2, 768), (1536, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((100, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((6912, ), (1, ), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((2, 256, 9, 9), (20736, 1, 2304, 256), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((2, 256, 6, 6), (9216, 1, 1536, 256), device='cuda:0', dtype=torch.float32)
    clamp_min_6 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((2, 1, 3, 3), (9, 1, 3, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((6912, ), (1, ), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_18 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((2, 256, 9, 9), (20736, 1, 2304, 256), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((2, 256, 6, 6), (9216, 1, 1536, 256), device='cuda:0', dtype=torch.float32)
    clamp_min_10 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_21 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((2, 1, 3, 3), (9, 1, 3, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_30 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    where_2 = rand_strided((2, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    clamp_min_14 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    sum_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    addmm_49 = rand_strided((2, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_129 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_4 = rand_strided((2, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.bool)
    gt_5 = rand_strided((2, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.bool)
    permute_135 = rand_strided((512, 768), (1, 512), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((2, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_148 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_202 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((50, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((2, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, primals_6, primals_7, primals_8, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, inductor_random_default_1, inductor_random_default, view, clamp_max, clamp_max_1, index_put, add_10, device_put_1, div, cat, getitem_1, rsqrt, getitem_3, rsqrt_1, view_13, view_14, view_15, getitem_4, getitem_5, getitem_6, getitem_7, mul_9, addmm_2, mul_13, view_28, view_29, view_30, getitem_12, getitem_13, getitem_14, getitem_15, mul_15, addmm_6, mul_19, view_43, view_44, view_45, getitem_20, getitem_21, getitem_22, getitem_23, mul_21, addmm_10, mul_25, view_58, view_59, view_60, getitem_28, getitem_29, getitem_30, getitem_31, mul_27, addmm_14, add_38, getitem_35, rsqrt_9, view_73, view_74, view_75, getitem_36, getitem_37, getitem_38, getitem_39, mul_33, addmm_18, mul_37, view_88, view_89, view_90, getitem_44, getitem_45, getitem_46, getitem_47, mul_39, addmm_22, mul_43, view_103, view_104, view_105, getitem_52, getitem_53, getitem_54, getitem_55, mul_45, addmm_26, mul_49, view_118, view_119, view_120, getitem_60, getitem_61, getitem_62, getitem_63, mul_51, addmm_30, add_62, getitem_67, rsqrt_17, view_133, view_134, view_135, getitem_68, getitem_69, getitem_70, getitem_71, mul_57, addmm_34, mul_61, view_148, view_149, view_150, getitem_76, getitem_77, getitem_78, getitem_79, mul_63, addmm_38, mul_67, view_163, view_164, view_165, getitem_84, getitem_85, getitem_86, getitem_87, mul_69, addmm_42, mul_73, view_178, view_179, view_180, getitem_92, getitem_93, getitem_94, getitem_95, mul_75, addmm_46, mul_79, mm, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, permute_129, gt_4, gt_5, permute_135, div_39, permute_137, permute_138, div_40, permute_139, permute_145, div_41, permute_146, permute_147, div_42, permute_148, permute_154, div_43, permute_155, permute_156, div_44, permute_157, permute_163, div_45, permute_164, permute_165, div_46, permute_166, permute_172, permute_174, permute_175, div_48, permute_176, permute_182, div_49, permute_183, permute_184, div_50, permute_185, permute_191, div_51, permute_192, permute_193, div_52, permute_194, permute_200, div_53, permute_201, permute_202, div_54, permute_203, permute_209, permute_211, permute_212, div_56, permute_213, permute_219, div_57, permute_220, permute_221, div_58, permute_222, permute_228, div_59, permute_229, permute_230, div_60, permute_231, permute_237, div_61, permute_238, permute_239, div_62, permute_240, permute_246, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
