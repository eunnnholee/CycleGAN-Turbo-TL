# AOT ID: ['4_backward']
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


# kernel path: /tmp/torchinductor_elicer/rd/crdzkitro3ewbglbwnr24dr73dcotrb5anx6cyvtd5xsp5hlh3cn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_49,), kwargs = {})
#   %mul_111 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_12, %tangents_1), kwargs = {})
triton_poi_fused_mul_sub_0 = async_compile.triton('triton_poi_fused_mul_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/pi/cpiqlwqykjyp4dihu4bou2hnur3eu52y6gkoisxej5ngib7d6j7j.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.sub, aten.mul, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_19 : [num_users=2] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_33, 9), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%squeeze_13,), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_13, %div_19), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%unsqueeze_22, [0, 2, 3]), kwargs = {})
triton_per_fused_convolution_backward_div_mul_sub_1 = async_compile.triton('triton_per_fused_convolution_backward_div_mul_sub_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_div_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_backward_div_mul_sub_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = 0.1111111111111111
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(in_out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp5, rmask)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sk/cskwgfzy6glihinlc7y3t7lw7xu2m5dievkjqcqcbiypx2wziwca.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [0], True), kwargs = {})
triton_poi_fused_sum_2 = async_compile.triton('triton_poi_fused_sum_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/43/c432fix7ayiw77csow2gsza5gncfvxvjhp47atn4ugw2c35sb3jp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.leaky_relu_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_2, 0), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_1, 0.2), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mm_1, %mul_117), kwargs = {})
triton_poi_fused_leaky_relu_backward_3 = async_compile.triton('triton_poi_fused_leaky_relu_backward_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_backward_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/zn/cznedo2sdhigtabllq2og33di7dnenvegxb6gabo37bj5qmdll2d.py
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
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_3, %div_22), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_93, %sum_36), kwargs = {})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_114,), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_123, %primals_179), kwargs = {})
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_104, [1]), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_31, %expand_27), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_200, %div_16), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_23, %mul_116), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_179, %div_17), kwargs = {})
triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4 = async_compile.triton('triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4', 'mutated_arg_names': ['in_out_ptr0', 'in_ptr3', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_elicer/zp/czp4budsjclas4ftzyxoak2fy3lzrhrp2uv7j6ii2i2oegiqayu6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_3, [0], True), kwargs = {})
triton_poi_fused_sum_5 = async_compile.triton('triton_poi_fused_sum_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/3f/c3fu4ick4spp2v6cwoarwz3ajmoox7nih6s2avt4tcyzehhlgy4w.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_3), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_24, %sum_30), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_95,), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_4, %div_25), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_118,), kwargs = {})
triton_red_fused_add_div_mul_neg_sum_6 = async_compile.triton('triton_red_fused_add_div_mul_neg_sum_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_neg_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_mul_neg_sum_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/c5/cc5w542etj5d7vw3yvzjqfyjk3ppnhofkndn5u46kefp7jvd7kzh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_3), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_24, %sum_30), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_95,), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_4, %div_25), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_118,), kwargs = {})
triton_per_fused_add_div_mul_neg_sum_7 = async_compile.triton('triton_per_fused_add_div_mul_neg_sum_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_neg_sum_7(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/5h/c5hfsjcz6bhqoo4seep64wa5nv4zpxhkxy6pvnqdcwwyxuxx7tpl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_6, %mm_3), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_95, %sum_30), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_203, %div_13), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_26, %mul_120), kwargs = {})
triton_poi_fused_add_div_mul_8 = async_compile.triton('triton_poi_fused_add_div_mul_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/xf/cxfdk2hxt744ysp67ejg5tsv7sthntgyyo26g26coh635zih6es6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_103, [-1, -1, -1, -1]), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd_3, 0.2), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %constant_pad_nd_3, %mul_124), kwargs = {})
triton_poi_fused_constant_pad_nd_leaky_relu_backward_9 = async_compile.triton('triton_poi_fused_constant_pad_nd_leaky_relu_backward_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_leaky_relu_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_leaky_relu_backward_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/3b/c3b2uyhzonkv5oanzgbmtvefubt2zpxp7kwnrewnpmedtfb2hrwl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_4, [0, 2, 3]), kwargs = {})
triton_red_fused_convolution_backward_10 = async_compile.triton('triton_red_fused_convolution_backward_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_backward_10(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/gy/cgy7kh222pr6bpulsbdssjd5lquqoysvgsdot6pqfl56kdbvqqt5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_165, %sum_18), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_30, %sum_18), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_6, %div_31), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_125,), kwargs = {})
triton_red_fused_add_div_mul_neg_sum_11 = async_compile.triton('triton_red_fused_add_div_mul_neg_sum_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_neg_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_mul_neg_sum_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/tq/ctqracsq5orh4bi4h2sjyp3ti5lpgfgbmoqv54mrctvjyaxn3rlg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_165, %sum_18), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_30, %sum_18), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_6, %div_31), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_125,), kwargs = {})
triton_per_fused_add_div_mul_neg_sum_12 = async_compile.triton('triton_per_fused_add_div_mul_neg_sum_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_neg_sum_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/yp/cyp7mdit46nqcc4iv6mbv6wkgt3witoxokoui5oyquntoy6bkg3u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_4, %getitem_107), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %sum_18), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_32, %view_208), kwargs = {})
triton_poi_fused_add_div_13 = async_compile.triton('triton_poi_fused_add_div_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, mm, view_187, view_188, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, gt_4, gt_5, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7 = args
    args.clear()
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
    assert_size_stride(mm, (2, 512), (512, 1))
    assert_size_stride(view_187, (2, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(view_188, (2, 768, 7, 7), (37632, 1, 5376, 768))
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
    assert_size_stride(gt_4, (2, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(gt_5, (2, 256, 7, 7), (12544, 1, 1792, 256))
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
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sub_0.run(buf0, tangents_1, 2, grid=grid(2), stream=stream0)
        buf12 = reinterpret_tensor(convolution_6, (2, 3, 3), (9, 3, 1), 0); del convolution_6  # reuse
        buf13 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.sub, aten.mul, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_backward_div_mul_sub_1.run(buf12, tangents_1, buf13, 1, 18, grid=grid(1), stream=stream0)
        buf28 = reinterpret_tensor(convolution_3, (2, 3, 3), (9, 3, 1), 0); del convolution_3  # reuse
        buf29 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.sub, aten.mul, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_backward_div_mul_sub_1.run(buf28, tangents_1, buf29, 1, 18, grid=grid(1), stream=stream0)
        del tangents_1
        buf1 = empty_strided_cuda((2, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_125, out=buf1)
        del permute_125
        buf2 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 2), (1, 1), 0), where_2, out=buf2)
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_2.run(buf0, buf3, 1, grid=grid(1), stream=stream0)
        del buf0
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf14 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf12, (2, 1, 3, 3), (9, 0, 3, 1), 0), convolution_5, div_12, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf12
        del convolution_5
        del div_12
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf30 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf28, (2, 1, 3, 3), (9, 0, 3, 1), 0), convolution_2, div_6, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf28
        del convolution_2
        del div_6
        buf6 = where_2; del where_2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_backward_3.run(buf6, buf1, 512, grid=grid(512), stream=stream0)
        del buf1
        buf5 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [u_10, sigma_5, mv_15, v_10], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum, aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4.run(buf5, tangents_7, primals_178, sum_33, primals_179, clamp_min_14, primals_179, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_14
        del primals_178
        del primals_179
        del sum_33
        del tangents_7
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf31 = buf30[0]
        buf32 = buf30[1]
        del buf30
        buf7 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (256, 2), (1, 256), 0), mm, out=buf7)
        del mm
        buf8 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_5.run(buf6, buf8, 256, grid=grid(256), stream=stream0)
        del buf6
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf19 = torch.ops.aten.convolution_backward.default(buf15, constant_pad_nd_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False])
        del buf15
        del constant_pad_nd_2
        del primals_169
        buf18 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [u_6, sigma_3], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4.run(buf18, tangents_5, primals_170, sum_21, primals_171, clamp_min_10, primals_171, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_10
        del primals_170
        del primals_171
        del sum_21
        del tangents_5
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf35 = torch.ops.aten.convolution_backward.default(buf31, constant_pad_nd_1, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False])
        del buf31
        del constant_pad_nd_1
        del primals_160
        buf34 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [u_2, sigma_1], Original ATen: [aten.add, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_dot_linalg_vector_norm_mul_mv_neg_sum_4.run(buf34, tangents_3, primals_161, sum_9, primals_162, clamp_min_6, primals_162, 1, 256, grid=grid(1), stream=stream0)
        del clamp_min_6
        del primals_161
        del primals_162
        del sum_9
        del tangents_3
        buf9 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_6.run(tangents_6, buf7, primals_174, sum_30, buf9, 16, 8192, grid=grid(16), stream=stream0)
        del primals_174
        buf20 = buf19[0]
        del buf19
        buf36 = buf35[0]
        del buf35
        buf10 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_7.run(buf9, buf10, 1, 16, grid=grid(1), stream=stream0)
        del buf9
        buf11 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_8.run(buf11, tangents_6, sum_30, buf10, div_14, div_13, 131072, grid=grid(131072), stream=stream0)
        del div_13
        del div_14
        del sum_30
        del tangents_6
        buf21 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_leaky_relu_backward_9.run(gt_4, buf20, buf21, 25088, grid=grid(25088), stream=stream0)
        del buf20
        del gt_4
        buf37 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_leaky_relu_backward_9.run(gt_5, buf36, buf37, 25088, grid=grid(25088), stream=stream0)
        del buf36
        del gt_5
        buf22 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_backward_10.run(buf21, buf22, 256, 98, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf23 = torch.ops.aten.convolution_backward.default(buf21, view_188, div_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf21
        del div_9
        del view_188
        buf38 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_backward_10.run(buf37, buf38, 256, 98, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf39 = torch.ops.aten.convolution_backward.default(buf37, view_187, div_3, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf37
        del div_3
        del view_187
        buf24 = buf23[1]
        del buf23
        buf40 = buf39[1]
        del buf39
        buf25 = empty_strided_cuda((216, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_11.run(tangents_4, buf24, primals_165, sum_18, buf25, 216, 8192, grid=grid(216), stream=stream0)
        del primals_165
        buf41 = empty_strided_cuda((216, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_mul_neg_sum_11.run(tangents_2, buf40, primals_156, sum_6, buf41, 216, 8192, grid=grid(216), stream=stream0)
        del primals_156
        buf26 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_12.run(buf25, buf26, 1, 216, grid=grid(1), stream=stream0)
        del buf25
        buf27 = empty_strided_cuda((256, 768, 3, 3), (6912, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_13.run(tangents_4, buf24, sum_18, buf26, div_8, div_7, buf27, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del div_7
        del div_8
        del sum_18
        del tangents_4
        buf42 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_sum_12.run(buf41, buf42, 1, 216, grid=grid(1), stream=stream0)
        del buf41
        buf43 = reinterpret_tensor(buf24, (256, 768, 3, 3), (6912, 9, 3, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_13.run(tangents_2, buf40, sum_6, buf42, div_2, div_1, buf43, 196608, 9, grid=grid(196608, 9), stream=stream0)
        del buf40
        del buf42
        del div_1
        del div_2
        del sum_6
        del tangents_2
    return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, buf43, None, None, buf38, None, buf34, None, None, buf29, buf27, None, None, buf22, None, buf18, None, None, buf13, buf11, None, None, reinterpret_tensor(buf8, (256, ), (1, ), 0), buf5, None, None, reinterpret_tensor(buf3, (1, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
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
    mm = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((2, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    view_188 = rand_strided((2, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
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
    gt_4 = rand_strided((2, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.bool)
    gt_5 = rand_strided((2, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((2, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, mm, view_187, view_188, div_1, div_2, sum_6, div_3, constant_pad_nd_1, convolution_2, clamp_min_6, sum_9, div_6, convolution_3, div_7, div_8, sum_18, div_9, constant_pad_nd_2, convolution_5, clamp_min_10, sum_21, div_12, convolution_6, div_13, div_14, sum_30, where_2, clamp_min_14, sum_33, addmm_49, permute_125, gt_4, gt_5, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
