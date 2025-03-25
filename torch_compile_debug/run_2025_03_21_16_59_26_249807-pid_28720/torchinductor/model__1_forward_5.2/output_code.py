# AOT ID: ['1_forward']
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


# kernel path: /tmp/torchinductor_elicer/ka/ckabggrutjktkti3pu62ksjwa7ickvma5gcvjnmjl7thxky3tyja.py
# Topologically Sorted Source Nodes: [arange], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   arange => iota
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_0 = async_compile.triton('triton_poi_fused_arange_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3s/c3sv2ba22j5n6tmae7vxp4tjuvzrek2yisnawlwnfjymnheyhnyr.py
# Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   mask => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 256, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_1 = async_compile.triton('triton_poi_fused_ones_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/7d/c7dikebnzz6qt6fo542zjooew56bjn5ibqxfflrr23vdowhqpb5v.py
# Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv => mul_81, sum_1
# Graph fragment:
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_117, %primals_157), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_81, [1]), kwargs = {})
triton_red_fused_mv_2 = async_compile.triton('triton_red_fused_mv_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6912
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 6912*r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/v6/cv6s5tifb5s7qljtvucsebewivi5aq346ib4qaglpe6okb3rmnvo.py
# Topologically Sorted Source Nodes: [mv_3, v_2, mv_4, u_2, sigma_1, weight_1], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
# Source node to ATen node mapping:
#   mv_3 => mul_86, sum_7
#   mv_4 => mul_87, sum_9
#   sigma_1 => mul_89, sum_12
#   u_2 => clamp_min_7, div_5, pow_7, pow_8, sum_10
#   v_2 => clamp_min_6, div_4, pow_5, pow_6, sum_8
#   weight_1 => div_6
# Graph fragment:
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_118, %primals_162), kwargs = {})
#   %sum_7 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_86, [1]), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 2.0), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [0], True), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_8, 0.5), kwargs = {})
#   %clamp_min_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-12), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_7, %expand_11), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_190, %div_4), kwargs = {})
#   %sum_9 : [num_users=4] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_87, [1]), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_9, 2.0), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [0], True), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_10, 0.5), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-12), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_9, %expand_13), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %sum_9), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_89,), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_161, %sum_12), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_162, %div_5), kwargs = {})
#   %copy__3 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_163, %div_4), kwargs = {})
triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3 = async_compile.triton('triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3', 'mutated_arg_names': ['in_ptr1', 'out_ptr5', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, out_ptr5, out_ptr6, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = 1e-12
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tmp3 / tmp10
    tmp12 = tmp0 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp15 * tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = triton_helpers.maximum(tmp17, tmp9)
    tmp19 = tmp15 / tmp18
    tmp20 = tmp19 * tmp15
    tmp21 = tmp0 / tmp20
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp10, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp21, None)
    tl.store(out_ptr5 + (tl.broadcast_to(r0, [RBLOCK])), tmp11, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp19, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/r5/cr5g7nae5ig5lkt7557hg3snrziikxher6klzze5lp2v5qij6zl5.py
# Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_12 => mul_99, sum_25
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_121, %primals_175), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_99, [1]), kwargs = {})
triton_red_fused_mv_4 = async_compile.triton('triton_red_fused_mv_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_4(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r2 + 65536*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + 128*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ev/cevh62su674jf3uvlkoanay2jjnkv5p7sdtf566kz7xaziqh7h4n.py
# Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   rand => inductor_lookup_seed_default, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([2, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
triton_poi_fused_rand_5 = async_compile.triton('triton_poi_fused_rand_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_rand_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_rand_5(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/wn/cwnm6jbil6vyy2iihkylczhidwnaei5za3igtxgpq7fscfsbfhgl.py
# Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   rand_1 => inductor_lookup_seed_default_1, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([2, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
triton_poi_fused_rand_6 = async_compile.triton('triton_poi_fused_rand_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'load_seed_offset': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_rand_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_rand_6(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/77/c77243rimq3oawvj5k7byft7kezgwee3sllnqdj3kqpaju6y43iy.py
# Topologically Sorted Source Nodes: [translation_x, add_4, add_5, grid_x_1], Original ATen: [aten.randint, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_5 => add_5
#   grid_x_1 => clamp_max, clamp_min
#   translation_x => inductor_lookup_seed_default_3, inductor_randint_default_3
# Graph fragment:
#   %inductor_lookup_seed_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 3), kwargs = {})
#   %inductor_randint_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (-32, 33, [2, 1, 1], %inductor_lookup_seed_default_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_1, %inductor_randint_default_3), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 257), kwargs = {})
triton_poi_fused_add_clamp_randint_7 = async_compile.triton('triton_poi_fused_add_clamp_randint_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'load_seed_offset': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_randint_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_randint_7(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 65536
    x1 = ((xindex // 256) % 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x2
    tmp2 = tl.full([1], -32, tl.int64)
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = x1
    tmp6 = tmp5 + tmp4
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/rr/crrobtyqkduae2gjlwuqtav2qvsgdaaqcxinxnjeloppltiqcwcz.py
# Topologically Sorted Source Nodes: [translation_y, add_6, add_7, grid_y_1], Original ATen: [aten.randint, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   add_6 => add_6
#   add_7 => add_7
#   grid_y_1 => clamp_max_1, clamp_min_1
#   translation_y => inductor_lookup_seed_default_4, inductor_randint_default_2
# Graph fragment:
#   %inductor_lookup_seed_default_4 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 4), kwargs = {})
#   %inductor_randint_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (-32, 33, [2, 1, 1], %inductor_lookup_seed_default_4), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_2, %inductor_randint_default_2), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0), kwargs = {})
#   %clamp_max_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 257), kwargs = {})
triton_poi_fused_add_clamp_randint_8 = async_compile.triton('triton_poi_fused_add_clamp_randint_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'load_seed_offset': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_randint_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_randint_8(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 65536
    x0 = (xindex % 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x2
    tmp2 = tl.full([1], -32, tl.int64)
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = x0
    tmp6 = tmp5 + tmp4
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ue/cue5zfnzprxb4n737dfz243j7g7ppqgfvwljrddqerc3lcfaitew.py
# Topologically Sorted Source Nodes: [mask, setitem], Original ATen: [aten.ones, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   mask => full_default
#   setitem => full_default_1, index_put
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 256, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%expand_3, %clamp_max_2, %clamp_max_3], %full_default_1), kwargs = {})
triton_poi_fused_index_put_lift_fresh_ones_9 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_ones_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'load_seed_offset': 'i32', 'load_seed_offset1': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_ones_9', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_ones_9(in_ptr0, in_ptr1, out_ptr0, load_seed_offset, load_seed_offset1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 16384
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2), "index out of bounds: 0 <= tmp4 < 2")
    tmp6 = tl.load(in_ptr1 + load_seed_offset)
    tmp7 = x2
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.full([1], 257, tl.int64)
    tmp10 = triton_helpers.randint64(tmp6, (tmp7).to(tl.uint32), tmp8, tmp9)
    tmp11 = x1
    tmp12 = tmp11 + tmp10
    tmp13 = tl.full([1], 64, tl.int64)
    tmp14 = tmp12 - tmp13
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = tl.full([1], 255, tl.int64)
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tl.load(in_ptr1 + load_seed_offset1)
    tmp19 = triton_helpers.randint64(tmp18, (tmp7).to(tl.uint32), tmp8, tmp9)
    tmp20 = x0
    tmp21 = tmp20 + tmp19
    tmp22 = tmp21 - tmp13
    tmp23 = triton_helpers.maximum(tmp22, tmp8)
    tmp24 = triton_helpers.minimum(tmp23, tmp16)
    tmp25 = 0.0
    tl.store(out_ptr0 + (tl.broadcast_to(tmp24 + 256*tmp17 + 65536*tmp4, [XBLOCK])), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/th/cthhlpyqksxr76zgtzhsxtdioft76v72lwg2djx5krolq3tkyc5j.py
# Topologically Sorted Source Nodes: [v], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   v => div_1, pow_1, sum_2
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [0], True), kwargs = {})
#   %div_1 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %expand_7), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_158, %div_1), kwargs = {})
triton_red_fused_div_linalg_vector_norm_10 = async_compile.triton('triton_red_fused_div_linalg_vector_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_linalg_vector_norm_10', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_linalg_vector_norm_10(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = libdevice.sqrt(tmp3)
        tmp7 = 1e-12
        tmp8 = triton_helpers.maximum(tmp6, tmp7)
        tmp9 = tmp5 / tmp8
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/up/cupxlusrh7fao7swcpllflupf2b7d6izrufrir5gp4hfxac6lgoj.py
# Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_12 => mul_99, sum_25
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_121, %primals_175), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_99, [1]), kwargs = {})
triton_per_fused_mv_11 = async_compile.triton('triton_per_fused_mv_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/uu/cuutrjw5emfpwwl7hwxs4qdsdlv6uyk32sj6wdmik7bqlnot22rw.py
# Topologically Sorted Source Nodes: [sub, x, x_mean, sub_1, mul, mul_1, x_1, x_mean_1], Original ATen: [aten.sub, aten.add, aten.mean, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   sub => sub
#   sub_1 => sub_1
#   x => add
#   x_1 => add_1
#   x_mean => mean
#   x_mean_1 => mean_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %sub), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add, [1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mean), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default_1, 2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %mul), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mean), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3], True), kwargs = {})
triton_red_fused_add_mean_mul_sub_12 = async_compile.triton('triton_red_fused_add_mean_mul_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_sub_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mean_mul_sub_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x2 = xindex // 24
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    x0 = (xindex % 8)
    tmp16 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + 8192*x4), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r3 + 8192*x0 + 196608*x2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (65536 + r3 + 8192*x0 + 196608*x2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr0 + (131072 + r3 + 8192*x0 + 196608*x2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 + tmp3
        tmp6 = tmp5 + tmp3
        tmp8 = tmp7 + tmp3
        tmp9 = tmp6 + tmp8
        tmp11 = tmp10 + tmp3
        tmp12 = tmp9 + tmp11
        tmp13 = 3.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp4 - tmp14
        tmp17 = 2.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tmp19 + tmp14
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tl.store(out_ptr0 + (r3 + 8192*x4), tmp20, rmask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x4), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/aa/caa5vw5hbdr4ulw2nuqxc7nqmh6ajetewo3gatmm4jwuphn3ttfu.py
# Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_1 => mul_82, sum_3
# Graph fragment:
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_189, %div_1), kwargs = {})
#   %sum_3 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_82, [1]), kwargs = {})
triton_red_fused_mv_13 = async_compile.triton('triton_red_fused_mv_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_13(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 6912*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/se/csexka4kxv3cpjhec3g4wbx5alzhz3kkz3s3zjeltd4lmzeqtcog.py
# Topologically Sorted Source Nodes: [u, sigma], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
# Source node to ATen node mapping:
#   sigma => mul_84, sum_6
#   u => div_2, pow_3, sum_4
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 2.0), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [0], True), kwargs = {})
#   %div_2 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %expand_9), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sum_3), kwargs = {})
#   %sum_6 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_84,), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_157, %div_2), kwargs = {})
triton_per_fused_div_dot_linalg_vector_norm_14 = async_compile.triton('triton_per_fused_div_dot_linalg_vector_norm_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_dot_linalg_vector_norm_14', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_dot_linalg_vector_norm_14(in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = 1e-12
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 / tmp7
    tmp9 = tmp8 * tmp0
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, None)
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, None)
    tl.store(out_ptr3 + (tl.full([1], 0, tl.int32)), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/kw/ckw6bzhpkmjmuajat6p6and6gvp5jimamqgyxdoela6ttpjba5nu.py
# Topologically Sorted Source Nodes: [v_8], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   v_8 => div_13, pow_17, sum_26
# Graph fragment:
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_25, 2.0), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_17, [0], True), kwargs = {})
#   %div_13 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_25, %expand_23), kwargs = {})
#   %copy__9 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_176, %div_13), kwargs = {})
triton_per_fused_div_linalg_vector_norm_15 = async_compile.triton('triton_per_fused_div_linalg_vector_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_linalg_vector_norm_15', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_linalg_vector_norm_15(in_out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_out_ptr0 + (r0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = 1e-12
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 / tmp7
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xn/cxnfbrdte5l3kwbujcjicegehmuril22j3yoqrycm7ludbsxs32z.py
# Topologically Sorted Source Nodes: [x_mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3], True), kwargs = {})
triton_per_fused_mean_16 = async_compile.triton('triton_per_fused_mean_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_16(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/74/c74tntgdxzfoy6e3hkt364heijmy2k35a5hoih7p3ha5r7pi4ndw.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul_4, add_10], Original ATen: [aten.mul, aten.clone, aten.add]
# Source node to ATen node mapping:
#   add_10 => add_10
#   mul_4 => mul_4
#   x_4 => mul_3
#   x_5 => clone_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %unsqueeze_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_3,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_1, 0.5), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.5), kwargs = {})
triton_poi_fused_add_clone_mul_17 = async_compile.triton('triton_poi_fused_add_clone_mul_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_mul_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 196608
    x0 = (xindex % 65536)
    x1 = ((xindex // 65536) % 3)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x0 + 65536*x2), None, eviction_policy='evict_last')
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
    tmp17 = (-1) + tmp10
    tmp18 = tmp17.to(tl.int32)
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 256, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = (-1) + tmp15
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tmp24 >= tmp19
    tmp26 = tmp24 < tmp21
    tmp27 = tmp20 & tmp22
    tmp28 = tmp27 & tmp25
    tmp29 = tmp28 & tmp26
    tmp30 = tl.load(in_ptr3 + ((-257) + tmp15 + 256*tmp10 + 65536*x1 + 196608*tmp4), tmp29, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr4 + (tl.broadcast_to(tmp4, [XBLOCK])), tmp29, eviction_policy='evict_last', other=0.0)
    tmp32 = 196608.0
    tmp33 = tmp31 / tmp32
    tmp34 = tmp30 - tmp33
    tmp35 = tl.load(in_ptr5 + (tl.broadcast_to(tmp4, [XBLOCK])), tmp29, eviction_policy='evict_last', other=0.0)
    tmp36 = 0.5
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 * tmp37
    tmp39 = tmp38 + tmp33
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp29, tmp39, tmp40)
    tmp43 = tmp41 * tmp42
    tmp44 = 0.5
    tmp45 = tmp43 * tmp44
    tmp46 = tmp45 + tmp44
    tl.store(out_ptr0 + (x3), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/qx/cqxgt6arl7xkibqmsziivjeczb2h4js2ynoqs5d6g54gsab43z3l.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8], Original ATen: [aten._adaptive_avg_pool2d, aten.sub, aten.div]
# Source node to ATen node mapping:
#   x_6 => _adaptive_avg_pool2d
#   x_7 => sub_5
#   x_8 => div
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%add_10, [224, 224]), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_adaptive_avg_pool2d, %device_put), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %device_put_1), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_div_sub_18 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_div_sub_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_div_sub_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_div_sub_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 224) % 224)
    x0 = (xindex % 224)
    x2 = xindex // 50176
    x7 = xindex
    x4 = ((xindex // 50176) % 3)
    tmp76 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = (8*x1) // 7
    tmp1 = (479 + 256*x1) // 224
    tmp2 = tmp0 < tmp1
    tmp3 = (8*x0) // 7
    tmp4 = (479 + 256*x0) // 224
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp6 & xmask, other=0.0)
    tmp8 = 1 + ((8*x0) // 7)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp10 & xmask, other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((8*x0) // 7)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp15 & xmask, other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 1 + ((8*x1) // 7)
    tmp19 = tmp18 < tmp1
    tmp20 = tmp19 & tmp5
    tmp21 = tl.load(in_ptr0 + (256 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp20 & xmask, other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp19 & tmp9
    tmp24 = tl.load(in_ptr0 + (257 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp22
    tmp26 = tmp19 & tmp14
    tmp27 = tl.load(in_ptr0 + (258 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp26 & xmask, other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = 2 + ((8*x1) // 7)
    tmp30 = tmp29 < tmp1
    tmp31 = tmp30 & tmp5
    tmp32 = tl.load(in_ptr0 + (512 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp31 & xmask, other=0.0)
    tmp33 = tmp32 + tmp28
    tmp34 = tmp30 & tmp9
    tmp35 = tl.load(in_ptr0 + (513 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp34 & xmask, other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp14
    tmp38 = tl.load(in_ptr0 + (514 + 256*((8*x1) // 7) + 65536*x2 + ((8*x0) // 7)), tmp37 & xmask, other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1.0
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp6, tmp40, tmp41)
    tmp43 = 1.0
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp10, tmp43, tmp44)
    tmp46 = tmp45 + tmp42
    tmp47 = 1.0
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp15, tmp47, tmp48)
    tmp50 = tmp49 + tmp46
    tmp51 = 1.0
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp20, tmp51, tmp52)
    tmp54 = tmp53 + tmp50
    tmp55 = 1.0
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp23, tmp55, tmp56)
    tmp58 = tmp57 + tmp54
    tmp59 = 1.0
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp26, tmp59, tmp60)
    tmp62 = tmp61 + tmp58
    tmp63 = 1.0
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp31, tmp63, tmp64)
    tmp66 = tmp65 + tmp62
    tmp67 = 1.0
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp34, tmp67, tmp68)
    tmp70 = tmp69 + tmp66
    tmp71 = 1.0
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp37, tmp71, tmp72)
    tmp74 = tmp73 + tmp70
    tmp75 = tmp39 / tmp74
    tmp77 = tmp75 - tmp76
    tmp79 = tmp77 / tmp78
    tl.store(in_out_ptr0 + (x7), tmp79, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cm/ccmhgvwwgtqcq2o3etxoesby2fukxxitcljs7e43gkxltgstw5pr.py
# Topologically Sorted Source Nodes: [weight, input_38], Original ATen: [aten.div, aten.convolution]
# Source node to ATen node mapping:
#   input_38 => convolution_1
#   weight => div_3
# Graph fragment:
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_156, %sum_6), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_187, %div_3, %primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_19 = async_compile.triton('triton_poi_fused_convolution_div_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_19(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196608
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 768)
    y3 = yindex // 768
    tmp0 = tl.load(in_ptr0 + (x1 + 9*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, YBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x1 + 9*y0), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (y2 + 768*x1 + 6912*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/go/cgojtt5hzhwl37u5vpcvabzolqzixp4vkadc6fn2sxlgn653z5i4.py
# Topologically Sorted Source Nodes: [mv_13], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_13 => mul_100, sum_27
# Graph fragment:
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_193, %div_13), kwargs = {})
#   %sum_27 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_100, [1]), kwargs = {})
triton_per_fused_mv_20 = async_compile.triton('triton_per_fused_mv_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_20(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5t/c5tggowkytgo4r3k7573g6cokmlqewgg6ccr2yqqwltn5pfhlyct.py
# Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_4 => div_15
# Graph fragment:
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
triton_poi_fused_div_21 = async_compile.triton('triton_poi_fused_div_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fi/cfi3qdonnevk2mffiq67jtdasifwx3vyl67p7jpftpsm6wzzt2cc.py
# Topologically Sorted Source Nodes: [x_12, x_13, ret, ret_1], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret => add_13, add_14, mul_5, mul_6, rsqrt, sub_6, var_mean
#   ret_1 => clone_2, var_mean_1
#   x_12 => cat
#   x_13 => add_12
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_11, %permute_2], 1), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_6), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %primals_7), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %primals_8), kwargs = {})
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_2, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_cat_native_layer_norm_22 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 3, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 50)
    x1 = xindex // 50
    x3 = xindex
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (r2 + 768*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 0.0
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 50, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr1 + (49*r2 + 37632*x1 + ((-1) + x0)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.where(tmp4, tmp9, tmp13)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
        tl.store(out_ptr0 + (r2 + 768*x3), tmp14, rmask & xmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tmp21 = 768.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, xmask)
    tmp36_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp36_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp36_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(out_ptr0 + (r2 + 768*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr2 + (r2 + 768*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28 - tmp18
        tmp30 = tmp29 * tmp25
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp36_mean_next, tmp36_m2_next, tmp36_weight_next = triton_helpers.welford_reduce(
            tmp35, tmp36_mean, tmp36_m2, tmp36_weight, roffset == 0
        )
        tmp36_mean = tl.where(rmask & xmask, tmp36_mean_next, tmp36_mean)
        tmp36_m2 = tl.where(rmask & xmask, tmp36_m2_next, tmp36_m2)
        tmp36_weight = tl.where(rmask & xmask, tmp36_weight_next, tmp36_weight)
        tl.store(out_ptr2 + (r2 + 768*x3), tmp34, rmask & xmask)
    tmp36_tmp, tmp37_tmp, tmp38_tmp = triton_helpers.welford(
        tmp36_mean, tmp36_m2, tmp36_weight, 1
    )
    tmp36 = tmp36_tmp[:, None]
    tmp37 = tmp37_tmp[:, None]
    tmp38 = tmp38_tmp[:, None]
    tl.store(out_ptr3 + (x3), tmp37, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/kp/ckp3dmsbvfzmjxcxe3a75q4xgl5esytaxig2sfipwlkdl4l2orkb.py
# Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_1 => add_15, add_16, clone_2, mul_7, mul_8, rsqrt_1, sub_7, var_mean_1
# Graph fragment:
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_2, %getitem_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_9), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_10), kwargs = {})
triton_per_fused_native_layer_norm_23 = async_compile.triton('triton_per_fused_native_layer_norm_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    x0 = (xindex % 2)
    x1 = xindex // 2
    x3 = xindex
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (x1 + 50*x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (r2 + 768*x1 + 38400*x0), rmask, other=0.0)
    tmp25 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 768.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = tmp23 * tmp5
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr0 + (x3), tmp5, None)
    tl.store(out_ptr2 + (r2 + 768*x3), tmp28, rmask)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/2m/c2mtyix2gy4skwfuavw7foscusta6kvxzvycpdujkvyf2pdfwreu.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_3
# Graph fragment:
#   %clone_3 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_24 = async_compile.triton('triton_poi_fused_clone_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 100)
    x2 = xindex // 76800
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 2304*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 768*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3v/c3veoott7qg4n2nz27iz3c6cnif2x63a6akyovypgh5fjaokx2y6.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_13
# Graph fragment:
#   %view_13 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [2, 12, 50, 64]), kwargs = {})
triton_poi_fused_view_25 = async_compile.triton('triton_poi_fused_view_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cl/ccls4xkgkbbd7yr3t3to4hmj74cxczjxewn5s4y45fas4kikd5s5.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_14
# Graph fragment:
#   %view_14 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [2, 12, 50, 64]), kwargs = {})
triton_poi_fused_view_26 = async_compile.triton('triton_poi_fused_view_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (76800 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ej/cejhmtdpr7zmm2zbza5mijjsehmtnosdyhib2ucxjjnz6zitjwms.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_15
# Graph fragment:
#   %view_15 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [2, 12, 50, 64]), kwargs = {})
triton_poi_fused_view_27 = async_compile.triton('triton_poi_fused_view_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (153600 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/j7/cj7f5nn54qhnhv3jwsd2q7qsejaqx2jz4tg6usvjpu4sjxkrnkjv.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 38400*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/es/cesl73nw5xmkis7mhdp57sfzrehrhjizvmoktve755i3c3vl2raf.py
# Topologically Sorted Source Nodes: [x_16, ret_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret_2 => add_18, add_19, clone_5, mul_10, mul_9, rsqrt_2, sub_8, var_mean_2
#   x_16 => add_17
# Graph fragment:
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_3, %view_17), kwargs = {})
#   %clone_5 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_17,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_5, %getitem_9), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_15), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_16), kwargs = {})
#   %div_62 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x0 = (xindex % 2)
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x1 + 38400*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 768*x3), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0013020833333333333
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r2 + 768*x3), tmp27, rmask)
    tl.store(out_ptr3 + (r2 + 768*x3), tmp31, rmask)
    tl.store(out_ptr4 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ac/cacw3wzxbk7poyjpq65b7x3dgkds6wdjerzkdploybzy72dnzoq5.py
# Topologically Sorted Source Nodes: [mul_5, sigmoid, input_2], Original ATen: [aten.mul, aten.sigmoid]
# Source node to ATen node mapping:
#   input_2 => mul_12
#   mul_5 => mul_11
#   sigmoid => sigmoid
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 1.702), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_11,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %sigmoid), kwargs = {})
triton_poi_fused_mul_sigmoid_30 = async_compile.triton('triton_poi_fused_mul_sigmoid_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 307200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.702
    tmp2 = tmp0 * tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/s6/cs6llrvisrt5jjfnoll2l7qj7ztgckhdgzcorgbrk6dffwysh5da.py
# Topologically Sorted Source Nodes: [x_16, x_17, ret_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret_3 => add_21, add_22, clone_6, mul_13, mul_14, rsqrt_3, sub_9, var_mean_3
#   x_16 => add_17
#   x_17 => add_20
# Graph fragment:
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_3, %view_17), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_21), kwargs = {})
#   %clone_6 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_20,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_6, %getitem_11), kwargs = {})
#   %mul_13 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_3), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %primals_21), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %primals_22), kwargs = {})
#   %div_61 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x0 = (xindex % 2)
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x1 + 38400*x0), rmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r2 + 768*x3), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + 768*x3), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0013020833333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r2 + 768*x3), tmp8, rmask)
    tl.store(out_ptr2 + (r2 + 768*x3), tmp31, rmask)
    tl.store(out_ptr3 + (r2 + 768*x3), tmp35, rmask)
    tl.store(out_ptr4 + (x3), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/wr/cwrfnxz54ew5doqskooiyiaakkhmiabl766pgnvtkpeirmrgg2ap.py
# Topologically Sorted Source Nodes: [x_18, ret_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret_4 => add_24, add_25, clone_9, mul_15, mul_16, rsqrt_4, sub_10, var_mean_4
#   x_18 => add_23
# Graph fragment:
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_32), kwargs = {})
#   %clone_9 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_23,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_9, %getitem_17), kwargs = {})
#   %mul_15 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_4), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_27), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %primals_28), kwargs = {})
#   %div_60 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0013020833333333333
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 768*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/mv/cmvwav3l4kmr5edusz4vs6en3ortlogdtzp25w4zrliquprc5p5i.py
# Topologically Sorted Source Nodes: [x_18, x_19, ret_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret_5 => add_27, add_28, clone_10, mul_19, mul_20, rsqrt_5, sub_11, var_mean_5
#   x_18 => add_23
#   x_19 => add_26
# Graph fragment:
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_32), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_36), kwargs = {})
#   %clone_10 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_26,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_10, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_10, %getitem_19), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_5), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %primals_33), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %primals_34), kwargs = {})
#   %div_59 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 768*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0013020833333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + 768*x0), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cz/cczikxlaqm6ouvrwp32dww3bmpxmzke6vichjnjmzsfuh6fouusd.py
# Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_22 => add_35
#   x_23 => add_38
# Graph fragment:
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_62), kwargs = {})
#   %add_38 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %view_66), kwargs = {})
triton_poi_fused_add_34 = async_compile.triton('triton_poi_fused_add_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x0 + 768*x2 + 38400*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/66/c66z3llc657es5tg5vq3nqnwitqb7irzj2sn2fj6q766rvdpw4fk.py
# Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_9 => clone_18, var_mean_9
# Graph fragment:
#   %clone_18 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_38,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_18, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_35 = async_compile.triton('triton_per_fused_native_layer_norm_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_35(in_ptr0, out_ptr0, xnumel, rnumel):
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/v4/cv4riuszom22ka3ztv3vqcp2befleubwhokrzqzuzuysp6r4sf5x.py
# Topologically Sorted Source Nodes: [input_38, input_39, pad_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_38 => convolution_1
#   input_39 => gt, mul_85, where
#   pad_1 => constant_pad_nd_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_187, %div_3, %primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul_85), kwargs = {})
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 2304) % 9)
    x1 = ((xindex // 256) % 9)
    x3 = xindex // 20736
    x4 = (xindex % 2304)
    x0 = (xindex % 256)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-2048) + x4 + 1792*x2 + 12544*x3), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp10, tmp18, tmp19)
    tl.store(out_ptr0 + (x6), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/h5/ch5rx7merc5uveqlttqofhp5rx54c3u27cgajbok6pwlimde77pe.py
# Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_38 => convolution_1
#   input_39 => gt, mul_85, where
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_187, %div_3, %primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul_85), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5d/c5duajbqmjtsejo4d7emx7o4uwjzemrcywlz23tfnn5f7tszpdhg.py
# Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_41 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_2, %div_6, %primals_164, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 6
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 1536*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 6*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/23/c23myiukcqaevtomhbzzml6wlabqtnek42sdr64iyuvlac24ygbz.py
# Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   ret_25 => add_87, add_88, clone_50, mul_79, mul_80, rsqrt_25, sub_31, var_mean_25
# Graph fragment:
#   %clone_50 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%select_36,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_50, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_87,), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_50, %getitem_99), kwargs = {})
#   %mul_79 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %rsqrt_25), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %primals_153), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %primals_154), kwargs = {})
#   %div_39 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_25, 768), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_39 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + 768*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0013020833333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp35, rmask)
    tl.store(out_ptr3 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xt/cxtfdr6n2sd47uaso2beakcimdn6jpwbknv5od6yuc4zlv2bbj3l.py
# Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.addmm, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_47 => add_tensor
#   input_48 => gt_2, mul_103, where_2
# Graph fragment:
#   %add_tensor : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_177), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_tensor, 0), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_tensor, %mul_103), kwargs = {})
triton_poi_fused_addmm_leaky_relu_40 = async_compile.triton('triton_poi_fused_addmm_leaky_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_leaky_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_leaky_relu_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/f6/cf65iatg3phv2zvc67vyabj7d7vt5jt5ylyjgtwxtcy6fm6y6vll.py
# Topologically Sorted Source Nodes: [input_41, input_46, loss_, mean_2, loss, loss__2, mean_3, loss_1, loss__4, loss_2, loss_3], Original ATen: [aten.convolution, aten.binary_cross_entropy_with_logits, aten.mean, aten.add]
# Source node to ATen node mapping:
#   input_41 => convolution_3
#   input_46 => convolution_6
#   loss => add_89
#   loss_ => abs_1, exp, full_default_4, full_default_5, log1p, minimum, mul_109, neg, sub_33, sub_34
#   loss_1 => add_90
#   loss_2 => add_91
#   loss_3 => add_92
#   loss__2 => abs_2, exp_1, log1p_1, minimum_1, mul_110, neg_1, sub_36, sub_37
#   loss__4 => abs_3, exp_2, full_default_10, log1p_2, minimum_2, mul_111, neg_2, sub_39, sub_40
#   mean_2 => mean_2
#   mean_3 => mean_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_2, %div_6, %primals_164, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_5, %div_12, %primals_173, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %full_default_4 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 3, 3], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_4, %squeeze_12), kwargs = {})
#   %full_default_5 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_5, %squeeze_12), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_12,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_109, %sub_33), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_34, [1, 2]), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_195, 0), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_4, %squeeze_13), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_5, %squeeze_13), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_13,), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %log1p_1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_110, %sub_36), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_37, [1, 2]), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %view_196), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 1], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_10, %addmm_49), kwargs = {})
#   %minimum_2 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_5, %addmm_49), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%addmm_49,), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_3,), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_2, %log1p_2), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_111, %sub_39), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %sub_40), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, 0), kwargs = {})
triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_41 = async_compile.triton('triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_41(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 9*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp17 = tl.load(in_out_ptr1 + (r1 + 9*x0), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr1 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp38 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp4 * tmp3
    tmp6 = triton_helpers.minimum(tmp4, tmp3)
    tmp7 = tl_math.abs(tmp3)
    tmp8 = -tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tmp5 - tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp20 = tmp17 + tmp19
    tmp21 = tmp4 * tmp20
    tmp22 = triton_helpers.minimum(tmp4, tmp20)
    tmp23 = tl_math.abs(tmp20)
    tmp24 = -tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = libdevice.log1p(tmp25)
    tmp27 = tmp22 - tmp26
    tmp28 = tmp21 - tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = 9.0
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 + tmp4
    tmp36 = tmp32 / tmp33
    tmp37 = tmp35 + tmp36
    tmp39 = tmp4 * tmp38
    tmp40 = triton_helpers.minimum(tmp4, tmp38)
    tmp41 = tl_math.abs(tmp38)
    tmp42 = -tmp41
    tmp43 = tl_math.exp(tmp42)
    tmp44 = libdevice.log1p(tmp43)
    tmp45 = tmp40 - tmp44
    tmp46 = tmp39 - tmp45
    tmp47 = tmp37 + tmp46
    tmp48 = tmp47 + tmp4
    tl.store(in_out_ptr0 + (r1 + 9*x0), tmp3, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + 9*x0), tmp20, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp48, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181 = args
    args.clear()
    assert_size_stride(primals_1, (2, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(primals_2, (3, ), (1, ))
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (768, 3, 32, 32), (3072, 1024, 32, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (50, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (2304, ), (1, ))
    assert_size_stride(primals_12, (2304, 768), (768, 1))
    assert_size_stride(primals_13, (768, 768), (768, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (3072, 768), (768, 1))
    assert_size_stride(primals_18, (3072, ), (1, ))
    assert_size_stride(primals_19, (768, 3072), (3072, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (2304, ), (1, ))
    assert_size_stride(primals_24, (2304, 768), (768, 1))
    assert_size_stride(primals_25, (768, 768), (768, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (3072, 768), (768, 1))
    assert_size_stride(primals_30, (3072, ), (1, ))
    assert_size_stride(primals_31, (768, 3072), (3072, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (2304, ), (1, ))
    assert_size_stride(primals_36, (2304, 768), (768, 1))
    assert_size_stride(primals_37, (768, 768), (768, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (3072, 768), (768, 1))
    assert_size_stride(primals_42, (3072, ), (1, ))
    assert_size_stride(primals_43, (768, 3072), (3072, 1))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (2304, ), (1, ))
    assert_size_stride(primals_48, (2304, 768), (768, 1))
    assert_size_stride(primals_49, (768, 768), (768, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (3072, 768), (768, 1))
    assert_size_stride(primals_54, (3072, ), (1, ))
    assert_size_stride(primals_55, (768, 3072), (3072, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (2304, ), (1, ))
    assert_size_stride(primals_60, (2304, 768), (768, 1))
    assert_size_stride(primals_61, (768, 768), (768, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (3072, 768), (768, 1))
    assert_size_stride(primals_66, (3072, ), (1, ))
    assert_size_stride(primals_67, (768, 3072), (3072, 1))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (2304, ), (1, ))
    assert_size_stride(primals_72, (2304, 768), (768, 1))
    assert_size_stride(primals_73, (768, 768), (768, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (3072, 768), (768, 1))
    assert_size_stride(primals_78, (3072, ), (1, ))
    assert_size_stride(primals_79, (768, 3072), (3072, 1))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (2304, ), (1, ))
    assert_size_stride(primals_84, (2304, 768), (768, 1))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (3072, 768), (768, 1))
    assert_size_stride(primals_90, (3072, ), (1, ))
    assert_size_stride(primals_91, (768, 3072), (3072, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (2304, ), (1, ))
    assert_size_stride(primals_96, (2304, 768), (768, 1))
    assert_size_stride(primals_97, (768, 768), (768, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (2304, ), (1, ))
    assert_size_stride(primals_108, (2304, 768), (768, 1))
    assert_size_stride(primals_109, (768, 768), (768, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (3072, 768), (768, 1))
    assert_size_stride(primals_114, (3072, ), (1, ))
    assert_size_stride(primals_115, (768, 3072), (3072, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (2304, ), (1, ))
    assert_size_stride(primals_120, (2304, 768), (768, 1))
    assert_size_stride(primals_121, (768, 768), (768, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (3072, 768), (768, 1))
    assert_size_stride(primals_126, (3072, ), (1, ))
    assert_size_stride(primals_127, (768, 3072), (3072, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (2304, ), (1, ))
    assert_size_stride(primals_132, (2304, 768), (768, 1))
    assert_size_stride(primals_133, (768, 768), (768, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (3072, 768), (768, 1))
    assert_size_stride(primals_138, (3072, ), (1, ))
    assert_size_stride(primals_139, (768, 3072), (3072, 1))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (2304, ), (1, ))
    assert_size_stride(primals_144, (2304, 768), (768, 1))
    assert_size_stride(primals_145, (768, 768), (768, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (3072, 768), (768, 1))
    assert_size_stride(primals_150, (3072, ), (1, ))
    assert_size_stride(primals_151, (768, 3072), (3072, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, 512), (512, 1))
    assert_size_stride(primals_156, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_157, (256, ), (1, ))
    assert_size_stride(primals_158, (6912, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(primals_161, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_162, (1, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (1, ), (1, ))
    assert_size_stride(primals_165, (256, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_167, (6912, ), (1, ))
    assert_size_stride(primals_168, (256, ), (1, ))
    assert_size_stride(primals_169, (256, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(primals_170, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_171, (1, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (1, ), (1, ))
    assert_size_stride(primals_174, (256, 512), (512, 1))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (256, ), (1, ))
    assert_size_stride(primals_178, (1, 256), (256, 1))
    assert_size_stride(primals_179, (1, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_181, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((7, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [7], out=buf0)
        buf7 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [arange], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_0.run(buf7, 2, grid=grid(2), stream=stream0)
        buf10 = empty_strided_cuda((2, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_1.run(buf10, 131072, grid=grid(131072), stream=stream0)
        buf14 = empty_strided_cuda((3, 1, 1), (1, 1, 1), torch.float32)
        buf14.copy_(reinterpret_tensor(primals_2, (3, 1, 1), (1, 1, 1), 0), False)
        del primals_2
        buf15 = empty_strided_cuda((3, 1, 1), (1, 1, 1), torch.float32)
        buf15.copy_(reinterpret_tensor(primals_3, (3, 1, 1), (1, 1, 1), 0), False)
        del primals_3
        buf342 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_2.run(primals_156, primals_157, buf342, 6912, 256, grid=grid(6912), stream=stream0)
        buf355 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf356 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf357 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mv_3, v_2, mv_4, u_2, sigma_1, weight_1], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3.run(primals_161, primals_162, buf355, buf356, buf357, primals_163, primals_162, 1, 256, grid=grid(1), stream=stream0)
        del primals_163
        buf374 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf375 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf376 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mv_9, v_6, mv_10, u_6, sigma_3, weight_3], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3.run(primals_170, primals_171, buf374, buf375, buf376, primals_172, primals_171, 1, 256, grid=grid(1), stream=stream0)
        del primals_172
        buf361 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_6], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_2.run(primals_165, primals_166, buf361, 6912, 256, grid=grid(6912), stream=stream0)
        buf380 = empty_strided_cuda((512, 2), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_4.run(primals_174, primals_175, buf380, 1024, 128, grid=grid(1024), stream=stream0)
        buf392 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf393 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf394 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mv_15, v_10, mv_16, u_10, sigma_5, weight_5], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_3.run(primals_178, primals_179, buf392, buf393, buf394, primals_180, primals_179, 1, 256, grid=grid(1), stream=stream0)
        del primals_180
        buf1 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_5.run(buf0, buf1, 0, 2, grid=grid(2), stream=stream0)
        buf2 = empty_strided_cuda((2, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_6.run(buf0, buf2, 1, 2, grid=grid(2), stream=stream0)
        buf6 = empty_strided_cuda((2, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rand_2], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_5.run(buf0, buf6, 2, 2, grid=grid(2), stream=stream0)
        buf8 = empty_strided_cuda((2, 256, 256), (65536, 256, 1), torch.int64)
        # Topologically Sorted Source Nodes: [translation_x, add_4, add_5, grid_x_1], Original ATen: [aten.randint, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_randint_7.run(buf0, buf8, 3, 131072, grid=grid(131072), stream=stream0)
        buf9 = empty_strided_cuda((2, 256, 256), (65536, 256, 1), torch.int64)
        # Topologically Sorted Source Nodes: [translation_y, add_6, add_7, grid_y_1], Original ATen: [aten.randint, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_randint_8.run(buf0, buf9, 4, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [mask, setitem], Original ATen: [aten.ones, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_ones_9.run(buf7, buf0, buf10, 5, 6, 32768, grid=grid(32768), stream=stream0)
        del buf0
        buf344 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_10.run(buf342, buf344, primals_158, 1, 6912, grid=grid(1), stream=stream0)
        del primals_158
        buf363 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_10.run(buf361, buf363, primals_167, 1, 6912, grid=grid(1), stream=stream0)
        del primals_167
        buf381 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_11.run(buf380, buf381, 512, 2, grid=grid(512), stream=stream0)
        buf3 = empty_strided_cuda((2, 3, 256, 256), (196608, 65536, 256, 1), torch.float32)
        buf4 = empty_strided_cuda((2, 1, 1, 1, 24), (24, 48, 48, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_mean, sub_1, mul, mul_1, x_1, x_mean_1], Original ATen: [aten.sub, aten.add, aten.mean, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_mul_sub_12.run(primals_1, buf1, buf2, buf3, buf4, 48, 8192, grid=grid(48), stream=stream0)
        del primals_1
        buf345 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_13.run(primals_156, buf344, buf345, 256, 6912, grid=grid(256), stream=stream0)
        buf347 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf348 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u, sigma], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_14.run(buf345, buf347, primals_157, buf348, 1, 256, grid=grid(1), stream=stream0)
        del buf342
        del primals_157
        buf364 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [mv_7], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_13.run(primals_165, buf363, buf364, 256, 6912, grid=grid(256), stream=stream0)
        buf366 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf367 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u_4, sigma_2], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_14.run(buf364, buf366, primals_166, buf367, 1, 256, grid=grid(1), stream=stream0)
        del buf361
        del primals_166
        buf383 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_15.run(buf383, primals_176, 1, 512, grid=grid(1), stream=stream0)
        del primals_176
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_16.run(buf4, buf5, 2, 24, grid=grid(2), stream=stream0)
        del buf4
        buf12 = empty_strided_cuda((2, 3, 256, 256), (196608, 65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, mul_4, add_10], Original ATen: [aten.mul, aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_mul_17.run(buf7, buf8, buf9, buf3, buf5, buf6, buf10, buf12, 393216, grid=grid(393216), stream=stream0)
        del buf3
        buf13 = empty_strided_cuda((2, 3, 224, 224), (150528, 50176, 224, 1), torch.float32)
        buf16 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8], Original ATen: [aten._adaptive_avg_pool2d, aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_div_sub_18.run(buf16, buf12, buf14, buf15, 301056, grid=grid(301056), stream=stream0)
        del buf14
        buf349 = empty_strided_cuda((256, 768, 3, 3), (6912, 9, 3, 1), torch.float32)
        buf350 = empty_strided_cuda((256, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Topologically Sorted Source Nodes: [weight, input_38], Original ATen: [aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_19.run(primals_156, buf348, buf349, buf350, 196608, 9, grid=grid(196608, 9), stream=stream0)
        buf368 = empty_strided_cuda((256, 768, 3, 3), (6912, 9, 3, 1), torch.float32)
        buf369 = empty_strided_cuda((256, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2, input_43], Original ATen: [aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_19.run(primals_165, buf367, buf368, buf369, 196608, 9, grid=grid(196608, 9), stream=stream0)
        buf384 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [mv_13], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_20.run(primals_174, buf383, buf384, 256, 512, grid=grid(256), stream=stream0)
        buf386 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf387 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u_8, sigma_4], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_14.run(buf384, buf386, primals_175, buf387, 1, 256, grid=grid(1), stream=stream0)
        del buf384
        del primals_175
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_4, stride=(32, 32), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (2, 768, 7, 7), (37632, 49, 7, 1))
        buf388 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_21.run(primals_174, buf387, buf388, 131072, grid=grid(131072), stream=stream0)
        buf18 = empty_strided_cuda((2, 50, 768), (38400, 768, 1), torch.float32)
        buf19 = empty_strided_cuda((2, 50, 1), (50, 1, 1), torch.float32)
        buf20 = empty_strided_cuda((2, 50, 1), (50, 1, 100), torch.float32)
        buf22 = reinterpret_tensor(buf20, (2, 50, 1), (50, 1, 1), 0); del buf20  # reuse
        buf23 = empty_strided_cuda((2, 50, 768), (38400, 768, 1), torch.float32)
        buf25 = empty_strided_cuda((50, 2, 1), (1, 50, 100), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, ret, ret_1], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_22.run(buf22, primals_5, buf17, primals_6, primals_7, primals_8, buf18, buf19, buf23, buf25, 100, 768, grid=grid(100), stream=stream0)
        del buf17
        del primals_5
        buf27 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf24 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf28 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_23.run(buf25, buf23, primals_9, primals_10, buf27, buf24, buf28, 100, 768, grid=grid(100), stream=stream0)
        del primals_10
        buf29 = empty_strided_cuda((100, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf28, (100, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 2304), (1, 768), 0), out=buf29)
        buf30 = empty_strided_cuda((3, 50, 2, 768), (76800, 1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf29, primals_11, buf30, 230400, grid=grid(230400), stream=stream0)
        del primals_11
        buf31 = reinterpret_tensor(buf28, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf30, buf31, 76800, grid=grid(76800), stream=stream0)
        buf32 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf30, buf32, 76800, grid=grid(76800), stream=stream0)
        buf33 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf30, buf33, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf31, buf32, buf33, None, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        buf37 = buf34[2]
        buf38 = buf34[3]
        del buf34
        buf39 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf35, buf39, 76800, grid=grid(76800), stream=stream0)
        buf40 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf39, (100, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 768), (1, 768), 0), out=buf40)
        buf44 = reinterpret_tensor(buf39, (50, 2, 768), (1536, 768, 1), 0); del buf39  # reuse
        buf45 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf423 = reinterpret_tensor(buf25, (50, 2, 1), (2, 1, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_16, ret_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29.run(buf23, buf40, primals_14, primals_15, primals_16, buf44, buf45, buf423, 100, 768, grid=grid(100), stream=stream0)
        del primals_16
        buf46 = empty_strided_cuda((100, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, reinterpret_tensor(buf45, (100, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf46)
        del primals_18
        buf47 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_5, sigmoid, input_2], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf46, buf47, 307200, grid=grid(307200), stream=stream0)
        buf48 = reinterpret_tensor(buf45, (100, 768), (768, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 768), (1, 3072), 0), out=buf48)
        buf49 = reinterpret_tensor(buf40, (50, 2, 768), (1536, 768, 1), 0); del buf40  # reuse
        buf53 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf54 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf422 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17, ret_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf49, buf23, primals_14, buf48, primals_20, primals_21, primals_22, buf53, buf54, buf422, 100, 768, grid=grid(100), stream=stream0)
        del primals_14
        del primals_20
        del primals_22
        buf55 = reinterpret_tensor(buf30, (100, 2304), (2304, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf54, (100, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 2304), (1, 768), 0), out=buf55)
        buf56 = reinterpret_tensor(buf29, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf55, primals_23, buf56, 230400, grid=grid(230400), stream=stream0)
        del primals_23
        buf57 = reinterpret_tensor(buf54, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf56, buf57, 76800, grid=grid(76800), stream=stream0)
        buf58 = reinterpret_tensor(buf48, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf56, buf58, 76800, grid=grid(76800), stream=stream0)
        buf59 = reinterpret_tensor(buf23, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf56, buf59, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf57, buf58, buf59, None, True)
        buf61 = buf60[0]
        buf62 = buf60[1]
        buf63 = buf60[2]
        buf64 = buf60[3]
        del buf60
        buf65 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf61, buf65, 76800, grid=grid(76800), stream=stream0)
        buf66 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf65, (100, 768), (768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), out=buf66)
        buf70 = reinterpret_tensor(buf65, (50, 2, 768), (1536, 768, 1), 0); del buf65  # reuse
        buf71 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf421 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18, ret_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf49, buf66, primals_26, primals_27, primals_28, buf70, buf71, buf421, 100, 768, grid=grid(100), stream=stream0)
        del primals_28
        buf72 = reinterpret_tensor(buf47, (100, 3072), (3072, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_30, reinterpret_tensor(buf71, (100, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf72)
        del primals_30
        buf73 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_7, sigmoid_1, input_5], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf72, buf73, 307200, grid=grid(307200), stream=stream0)
        buf74 = reinterpret_tensor(buf71, (100, 768), (768, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf73, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_31, (3072, 768), (1, 3072), 0), out=buf74)
        buf75 = buf49; del buf49  # reuse
        buf79 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf80 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf420 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18, x_19, ret_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf75, buf66, primals_26, buf74, primals_32, primals_33, primals_34, buf79, buf80, buf420, 100, 768, grid=grid(100), stream=stream0)
        del primals_26
        del primals_32
        del primals_34
        buf81 = reinterpret_tensor(buf56, (100, 2304), (2304, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf80, (100, 768), (768, 1), 0), reinterpret_tensor(primals_36, (768, 2304), (1, 768), 0), out=buf81)
        buf82 = reinterpret_tensor(buf55, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf81, primals_35, buf82, 230400, grid=grid(230400), stream=stream0)
        del primals_35
        buf83 = reinterpret_tensor(buf80, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf82, buf83, 76800, grid=grid(76800), stream=stream0)
        buf84 = reinterpret_tensor(buf74, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf82, buf84, 76800, grid=grid(76800), stream=stream0)
        buf85 = reinterpret_tensor(buf66, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf82, buf85, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf86 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf83, buf84, buf85, None, True)
        buf87 = buf86[0]
        buf88 = buf86[1]
        buf89 = buf86[2]
        buf90 = buf86[3]
        del buf86
        buf91 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf87, buf91, 76800, grid=grid(76800), stream=stream0)
        buf92 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf91, (100, 768), (768, 1), 0), reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), out=buf92)
        buf96 = reinterpret_tensor(buf91, (50, 2, 768), (1536, 768, 1), 0); del buf91  # reuse
        buf97 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf419 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20, ret_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf75, buf92, primals_38, primals_39, primals_40, buf96, buf97, buf419, 100, 768, grid=grid(100), stream=stream0)
        del primals_40
        buf98 = reinterpret_tensor(buf73, (100, 3072), (3072, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, reinterpret_tensor(buf97, (100, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf98)
        del primals_42
        buf99 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_9, sigmoid_2, input_8], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf98, buf99, 307200, grid=grid(307200), stream=stream0)
        buf100 = reinterpret_tensor(buf97, (100, 768), (768, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf99, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 768), (1, 3072), 0), out=buf100)
        buf101 = buf75; del buf75  # reuse
        buf105 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf106 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf418 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20, x_21, ret_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf101, buf92, primals_38, buf100, primals_44, primals_45, primals_46, buf105, buf106, buf418, 100, 768, grid=grid(100), stream=stream0)
        del primals_38
        del primals_44
        del primals_46
        buf107 = reinterpret_tensor(buf82, (100, 2304), (2304, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf106, (100, 768), (768, 1), 0), reinterpret_tensor(primals_48, (768, 2304), (1, 768), 0), out=buf107)
        buf108 = reinterpret_tensor(buf81, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf107, primals_47, buf108, 230400, grid=grid(230400), stream=stream0)
        del primals_47
        buf109 = reinterpret_tensor(buf106, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf108, buf109, 76800, grid=grid(76800), stream=stream0)
        buf110 = reinterpret_tensor(buf92, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf108, buf110, 76800, grid=grid(76800), stream=stream0)
        buf111 = reinterpret_tensor(buf100, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf108, buf111, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf112 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf109, buf110, buf111, None, True)
        buf113 = buf112[0]
        buf114 = buf112[1]
        buf115 = buf112[2]
        buf116 = buf112[3]
        del buf112
        buf117 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf113, buf117, 76800, grid=grid(76800), stream=stream0)
        buf118 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf117, (100, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 768), (1, 768), 0), out=buf118)
        buf122 = reinterpret_tensor(buf117, (50, 2, 768), (1536, 768, 1), 0); del buf117  # reuse
        buf123 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf417 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, ret_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf101, buf118, primals_50, primals_51, primals_52, buf122, buf123, buf417, 100, 768, grid=grid(100), stream=stream0)
        del primals_52
        buf124 = reinterpret_tensor(buf99, (100, 3072), (3072, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_54, reinterpret_tensor(buf123, (100, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf124)
        del primals_54
        buf125 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_11, sigmoid_3, input_11], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf124, buf125, 307200, grid=grid(307200), stream=stream0)
        buf126 = reinterpret_tensor(buf123, (100, 768), (768, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf125, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_55, (3072, 768), (1, 3072), 0), out=buf126)
        buf127 = empty_strided_cuda((50, 2, 768), (768, 38400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_34.run(buf101, buf118, primals_50, buf126, primals_56, buf127, 76800, grid=grid(76800), stream=stream0)
        del primals_50
        del primals_56
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(reinterpret_tensor(buf127, (2, 768, 7, 7), (38400, 1, 5376, 768), 768), buf350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (2, 256, 7, 7), (12544, 1, 1792, 256))
        del buf350
        buf129 = empty_strided_cuda((50, 2, 1), (1, 50, 100), torch.float32)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_35.run(buf127, buf129, 100, 768, grid=grid(100), stream=stream0)
        buf131 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf128 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf132 = reinterpret_tensor(buf126, (50, 2, 768), (1536, 768, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_23.run(buf129, buf127, primals_57, primals_58, buf131, buf128, buf132, 100, 768, grid=grid(100), stream=stream0)
        del primals_58
        buf352 = empty_strided_cuda((2, 256, 9, 9), (20736, 1, 2304, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, pad_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36.run(buf351, primals_159, buf352, 41472, grid=grid(41472), stream=stream0)
        buf401 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.bool)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37.run(buf351, primals_159, buf401, 25088, grid=grid(25088), stream=stream0)
        del buf351
        del primals_159
        buf133 = reinterpret_tensor(buf108, (100, 2304), (2304, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf132, (100, 768), (768, 1), 0), reinterpret_tensor(primals_60, (768, 2304), (1, 768), 0), out=buf133)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf353, (2, 256, 6, 6), (9216, 1, 1536, 256))
        buf134 = reinterpret_tensor(buf107, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf133, primals_59, buf134, 230400, grid=grid(230400), stream=stream0)
        del primals_59
        buf358 = empty_strided_cuda((2, 256, 6, 6), (9216, 6, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf353, buf358, 3072, 6, grid=grid(3072, 6), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, buf357, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (2, 1, 3, 3), (9, 1, 3, 1))
        buf135 = reinterpret_tensor(buf132, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf134, buf135, 76800, grid=grid(76800), stream=stream0)
        buf136 = reinterpret_tensor(buf118, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf134, buf136, 76800, grid=grid(76800), stream=stream0)
        buf137 = reinterpret_tensor(buf101, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf134, buf137, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf138 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf135, buf136, buf137, None, True)
        buf139 = buf138[0]
        buf140 = buf138[1]
        buf141 = buf138[2]
        buf142 = buf138[3]
        del buf138
        buf143 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf139, buf143, 76800, grid=grid(76800), stream=stream0)
        buf144 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf143, (100, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 768), (1, 768), 0), out=buf144)
        buf148 = reinterpret_tensor(buf143, (50, 2, 768), (1536, 768, 1), 0); del buf143  # reuse
        buf149 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf416 = reinterpret_tensor(buf129, (50, 2, 1), (2, 1, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_24, ret_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29.run(buf127, buf144, primals_62, primals_63, primals_64, buf148, buf149, buf416, 100, 768, grid=grid(100), stream=stream0)
        del primals_64
        buf150 = reinterpret_tensor(buf125, (100, 3072), (3072, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_66, reinterpret_tensor(buf149, (100, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf150)
        del primals_66
        buf151 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_13, sigmoid_4, input_14], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf150, buf151, 307200, grid=grid(307200), stream=stream0)
        buf152 = reinterpret_tensor(buf149, (100, 768), (768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf151, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_67, (3072, 768), (1, 3072), 0), out=buf152)
        buf153 = reinterpret_tensor(buf144, (50, 2, 768), (1536, 768, 1), 0); del buf144  # reuse
        buf157 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf158 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf415 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24, x_25, ret_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf153, buf127, primals_62, buf152, primals_68, primals_69, primals_70, buf157, buf158, buf415, 100, 768, grid=grid(100), stream=stream0)
        del primals_62
        del primals_68
        del primals_70
        buf159 = reinterpret_tensor(buf134, (100, 2304), (2304, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf158, (100, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 2304), (1, 768), 0), out=buf159)
        buf160 = reinterpret_tensor(buf133, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf159, primals_71, buf160, 230400, grid=grid(230400), stream=stream0)
        del primals_71
        buf161 = reinterpret_tensor(buf158, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf160, buf161, 76800, grid=grid(76800), stream=stream0)
        buf162 = reinterpret_tensor(buf152, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf160, buf162, 76800, grid=grid(76800), stream=stream0)
        buf163 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf160, buf163, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf164 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf161, buf162, buf163, None, True)
        buf165 = buf164[0]
        buf166 = buf164[1]
        buf167 = buf164[2]
        buf168 = buf164[3]
        del buf164
        buf169 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf165, buf169, 76800, grid=grid(76800), stream=stream0)
        buf170 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf169, (100, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), out=buf170)
        buf174 = reinterpret_tensor(buf169, (50, 2, 768), (1536, 768, 1), 0); del buf169  # reuse
        buf175 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf414 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26, ret_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf153, buf170, primals_74, primals_75, primals_76, buf174, buf175, buf414, 100, 768, grid=grid(100), stream=stream0)
        del primals_76
        buf176 = reinterpret_tensor(buf151, (100, 3072), (3072, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_78, reinterpret_tensor(buf175, (100, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf176)
        del primals_78
        buf177 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_15, sigmoid_5, input_17], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf176, buf177, 307200, grid=grid(307200), stream=stream0)
        buf178 = reinterpret_tensor(buf175, (100, 768), (768, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf177, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 768), (1, 3072), 0), out=buf178)
        buf179 = buf153; del buf153  # reuse
        buf183 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf184 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf413 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26, x_27, ret_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf179, buf170, primals_74, buf178, primals_80, primals_81, primals_82, buf183, buf184, buf413, 100, 768, grid=grid(100), stream=stream0)
        del primals_74
        del primals_80
        del primals_82
        buf185 = reinterpret_tensor(buf160, (100, 2304), (2304, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf184, (100, 768), (768, 1), 0), reinterpret_tensor(primals_84, (768, 2304), (1, 768), 0), out=buf185)
        buf186 = reinterpret_tensor(buf159, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf185, primals_83, buf186, 230400, grid=grid(230400), stream=stream0)
        del primals_83
        buf187 = reinterpret_tensor(buf184, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf186, buf187, 76800, grid=grid(76800), stream=stream0)
        buf188 = reinterpret_tensor(buf178, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf186, buf188, 76800, grid=grid(76800), stream=stream0)
        buf189 = reinterpret_tensor(buf170, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf186, buf189, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf190 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf187, buf188, buf189, None, True)
        buf191 = buf190[0]
        buf192 = buf190[1]
        buf193 = buf190[2]
        buf194 = buf190[3]
        del buf190
        buf195 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf191, buf195, 76800, grid=grid(76800), stream=stream0)
        buf196 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf195, (100, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), out=buf196)
        buf200 = reinterpret_tensor(buf195, (50, 2, 768), (1536, 768, 1), 0); del buf195  # reuse
        buf201 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf412 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, ret_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf179, buf196, primals_86, primals_87, primals_88, buf200, buf201, buf412, 100, 768, grid=grid(100), stream=stream0)
        del primals_88
        buf202 = reinterpret_tensor(buf177, (100, 3072), (3072, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_90, reinterpret_tensor(buf201, (100, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf202)
        del primals_90
        buf203 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_17, sigmoid_6, input_20], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf202, buf203, 307200, grid=grid(307200), stream=stream0)
        buf204 = reinterpret_tensor(buf201, (100, 768), (768, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf203, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (1, 3072), 0), out=buf204)
        buf205 = buf179; del buf179  # reuse
        buf209 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf210 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf411 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, x_29, ret_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf205, buf196, primals_86, buf204, primals_92, primals_93, primals_94, buf209, buf210, buf411, 100, 768, grid=grid(100), stream=stream0)
        del primals_86
        del primals_92
        del primals_94
        buf211 = reinterpret_tensor(buf186, (100, 2304), (2304, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf210, (100, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 2304), (1, 768), 0), out=buf211)
        buf212 = reinterpret_tensor(buf185, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf211, primals_95, buf212, 230400, grid=grid(230400), stream=stream0)
        del primals_95
        buf213 = reinterpret_tensor(buf210, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf212, buf213, 76800, grid=grid(76800), stream=stream0)
        buf214 = reinterpret_tensor(buf204, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf212, buf214, 76800, grid=grid(76800), stream=stream0)
        buf215 = reinterpret_tensor(buf196, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf212, buf215, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf216 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf213, buf214, buf215, None, True)
        buf217 = buf216[0]
        buf218 = buf216[1]
        buf219 = buf216[2]
        buf220 = buf216[3]
        del buf216
        buf221 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf217, buf221, 76800, grid=grid(76800), stream=stream0)
        buf222 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf221, (100, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 768), (1, 768), 0), out=buf222)
        buf226 = reinterpret_tensor(buf221, (50, 2, 768), (1536, 768, 1), 0); del buf221  # reuse
        buf227 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf410 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, ret_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf205, buf222, primals_98, primals_99, primals_100, buf226, buf227, buf410, 100, 768, grid=grid(100), stream=stream0)
        del primals_100
        buf228 = reinterpret_tensor(buf203, (100, 3072), (3072, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, reinterpret_tensor(buf227, (100, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf228)
        del primals_102
        buf229 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_19, sigmoid_7, input_23], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf228, buf229, 307200, grid=grid(307200), stream=stream0)
        buf230 = reinterpret_tensor(buf227, (100, 768), (768, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf229, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), out=buf230)
        buf231 = empty_strided_cuda((50, 2, 768), (768, 38400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_34.run(buf205, buf222, primals_98, buf230, primals_104, buf231, 76800, grid=grid(76800), stream=stream0)
        del primals_104
        del primals_98
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(reinterpret_tensor(buf231, (2, 768, 7, 7), (38400, 1, 5376, 768), 768), buf369, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (2, 256, 7, 7), (12544, 1, 1792, 256))
        del buf369
        buf233 = empty_strided_cuda((50, 2, 1), (1, 50, 100), torch.float32)
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_35.run(buf231, buf233, 100, 768, grid=grid(100), stream=stream0)
        buf235 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf232 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        buf236 = reinterpret_tensor(buf230, (50, 2, 768), (1536, 768, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_23.run(buf233, buf231, primals_105, primals_106, buf235, buf232, buf236, 100, 768, grid=grid(100), stream=stream0)
        del primals_106
        buf371 = empty_strided_cuda((2, 256, 9, 9), (20736, 1, 2304, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, pad_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convolution_leaky_relu_36.run(buf370, primals_168, buf371, 41472, grid=grid(41472), stream=stream0)
        buf400 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.bool)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_37.run(buf370, primals_168, buf400, 25088, grid=grid(25088), stream=stream0)
        del buf370
        del primals_168
        buf237 = reinterpret_tensor(buf212, (100, 2304), (2304, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf236, (100, 768), (768, 1), 0), reinterpret_tensor(primals_108, (768, 2304), (1, 768), 0), out=buf237)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf372, (2, 256, 6, 6), (9216, 1, 1536, 256))
        buf238 = reinterpret_tensor(buf211, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf237, primals_107, buf238, 230400, grid=grid(230400), stream=stream0)
        del primals_107
        buf377 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf372, buf377, 3072, 6, grid=grid(3072, 6), stream=stream0)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, buf376, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (2, 1, 3, 3), (9, 1, 3, 1))
        del buf377
        buf239 = reinterpret_tensor(buf236, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf238, buf239, 76800, grid=grid(76800), stream=stream0)
        buf240 = reinterpret_tensor(buf222, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf238, buf240, 76800, grid=grid(76800), stream=stream0)
        buf241 = reinterpret_tensor(buf205, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf238, buf241, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf242 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf239, buf240, buf241, None, True)
        buf243 = buf242[0]
        buf244 = buf242[1]
        buf245 = buf242[2]
        buf246 = buf242[3]
        del buf242
        buf247 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf243, buf247, 76800, grid=grid(76800), stream=stream0)
        buf248 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf247, (100, 768), (768, 1), 0), reinterpret_tensor(primals_109, (768, 768), (1, 768), 0), out=buf248)
        buf252 = reinterpret_tensor(buf247, (50, 2, 768), (1536, 768, 1), 0); del buf247  # reuse
        buf253 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf409 = reinterpret_tensor(buf233, (50, 2, 1), (2, 1, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_32, ret_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_29.run(buf231, buf248, primals_110, primals_111, primals_112, buf252, buf253, buf409, 100, 768, grid=grid(100), stream=stream0)
        del primals_112
        buf254 = reinterpret_tensor(buf229, (100, 3072), (3072, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_114, reinterpret_tensor(buf253, (100, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf254)
        del primals_114
        buf255 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_21, sigmoid_8, input_26], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf254, buf255, 307200, grid=grid(307200), stream=stream0)
        buf256 = reinterpret_tensor(buf253, (100, 768), (768, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf255, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 768), (1, 3072), 0), out=buf256)
        buf257 = reinterpret_tensor(buf248, (50, 2, 768), (1536, 768, 1), 0); del buf248  # reuse
        buf261 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf262 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf408 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_32, x_33, ret_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf257, buf231, primals_110, buf256, primals_116, primals_117, primals_118, buf261, buf262, buf408, 100, 768, grid=grid(100), stream=stream0)
        del primals_110
        del primals_116
        del primals_118
        buf263 = reinterpret_tensor(buf238, (100, 2304), (2304, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf262, (100, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 2304), (1, 768), 0), out=buf263)
        buf264 = reinterpret_tensor(buf237, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf263, primals_119, buf264, 230400, grid=grid(230400), stream=stream0)
        del primals_119
        buf265 = reinterpret_tensor(buf262, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf264, buf265, 76800, grid=grid(76800), stream=stream0)
        buf266 = reinterpret_tensor(buf256, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf264, buf266, 76800, grid=grid(76800), stream=stream0)
        buf267 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf264, buf267, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf268 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf265, buf266, buf267, None, True)
        buf269 = buf268[0]
        buf270 = buf268[1]
        buf271 = buf268[2]
        buf272 = buf268[3]
        del buf268
        buf273 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf269, buf273, 76800, grid=grid(76800), stream=stream0)
        buf274 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf273, (100, 768), (768, 1), 0), reinterpret_tensor(primals_121, (768, 768), (1, 768), 0), out=buf274)
        buf278 = reinterpret_tensor(buf273, (50, 2, 768), (1536, 768, 1), 0); del buf273  # reuse
        buf279 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf407 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34, ret_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf257, buf274, primals_122, primals_123, primals_124, buf278, buf279, buf407, 100, 768, grid=grid(100), stream=stream0)
        del primals_124
        buf280 = reinterpret_tensor(buf255, (100, 3072), (3072, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, reinterpret_tensor(buf279, (100, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf280)
        del primals_126
        buf281 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_23, sigmoid_9, input_29], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf280, buf281, 307200, grid=grid(307200), stream=stream0)
        buf282 = reinterpret_tensor(buf279, (100, 768), (768, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf281, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_127, (3072, 768), (1, 3072), 0), out=buf282)
        buf283 = buf257; del buf257  # reuse
        buf287 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf288 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf406 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34, x_35, ret_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf283, buf274, primals_122, buf282, primals_128, primals_129, primals_130, buf287, buf288, buf406, 100, 768, grid=grid(100), stream=stream0)
        del primals_122
        del primals_128
        del primals_130
        buf289 = reinterpret_tensor(buf264, (100, 2304), (2304, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf288, (100, 768), (768, 1), 0), reinterpret_tensor(primals_132, (768, 2304), (1, 768), 0), out=buf289)
        buf290 = reinterpret_tensor(buf263, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf289, primals_131, buf290, 230400, grid=grid(230400), stream=stream0)
        del primals_131
        buf291 = reinterpret_tensor(buf288, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf290, buf291, 76800, grid=grid(76800), stream=stream0)
        buf292 = reinterpret_tensor(buf282, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf290, buf292, 76800, grid=grid(76800), stream=stream0)
        buf293 = reinterpret_tensor(buf274, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf290, buf293, 76800, grid=grid(76800), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf294 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf291, buf292, buf293, None, True)
        buf295 = buf294[0]
        buf296 = buf294[1]
        buf297 = buf294[2]
        buf298 = buf294[3]
        del buf294
        buf299 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf295, buf299, 76800, grid=grid(76800), stream=stream0)
        buf300 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf299, (100, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 768), (1, 768), 0), out=buf300)
        buf304 = reinterpret_tensor(buf299, (50, 2, 768), (1536, 768, 1), 0); del buf299  # reuse
        buf305 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf405 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, ret_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf283, buf300, primals_134, primals_135, primals_136, buf304, buf305, buf405, 100, 768, grid=grid(100), stream=stream0)
        del primals_136
        buf306 = reinterpret_tensor(buf281, (100, 3072), (3072, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_138, reinterpret_tensor(buf305, (100, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf306)
        del primals_138
        buf307 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_25, sigmoid_10, input_32], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf306, buf307, 307200, grid=grid(307200), stream=stream0)
        buf308 = reinterpret_tensor(buf305, (100, 768), (768, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf307, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_139, (3072, 768), (1, 3072), 0), out=buf308)
        buf309 = buf283; del buf283  # reuse
        buf313 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf314 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf404 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36, x_37, ret_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_33.run(buf309, buf300, primals_134, buf308, primals_140, primals_141, primals_142, buf313, buf314, buf404, 100, 768, grid=grid(100), stream=stream0)
        del primals_134
        del primals_140
        del primals_142
        buf315 = reinterpret_tensor(buf290, (100, 2304), (2304, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf314, (100, 768), (768, 1), 0), reinterpret_tensor(primals_144, (768, 2304), (1, 768), 0), out=buf315)
        buf316 = reinterpret_tensor(buf289, (3, 50, 2, 768), (76800, 1536, 768, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf315, primals_143, buf316, 230400, grid=grid(230400), stream=stream0)
        del buf315
        del primals_143
        buf317 = reinterpret_tensor(buf314, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_25.run(buf316, buf317, 76800, grid=grid(76800), stream=stream0)
        buf318 = reinterpret_tensor(buf308, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf316, buf318, 76800, grid=grid(76800), stream=stream0)
        buf319 = reinterpret_tensor(buf300, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_27.run(buf316, buf319, 76800, grid=grid(76800), stream=stream0)
        del buf316
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf320 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf317, buf318, buf319, None, True)
        buf321 = buf320[0]
        buf322 = buf320[1]
        buf323 = buf320[2]
        buf324 = buf320[3]
        del buf320
        buf325 = empty_strided_cuda((50, 2, 12, 64), (1536, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf321, buf325, 76800, grid=grid(76800), stream=stream0)
        buf326 = empty_strided_cuda((100, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf325, (100, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 768), (1, 768), 0), out=buf326)
        buf330 = reinterpret_tensor(buf325, (50, 2, 768), (1536, 768, 1), 0); del buf325  # reuse
        buf331 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        buf403 = empty_strided_cuda((50, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_38, ret_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf309, buf326, primals_146, primals_147, primals_148, buf330, buf331, buf403, 100, 768, grid=grid(100), stream=stream0)
        del primals_148
        buf332 = reinterpret_tensor(buf307, (100, 3072), (3072, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, reinterpret_tensor(buf331, (100, 768), (768, 1), 0), reinterpret_tensor(primals_149, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf332)
        del primals_150
        buf333 = empty_strided_cuda((50, 2, 3072), (6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_27, sigmoid_11, input_35], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf332, buf333, 307200, grid=grid(307200), stream=stream0)
        buf334 = reinterpret_tensor(buf331, (100, 768), (768, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf333, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 768), (1, 3072), 0), out=buf334)
        del buf333
        buf335 = empty_strided_cuda((2, 768), (768, 1), torch.float32)
        buf339 = buf335; del buf335  # reuse
        buf340 = empty_strided_cuda((2, 768), (768, 1), torch.float32)
        buf402 = reinterpret_tensor(buf5, (2, 1), (1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_39.run(buf339, buf309, buf326, primals_146, buf334, primals_152, primals_153, primals_154, buf340, buf402, 2, 768, grid=grid(2), stream=stream0)
        del buf309
        del buf326
        del buf334
        del primals_146
        del primals_152
        del primals_154
        buf341 = reinterpret_tensor(buf380, (2, 512), (512, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [ret_25, x_41], Original ATen: [aten.native_layer_norm, aten.mm]
        extern_kernels.mm(buf340, primals_155, out=buf341)
        del buf340
        buf389 = empty_strided_cuda((2, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.addmm]
        extern_kernels.mm(buf341, reinterpret_tensor(buf388, (512, 256), (1, 512), 0), out=buf389)
        buf390 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.addmm, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_leaky_relu_40.run(buf390, primals_177, 512, grid=grid(512), stream=stream0)
        del primals_177
        buf396 = empty_strided_cuda((2, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, buf390, reinterpret_tensor(buf394, (256, 1), (1, 256), 0), alpha=1, beta=1, out=buf396)
        del primals_181
        buf360 = buf359; del buf359  # reuse
        buf397 = empty_strided_cuda((2, ), (1, ), torch.float32)
        buf379 = buf378; del buf378  # reuse
        buf399 = reinterpret_tensor(buf397, (2, 1), (1, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_46, loss_, mean_2, loss, loss__2, mean_3, loss_1, loss__4, loss_2, loss_3], Original ATen: [aten.convolution, aten.binary_cross_entropy_with_logits, aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_41.run(buf360, buf379, buf399, primals_164, primals_173, buf396, 2, 9, grid=grid(2), stream=stream0)
        del primals_164
        del primals_173
    return (buf399, buf349, buf357, buf368, buf376, buf388, buf394, primals_4, primals_6, primals_7, primals_8, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, buf2, buf6, reinterpret_tensor(buf7, (2, 1, 1), (1, 1, 1), 0), buf8, buf9, buf10, buf12, buf15, buf16, buf18, buf19, buf22, buf24, buf27, buf31, buf32, buf33, buf35, buf36, buf37, buf38, buf44, buf46, buf53, buf57, buf58, buf59, buf61, buf62, buf63, buf64, buf70, buf72, buf79, buf83, buf84, buf85, buf87, buf88, buf89, buf90, buf96, buf98, buf105, buf109, buf110, buf111, buf113, buf114, buf115, buf116, buf122, buf124, buf127, buf128, buf131, buf135, buf136, buf137, buf139, buf140, buf141, buf142, buf148, buf150, buf157, buf161, buf162, buf163, buf165, buf166, buf167, buf168, buf174, buf176, buf183, buf187, buf188, buf189, buf191, buf192, buf193, buf194, buf200, buf202, buf209, buf213, buf214, buf215, buf217, buf218, buf219, buf220, buf226, buf228, buf231, buf232, buf235, buf239, buf240, buf241, buf243, buf244, buf245, buf246, buf252, buf254, buf261, buf265, buf266, buf267, buf269, buf270, buf271, buf272, buf278, buf280, buf287, buf291, buf292, buf293, buf295, buf296, buf297, buf298, buf304, buf306, buf313, buf317, buf318, buf319, buf321, buf322, buf323, buf324, buf330, buf332, buf339, buf341, buf344, buf347, buf348, buf349, buf352, buf353, buf355, buf356, buf357, buf360, buf363, buf366, buf367, buf368, buf371, buf372, buf374, buf375, buf376, buf379, buf383, buf386, buf387, buf390, buf392, buf393, buf396, buf394, buf388, buf400, buf401, reinterpret_tensor(primals_155, (512, 768), (1, 512), 0), buf402, primals_151, primals_149, buf403, primals_145, primals_144, buf404, primals_139, primals_137, buf405, primals_133, primals_132, buf406, primals_127, primals_125, buf407, primals_121, primals_120, buf408, primals_115, primals_113, buf409, primals_109, primals_108, primals_103, primals_101, buf410, primals_97, primals_96, buf411, primals_91, primals_89, buf412, primals_85, primals_84, buf413, primals_79, primals_77, buf414, primals_73, primals_72, buf415, primals_67, primals_65, buf416, primals_61, primals_60, primals_55, primals_53, buf417, primals_49, primals_48, buf418, primals_43, primals_41, buf419, primals_37, primals_36, buf420, primals_31, primals_29, buf421, primals_25, primals_24, buf422, primals_19, primals_17, buf423, primals_13, primals_12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((50, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((6912, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((256, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((6912, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
