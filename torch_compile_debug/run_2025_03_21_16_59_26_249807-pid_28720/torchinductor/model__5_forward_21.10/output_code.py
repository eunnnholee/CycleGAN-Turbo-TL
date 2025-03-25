# AOT ID: ['5_forward']
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


# kernel path: /tmp/torchinductor_elicer/vs/cvsljsjzgul3s5o5fmvkfwlnvhag2di5gbi3mruc62lkkfptss2h.py
# Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   rand => inductor_lookup_seed_default, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([2, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
triton_poi_fused_rand_0 = async_compile.triton('triton_poi_fused_rand_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_rand_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/le/clenghno47ynlvjlt763hfv7zdjzhlgnq5ew3skopgulwu3bfpxj.py
# Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   rand_1 => inductor_lookup_seed_default_1, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([2, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
triton_poi_fused_rand_1 = async_compile.triton('triton_poi_fused_rand_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_rand_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_rand_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/36/c362xon2zuph637rysktbx2izig4d3q7kytdldhexh4kconckn7n.py
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
triton_red_fused_add_mean_mul_sub_2 = async_compile.triton('triton_red_fused_add_mean_mul_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mean_mul_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/3t/c3tfvqtgj7unjxoa3qwlp6ed6ud5rka5coy7wfez7v2eymoqmzhp.py
# Topologically Sorted Source Nodes: [x_mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3], True), kwargs = {})
triton_per_fused_mean_3 = async_compile.triton('triton_per_fused_mean_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/k7/ck7vrh46xpzloll56z7bsampid6z72jng5awjy72jc6nw57x74bn.py
# Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   mask => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 256, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_4 = async_compile.triton('triton_poi_fused_ones_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/no/cnookbr2mc2exjlbjj3oiqfbr6hbqvoj6ejg47b5vl3hkppn2ncr.py
# Topologically Sorted Source Nodes: [mask, setitem], Original ATen: [aten.ones, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   mask => full_default
#   setitem => full_default_1, index_put
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 256, 256], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%expand_3, %clamp_max_2, %clamp_max_3], %full_default_1), kwargs = {})
triton_poi_fused_index_put_lift_fresh_ones_5 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_ones_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'load_seed_offset': 'i32', 'load_seed_offset1': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_ones_5', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_ones_5(in_ptr0, out_ptr0, load_seed_offset, load_seed_offset1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 16384
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 257, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = x1
    tmp6 = tmp5 + tmp4
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp6 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = tl.full([1], 255, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.load(in_ptr0 + load_seed_offset1)
    tmp13 = triton_helpers.randint64(tmp12, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp14 = x0
    tmp15 = tmp14 + tmp13
    tmp16 = tmp15 - tmp7
    tmp17 = triton_helpers.maximum(tmp16, tmp2)
    tmp18 = triton_helpers.minimum(tmp17, tmp10)
    tmp19 = 0.0
    tl.store(out_ptr0 + (tmp18 + 256*tmp11 + 65536*x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/nz/cnzeek7wmhyqsrnxdrl5shhhwi23zsbo7q2lirddqejgxdriirbe.py
# Topologically Sorted Source Nodes: [contiguous, getitem_3, x_4, x_5, mul_4, add_10], Original ATen: [aten.clone, aten.index, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_10 => add_10
#   contiguous => clone
#   getitem_3 => index
#   mul_4 => mul_4
#   x_4 => mul_3
#   x_5 => clone_1
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%clone, [%expand, %clamp_max, %clamp_max_1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %unsqueeze_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_3,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_1, 0.5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.5), kwargs = {})
triton_poi_fused_add_clone_index_mul_6 = async_compile.triton('triton_poi_fused_add_clone_index_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'load_seed_offset': 'i32', 'load_seed_offset1': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_index_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_index_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, load_seed_offset, load_seed_offset1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex // 196608
    x2 = ((xindex // 768) % 256)
    x1 = ((xindex // 3) % 256)
    x0 = (xindex % 3)
    x5 = xindex
    x4 = xindex // 3
    tmp44 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x3
    tmp2 = tl.full([1], -32, tl.int64)
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = x2
    tmp6 = tmp5 + tmp4
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = tl.load(in_ptr0 + load_seed_offset1)
    tmp14 = triton_helpers.randint64(tmp13, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp15 = x1
    tmp16 = tmp15 + tmp14
    tmp17 = tmp16 + tmp7
    tmp18 = triton_helpers.maximum(tmp17, tmp9)
    tmp19 = triton_helpers.minimum(tmp18, tmp11)
    tmp20 = (-1) + tmp12
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp21 >= tmp9
    tmp23 = tl.full([1], 256, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = (-1) + tmp19
    tmp26 = tmp25.to(tl.int32)
    tmp27 = tmp26 >= tmp9
    tmp28 = tmp26 < tmp23
    tmp29 = tmp22 & tmp24
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp28
    tmp32 = tl.load(in_ptr1 + ((-257) + tmp19 + 256*tmp12 + 65536*x0 + 196608*x3), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr2 + (x3), tmp31, eviction_policy='evict_last', other=0.0)
    tmp34 = 196608.0
    tmp35 = tmp33 / tmp34
    tmp36 = tmp32 - tmp35
    tmp37 = tl.load(in_ptr3 + (x3), tmp31, eviction_policy='evict_last', other=0.0)
    tmp38 = 0.5
    tmp39 = tmp37 + tmp38
    tmp40 = tmp36 * tmp39
    tmp41 = tmp40 + tmp35
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp31, tmp41, tmp42)
    tmp45 = tmp43 * tmp44
    tmp46 = 0.5
    tmp47 = tmp45 * tmp46
    tmp48 = tmp47 + tmp46
    tl.store(in_out_ptr0 + (x5), tmp48, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/x5/cx5ukznqn2rbwsknlyfbnk5f2wurnst7wcmvhlylzuyh4cinkq6x.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul_4, add_10, x_6], Original ATen: [aten.mul, aten.clone, aten.add, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   add_10 => add_10
#   mul_4 => mul_4
#   x_4 => mul_3
#   x_5 => clone_1
#   x_6 => _adaptive_avg_pool2d
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %unsqueeze_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_3,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_1, 0.5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.5), kwargs = {})
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%add_10, [224, 224]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_add_clone_mul_7 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_add_clone_mul_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_add_clone_mul_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_add_clone_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 672) % 224)
    x1 = ((xindex // 3) % 224)
    x0 = (xindex % 3)
    x3 = xindex // 150528
    x6 = xindex
    tmp0 = (8*x2) // 7
    tmp1 = (479 + 256*x2) // 224
    tmp2 = tmp0 < tmp1
    tmp3 = (8*x1) // 7
    tmp4 = (479 + 256*x1) // 224
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp6 & xmask, other=0.0)
    tmp8 = 1 + ((8*x1) // 7)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (3 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp10 & xmask, other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((8*x1) // 7)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (6 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp15 & xmask, other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 1 + ((8*x2) // 7)
    tmp19 = tmp18 < tmp1
    tmp20 = tmp19 & tmp5
    tmp21 = tl.load(in_ptr0 + (768 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp20 & xmask, other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp19 & tmp9
    tmp24 = tl.load(in_ptr0 + (771 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp22
    tmp26 = tmp19 & tmp14
    tmp27 = tl.load(in_ptr0 + (774 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp26 & xmask, other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = 2 + ((8*x2) // 7)
    tmp30 = tmp29 < tmp1
    tmp31 = tmp30 & tmp5
    tmp32 = tl.load(in_ptr0 + (1536 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp31 & xmask, other=0.0)
    tmp33 = tmp32 + tmp28
    tmp34 = tmp30 & tmp9
    tmp35 = tl.load(in_ptr0 + (1539 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp34 & xmask, other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp14
    tmp38 = tl.load(in_ptr0 + (1542 + x0 + 3*((8*x1) // 7) + 768*((8*x2) // 7) + 196608*x3), tmp37 & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp75, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5w/c5wgbkyiikc2wyr54fdeqsf666qalvqdvznynbkjskd7rdv5z3o4.py
# Topologically Sorted Source Nodes: [x_7, x_8, x_9], Original ATen: [aten.sub, aten.div, aten.convolution]
# Source node to ATen node mapping:
#   x_7 => sub_5
#   x_8 => div
#   x_9 => convolution
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_adaptive_avg_pool2d, %device_put), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %device_put_1), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_4, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_sub_8 = async_compile.triton('triton_poi_fused_convolution_div_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_sub_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3*x2 + 150528*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (x2 + 50176*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3b/c3btxui47a4thttb2w5emrh5sc2hxiq27owh3t4z5sz6yy7zisep.py
# Topologically Sorted Source Nodes: [x_12, x_13, ret], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret => var_mean
#   x_12 => cat
#   x_13 => add_12
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_11, %permute_2], 1), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_6), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_cat_native_layer_norm_9 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 600
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = ((xindex // 6) % 50)
    x0 = (xindex % 6)
    x2 = xindex // 300
    x5 = (xindex % 300)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp15 = tl.load(in_ptr2 + (r3 + 128*x5), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + 128*x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 0.0
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 50, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr1 + (49*r3 + 6272*x0 + 37632*x2 + ((-1) + x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.where(tmp4, tmp9, tmp13)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp18, xmask)
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/uk/cukq73ftqqolfxq2chpvg47ys76ev4skzhdv7bepai6qsbd5xfnk.py
# Topologically Sorted Source Nodes: [x_12, x_13, ret], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret => var_mean
#   x_12 => cat
#   x_13 => add_12
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_11, %permute_2], 1), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_6), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_cat_native_layer_norm_10 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 6*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 6*x0), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 6*x0), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/rx/crxrgv7ddkc6wx6qy6vlwvxci33fczobip4we4hns7yah6ng7ybf.py
# Topologically Sorted Source Nodes: [x_12, x_13, ret, ret_1], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret => add_13, add_14, mul_5, mul_6, rsqrt, sub_6, var_mean
#   ret_1 => add_15, add_16, clone_2, mul_7, mul_8, rsqrt_1, sub_7, var_mean_1
#   x_12 => cat
#   x_13 => add_12
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_11, %permute_2], 1), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_6), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %primals_7), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %primals_8), kwargs = {})
#   %clone_2 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_2, %getitem_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_9), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_10), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_11 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 50)
    x1 = xindex // 50
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp31_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (r2 + 768*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp18 = tmp16 - tmp17
        tmp20 = 768.0
        tmp21 = tmp19 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp18 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight, roffset == 0
        )
        tmp31_mean = tl.where(rmask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(rmask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(rmask & xmask, tmp31_weight_next, tmp31_weight)
        tl.store(out_ptr0 + (r2 + 768*x3), tmp29, rmask & xmask)
    tmp31_tmp, tmp32_tmp, tmp33_tmp = triton_helpers.welford(
        tmp31_mean, tmp31_m2, tmp31_weight, 1
    )
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    tmp33 = tmp33_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(out_ptr0 + (r2 + 768*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp34 - tmp31
        tmp36 = 768.0
        tmp37 = tmp32 / tmp36
        tmp38 = 1e-05
        tmp39 = tmp37 + tmp38
        tmp40 = libdevice.rsqrt(tmp39)
        tmp41 = tmp35 * tmp40
        tmp43 = tmp41 * tmp42
        tmp45 = tmp43 + tmp44
        tl.store(out_ptr3 + (r2 + 768*x1 + 1536*x0), tmp45, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ja/cja53n2xzqyvz6kd4yx77ke5osbkzasjrdn6glwwwr2ttnbzdscl.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_13, %view_14, %view_15, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_12 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2304*x1 + 4608*x2 + 4608*((x0 + 768*x1) // 1536)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gg/cgguos6exf4mrqr2fkvo5tm2pzojvrkjxghgngrveqhoxmrsuy4z.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_13, %view_14, %view_15, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_13 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 2304*x1 + 4608*x2 + 4608*((x0 + 768*x1) // 1536)), xmask)
    tmp1 = tl.load(in_ptr1 + (768 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fb/cfbouxj5ngrol6ymruhyq6iswyslv5itippfrwsqcj4dx7sfdfcp.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_13, %view_14, %view_15, None, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_14 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 2)
    x2 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 2304*x1 + 4608*x2 + 4608*((x0 + 768*x1) // 1536)), xmask)
    tmp1 = tl.load(in_ptr1 + (1536 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/2e/c2ext6argdmseeje33j4ndbo7uzpdexpmn3qhrjjuuy6ovayhjha.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/fc/cfcqsx4mx2qbska333ejsbq2vmhzkaj5vzu3y4qdkhpgrj5u67c3.py
# Topologically Sorted Source Nodes: [x_16, ret_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_2 => add_18, add_19, clone_5, mul_10, mul_9, rsqrt_2, sub_8, var_mean_2
#   x_16 => add_17
# Graph fragment:
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_3, %view_17), kwargs = {})
#   %clone_5 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_17,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_5, %getitem_9), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_15), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_16), kwargs = {})
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_per_fused_add_native_layer_norm_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r2 + 768*x3), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/yh/cyhb74tnbxkx4er74pumu7b542q3athzigvdc5joi4o4vhpd66rh.py
# Topologically Sorted Source Nodes: [mul_5, sigmoid, input_2], Original ATen: [aten.mul, aten.sigmoid]
# Source node to ATen node mapping:
#   input_2 => mul_12
#   mul_5 => mul_11
#   sigmoid => sigmoid
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 1.702), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_11,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %sigmoid), kwargs = {})
triton_poi_fused_mul_sigmoid_17 = async_compile.triton('triton_poi_fused_mul_sigmoid_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 307200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.702
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp2 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5u/c5u32dxckw6kayoxavsmcc7tydac6m3d2uwv5jm7ibypzsy5sy3t.py
# Topologically Sorted Source Nodes: [x_16, x_17, ret_3], Original ATen: [aten.add, aten.native_layer_norm]
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
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_6, %getitem_11), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_3), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %primals_21), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %primals_22), kwargs = {})
triton_per_fused_add_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r2 + 768*x3), tmp8, rmask)
    tl.store(out_ptr2 + (r2 + 768*x3), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/v5/cv5f2uzeuae67shtrgi6yrvfcdfj2nq2zgoihf6e54acomwapdjs.py
# Topologically Sorted Source Nodes: [x_18, ret_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_4 => add_24, add_25, clone_9, mul_15, mul_16, rsqrt_4, sub_10, var_mean_4
#   x_18 => add_23
# Graph fragment:
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_32), kwargs = {})
#   %clone_9 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_23,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_9, %getitem_17), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_4), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_27), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %primals_28), kwargs = {})
triton_per_fused_add_native_layer_norm_19 = async_compile.triton('triton_per_fused_add_native_layer_norm_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + 768*x0), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sy/csyhs6kozx2iur5ylqbsusf4wnvn7ghhytlqzvyvvor5zjd44i3c.py
# Topologically Sorted Source Nodes: [x_18, x_19, ret_5], Original ATen: [aten.add, aten.native_layer_norm]
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
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_10, %getitem_19), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_5), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %primals_33), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %primals_34), kwargs = {})
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_per_fused_add_native_layer_norm_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fw/cfwvbnok7lkknot7qsbt73rkikrwaoxacw2httcwaseqfhb32ybh.py
# Topologically Sorted Source Nodes: [x_24, x_25, ret_11], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_11 => add_45, add_46, clone_22, mul_37, mul_38, rsqrt_11, sub_17, var_mean_11
#   x_24 => add_41
#   x_25 => add_44
# Graph fragment:
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %view_77), kwargs = {})
#   %add_44 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %view_81), kwargs = {})
#   %clone_22 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_44,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_22, %getitem_43), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %rsqrt_11), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %primals_69), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %primals_70), kwargs = {})
triton_per_fused_add_native_layer_norm_21 = async_compile.triton('triton_per_fused_add_native_layer_norm_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
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
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xt/cxt3badkrils2s6cqay3zdbniygftvos2w6cekpjbq2odi4nwvt2.py
# Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   ret_25 => add_87, add_88, clone_50, mul_79, mul_80, rsqrt_25, sub_31, var_mean_25
# Graph fragment:
#   %clone_50 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%select_36,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_50, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_87,), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_50, %getitem_99), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %rsqrt_25), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %primals_153), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %primals_154), kwargs = {})
triton_per_fused_native_layer_norm_22 = async_compile.triton('triton_per_fused_native_layer_norm_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/xq/cxqklzzqkeocg3iemcjsyefiwfp4yiis2dqezawuwu4jur5uzsku.py
# Topologically Sorted Source Nodes: [reshape_1], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   reshape_1 => view_187
# Graph fragment:
#   %view_187 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_115, [2, 768, 7, 7]), kwargs = {})
triton_poi_fused_view_23 = async_compile.triton('triton_poi_fused_view_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 49)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 768*x2 + 1536*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/re/crez5xndjod5xd4rlhyytpwzyong32gkr6dd65djkpzn2dhdkjgk.py
# Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv => mul_81, sum_1
# Graph fragment:
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_117, %primals_157), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_81, [1]), kwargs = {})
triton_red_fused_mv_24 = async_compile.triton('triton_red_fused_mv_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_24(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/kj/ckjigs2nyloolmgypizoimmimshgbnk3vyibz6a4nrrl346iytib.py
# Topologically Sorted Source Nodes: [v], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   v => div_1, pow_1, sum_2
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [0], True), kwargs = {})
#   %div_1 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %expand_7), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_158, %div_1), kwargs = {})
triton_red_fused_div_linalg_vector_norm_25 = async_compile.triton('triton_red_fused_div_linalg_vector_norm_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_linalg_vector_norm_25', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_linalg_vector_norm_25(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/is/cisnvcgn2zi55d756gvld6pz55qom42ai7456jgaxdv5fd7n5prd.py
# Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_1 => mul_82, sum_3
# Graph fragment:
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_189, %div_1), kwargs = {})
#   %sum_3 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_82, [1]), kwargs = {})
triton_red_fused_mv_26 = async_compile.triton('triton_red_fused_mv_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_26(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/lk/clkvvm6tll5clqrmtotcgmld5t6eni3wzj6vtt5fh4bwowlfwsm7.py
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
triton_per_fused_div_dot_linalg_vector_norm_27 = async_compile.triton('triton_per_fused_div_dot_linalg_vector_norm_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_dot_linalg_vector_norm_27', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_dot_linalg_vector_norm_27(in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_elicer/fg/cfgpvkv6hivjez5xovzm43cgxeej3jink4rwca3s3myuxmzxbi7s.py
# Topologically Sorted Source Nodes: [weight, input_38], Original ATen: [aten.div, aten.convolution]
# Source node to ATen node mapping:
#   input_38 => convolution_1
#   weight => div_3
# Graph fragment:
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_156, %sum_6), kwargs = {})
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_187, %div_3, %primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_div_28 = async_compile.triton('triton_poi_fused_convolution_div_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/6e/c6eqcugoapcny26nzeciowlf2heae6s5xnqsigtmf2rr3p4uptj3.py
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
triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/5j/c5jihfgker4wqpednk4ahieviucfvuvxxntz6yoayb4tmlho7xgf.py
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
triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30 = async_compile.triton('triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30', 'mutated_arg_names': ['in_ptr1', 'out_ptr5', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, out_ptr5, out_ptr6, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_elicer/oz/cozzmj7kdgfx63wijchbwow2urgijlv2mbjv2oldqbcmfhmjen7b.py
# Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_41 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_2, %div_6, %primals_164, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/ve/cvelndtz4ofwprtqm3eo33lkdzdbqjtg46sb5n7lpxs7lffpxfiy.py
# Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_12 => mul_99, sum_25
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_121, %primals_175), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_99, [1]), kwargs = {})
triton_red_fused_mv_32 = async_compile.triton('triton_red_fused_mv_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mv_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mv_32(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/n2/cn2aie3ewsnguyfggtmx3mfl6wnnjln7wgpflwhteapqwwz3ipkh.py
# Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_12 => mul_99, sum_25
# Graph fragment:
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_121, %primals_175), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_99, [1]), kwargs = {})
triton_per_fused_mv_33 = async_compile.triton('triton_per_fused_mv_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_33(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/lb/clbvgdkma6w5xcypx7zscj53cpya7ma4a6bkmjamih6leoyjmmrl.py
# Topologically Sorted Source Nodes: [v_8], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   v_8 => div_13, pow_17, sum_26
# Graph fragment:
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_25, 2.0), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_17, [0], True), kwargs = {})
#   %div_13 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_25, %expand_23), kwargs = {})
#   %copy__9 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_176, %div_13), kwargs = {})
triton_per_fused_div_linalg_vector_norm_34 = async_compile.triton('triton_per_fused_div_linalg_vector_norm_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_linalg_vector_norm_34', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_linalg_vector_norm_34(in_out_ptr0, out_ptr1, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_elicer/lt/cltlw2ifwfxkddo2dgdu42d5736wpf2k6nm6gcudy7h4mv7ugefn.py
# Topologically Sorted Source Nodes: [mv_13], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   mv_13 => mul_100, sum_27
# Graph fragment:
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_193, %div_13), kwargs = {})
#   %sum_27 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_100, [1]), kwargs = {})
triton_per_fused_mv_35 = async_compile.triton('triton_per_fused_mv_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mv_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mv_35(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_elicer/we/cwe2zu3deav7fwhadu575bkqlqxowewf47nbhzykeugrjvtvqgjt.py
# Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   weight_4 => div_15
# Graph fragment:
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_174, %sum_30), kwargs = {})
triton_poi_fused_div_36 = async_compile.triton('triton_poi_fused_div_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/bc/cbcsjvigdvef7xe4nmb3vbmwumpce4w34sc32i2v4bjvg32i677x.py
# Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.addmm, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_47 => add_tensor
#   input_48 => gt_2, mul_103, where_2
# Graph fragment:
#   %add_tensor : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_177), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_tensor, 0), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_tensor, %mul_103), kwargs = {})
triton_poi_fused_addmm_leaky_relu_37 = async_compile.triton('triton_poi_fused_addmm_leaky_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_leaky_relu_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/f2/cf2gv4tqlqwv3oyiwbkphwvrakfxfjqqozqjgjwsukfxynlspc7t.py
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
triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_38 = async_compile.triton('triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_38', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_38(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/zi/cziomunxbrldw4bifjkfnwt4xpytrap5x5k65qbiy5oethzb5u7y.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_43 => convolution_4
#   input_44 => gt_1, mul_94, where_1
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_188, %div_9, %primals_168, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_4, 0), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_4, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_4, %mul_94), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_1, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        buf1 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0.run(buf0, buf1, 0, 2, grid=grid(2), stream=stream0)
        buf2 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_1.run(buf0, buf2, 1, 2, grid=grid(2), stream=stream0)
        buf3 = empty_strided_cuda((2, 3, 256, 256), (196608, 65536, 256, 1), torch.float32)
        buf4 = empty_strided_cuda((2, 1, 1, 1, 24), (24, 48, 48, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x, x_mean, sub_1, mul, mul_1, x_1, x_mean_1], Original ATen: [aten.sub, aten.add, aten.mean, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_mul_sub_2.run(primals_1, buf1, buf2, buf3, buf4, 48, 8192, grid=grid(48), stream=stream0)
        del primals_1
        buf5 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_3.run(buf4, buf5, 2, 24, grid=grid(2), stream=stream0)
        del buf4
        buf6 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [rand_2], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0.run(buf0, buf6, 2, 2, grid=grid(2), stream=stream0)
        buf8 = empty_strided_cuda((2, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_4.run(buf8, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [mask, setitem], Original ATen: [aten.ones, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_ones_5.run(buf0, buf8, 5, 6, 32768, grid=grid(32768), stream=stream0)
        buf7 = empty_strided_cuda((2, 256, 256, 3), (196608, 768, 3, 1), torch.float32)
        buf10 = reinterpret_tensor(buf7, (2, 3, 256, 256), (196608, 1, 768, 3), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [contiguous, getitem_3, x_4, x_5, mul_4, add_10], Original ATen: [aten.clone, aten.index, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_index_mul_6.run(buf10, buf0, buf3, buf5, buf6, buf8, 3, 4, 393216, grid=grid(393216), stream=stream0)
        del buf0
        del buf3
        buf11 = empty_strided_cuda((2, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, mul_4, add_10, x_6], Original ATen: [aten.mul, aten.clone, aten.add, aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_add_clone_mul_7.run(buf10, buf11, 301056, grid=grid(301056), stream=stream0)
        del buf10
        buf12 = empty_strided_cuda((3, 1, 1), (1, 1, 1), torch.float32)
        buf12.copy_(reinterpret_tensor(primals_2, (3, 1, 1), (1, 1, 1), 0), False)
        del primals_2
        buf13 = empty_strided_cuda((3, 1, 1), (1, 1, 1), torch.float32)
        buf13.copy_(reinterpret_tensor(primals_3, (3, 1, 1), (1, 1, 1), 0), False)
        del primals_3
        buf14 = empty_strided_cuda((2, 3, 224, 224), (150528, 50176, 224, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8, x_9], Original ATen: [aten.sub, aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_sub_8.run(buf11, buf12, buf13, buf14, 6, 50176, grid=grid(6, 50176), stream=stream0)
        del buf11
        del buf12
        del buf13
        # Topologically Sorted Source Nodes: [x_7, x_8, x_9], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_4, stride=(32, 32), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (2, 768, 7, 7), (37632, 49, 7, 1))
        del buf14
        del primals_4
        buf16 = empty_strided_cuda((2, 50, 1, 6), (300, 6, 600, 1), torch.float32)
        buf17 = empty_strided_cuda((2, 50, 1, 6), (300, 6, 600, 1), torch.float32)
        buf18 = empty_strided_cuda((2, 50, 1, 6), (300, 6, 600, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, ret], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_9.run(primals_5, buf15, primals_6, buf16, buf17, buf18, 600, 128, grid=grid(600), stream=stream0)
        buf19 = empty_strided_cuda((2, 50, 1), (50, 1, 100), torch.float32)
        buf20 = empty_strided_cuda((2, 50, 1), (50, 1, 100), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, ret], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_native_layer_norm_10.run(buf16, buf17, buf18, buf19, buf20, 100, 6, grid=grid(100), stream=stream0)
        del buf16
        del buf17
        del buf18
        buf22 = empty_strided_cuda((2, 50, 768), (38400, 768, 1), torch.float32)
        buf26 = empty_strided_cuda((50, 2, 768), (1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, x_13, ret, ret_1], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_11.run(primals_5, buf15, primals_6, buf19, buf20, primals_7, primals_8, primals_9, primals_10, buf22, buf26, 100, 768, grid=grid(100), stream=stream0)
        del buf19
        del buf20
        del primals_10
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        buf27 = empty_strided_cuda((100, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf26, (100, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 2304), (1, 768), 0), out=buf27)
        del primals_12
        buf28 = reinterpret_tensor(buf26, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf27, primals_11, buf28, 76800, grid=grid(76800), stream=stream0)
        buf29 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf27, primals_11, buf29, 76800, grid=grid(76800), stream=stream0)
        buf30 = empty_strided_cuda((2, 12, 50, 64), (768, 64, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf27, primals_11, buf30, 76800, grid=grid(76800), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf28, buf29, buf30, None, False)
        del buf28
        buf32 = buf31[0]
        del buf31
        buf36 = reinterpret_tensor(buf30, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf32, buf36, 76800, grid=grid(76800), stream=stream0)
        buf37 = reinterpret_tensor(buf32, (100, 768), (768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf36, (100, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 768), (1, 768), 0), out=buf37)
        del primals_13
        buf41 = reinterpret_tensor(buf36, (50, 2, 768), (1536, 768, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_16, ret_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_16.run(buf22, buf37, primals_14, primals_15, primals_16, buf41, 100, 768, grid=grid(100), stream=stream0)
        del primals_15
        del primals_16
        buf42 = empty_strided_cuda((100, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf41, (100, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 3072), (1, 768), 0), out=buf42)
        del primals_17
        buf43 = reinterpret_tensor(buf42, (50, 2, 3072), (6144, 3072, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [mul_5, sigmoid, input_2], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf43, primals_18, 307200, grid=grid(307200), stream=stream0)
        del primals_18
        buf44 = reinterpret_tensor(buf41, (100, 768), (768, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf43, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 768), (1, 3072), 0), out=buf44)
        del primals_19
        buf45 = reinterpret_tensor(buf37, (50, 2, 768), (1536, 768, 1), 0); del buf37  # reuse
        buf49 = reinterpret_tensor(buf29, (50, 2, 768), (1536, 768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_16, x_17, ret_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_18.run(buf45, buf22, primals_14, buf44, primals_20, primals_21, primals_22, buf49, 100, 768, grid=grid(100), stream=stream0)
        del primals_14
        del primals_20
        del primals_21
        del primals_22
        buf50 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (100, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 2304), (1, 768), 0), out=buf50)
        del primals_24
        buf51 = reinterpret_tensor(buf49, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf50, primals_23, buf51, 76800, grid=grid(76800), stream=stream0)
        buf52 = reinterpret_tensor(buf44, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf50, primals_23, buf52, 76800, grid=grid(76800), stream=stream0)
        buf53 = reinterpret_tensor(buf22, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf50, primals_23, buf53, 76800, grid=grid(76800), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf54 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf51, buf52, buf53, None, False)
        del buf51
        buf55 = buf54[0]
        del buf54
        buf59 = reinterpret_tensor(buf53, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf55, buf59, 76800, grid=grid(76800), stream=stream0)
        buf60 = reinterpret_tensor(buf55, (100, 768), (768, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf59, (100, 768), (768, 1), 0), reinterpret_tensor(primals_25, (768, 768), (1, 768), 0), out=buf60)
        del primals_25
        buf64 = reinterpret_tensor(buf59, (50, 2, 768), (1536, 768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_18, ret_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf45, buf60, primals_26, primals_27, primals_28, buf64, 100, 768, grid=grid(100), stream=stream0)
        del primals_27
        del primals_28
        buf65 = reinterpret_tensor(buf43, (100, 3072), (3072, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf64, (100, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 3072), (1, 768), 0), out=buf65)
        del primals_29
        buf66 = reinterpret_tensor(buf65, (50, 2, 3072), (6144, 3072, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [mul_7, sigmoid_1, input_5], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf66, primals_30, 307200, grid=grid(307200), stream=stream0)
        del primals_30
        buf67 = reinterpret_tensor(buf64, (100, 768), (768, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf66, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_31, (3072, 768), (1, 3072), 0), out=buf67)
        del primals_31
        buf68 = buf45; del buf45  # reuse
        buf72 = reinterpret_tensor(buf52, (50, 2, 768), (1536, 768, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_18, x_19, ret_5], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf68, buf60, primals_26, buf67, primals_32, primals_33, primals_34, buf72, 100, 768, grid=grid(100), stream=stream0)
        del primals_26
        del primals_32
        del primals_33
        del primals_34
        buf73 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf72, (100, 768), (768, 1), 0), reinterpret_tensor(primals_36, (768, 2304), (1, 768), 0), out=buf73)
        del primals_36
        buf74 = reinterpret_tensor(buf72, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf73, primals_35, buf74, 76800, grid=grid(76800), stream=stream0)
        buf75 = reinterpret_tensor(buf67, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf73, primals_35, buf75, 76800, grid=grid(76800), stream=stream0)
        buf76 = reinterpret_tensor(buf60, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf73, primals_35, buf76, 76800, grid=grid(76800), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf77 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf74, buf75, buf76, None, False)
        del buf74
        buf78 = buf77[0]
        del buf77
        buf82 = reinterpret_tensor(buf76, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf78, buf82, 76800, grid=grid(76800), stream=stream0)
        buf83 = reinterpret_tensor(buf78, (100, 768), (768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf82, (100, 768), (768, 1), 0), reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), out=buf83)
        del primals_37
        buf87 = reinterpret_tensor(buf82, (50, 2, 768), (1536, 768, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_20, ret_6], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf68, buf83, primals_38, primals_39, primals_40, buf87, 100, 768, grid=grid(100), stream=stream0)
        del primals_39
        del primals_40
        buf88 = reinterpret_tensor(buf66, (100, 3072), (3072, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf87, (100, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 3072), (1, 768), 0), out=buf88)
        del primals_41
        buf89 = reinterpret_tensor(buf88, (50, 2, 3072), (6144, 3072, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [mul_9, sigmoid_2, input_8], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf89, primals_42, 307200, grid=grid(307200), stream=stream0)
        del primals_42
        buf90 = reinterpret_tensor(buf87, (100, 768), (768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf89, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 768), (1, 3072), 0), out=buf90)
        del primals_43
        buf91 = buf68; del buf68  # reuse
        buf95 = reinterpret_tensor(buf75, (50, 2, 768), (1536, 768, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_20, x_21, ret_7], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf91, buf83, primals_38, buf90, primals_44, primals_45, primals_46, buf95, 100, 768, grid=grid(100), stream=stream0)
        del primals_38
        del primals_44
        del primals_45
        del primals_46
        buf96 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf95, (100, 768), (768, 1), 0), reinterpret_tensor(primals_48, (768, 2304), (1, 768), 0), out=buf96)
        del primals_48
        buf97 = reinterpret_tensor(buf95, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf96, primals_47, buf97, 76800, grid=grid(76800), stream=stream0)
        buf98 = reinterpret_tensor(buf90, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf96, primals_47, buf98, 76800, grid=grid(76800), stream=stream0)
        buf99 = reinterpret_tensor(buf83, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf96, primals_47, buf99, 76800, grid=grid(76800), stream=stream0)
        del primals_47
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf100 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf97, buf98, buf99, None, False)
        del buf97
        buf101 = buf100[0]
        del buf100
        buf105 = reinterpret_tensor(buf99, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf101, buf105, 76800, grid=grid(76800), stream=stream0)
        buf106 = reinterpret_tensor(buf101, (100, 768), (768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf105, (100, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 768), (1, 768), 0), out=buf106)
        del primals_49
        buf110 = reinterpret_tensor(buf105, (50, 2, 768), (1536, 768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_22, ret_8], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf91, buf106, primals_50, primals_51, primals_52, buf110, 100, 768, grid=grid(100), stream=stream0)
        del primals_51
        del primals_52
        buf111 = reinterpret_tensor(buf89, (100, 3072), (3072, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf110, (100, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 3072), (1, 768), 0), out=buf111)
        del primals_53
        buf112 = reinterpret_tensor(buf111, (50, 2, 3072), (6144, 3072, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [mul_11, sigmoid_3, input_11], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf112, primals_54, 307200, grid=grid(307200), stream=stream0)
        del primals_54
        buf113 = reinterpret_tensor(buf110, (100, 768), (768, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf112, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_55, (3072, 768), (1, 3072), 0), out=buf113)
        del primals_55
        buf114 = buf91; del buf91  # reuse
        buf118 = reinterpret_tensor(buf98, (50, 2, 768), (1536, 768, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_22, x_23, ret_9], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf114, buf106, primals_50, buf113, primals_56, primals_57, primals_58, buf118, 100, 768, grid=grid(100), stream=stream0)
        del primals_50
        del primals_56
        del primals_57
        del primals_58
        buf119 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf118, (100, 768), (768, 1), 0), reinterpret_tensor(primals_60, (768, 2304), (1, 768), 0), out=buf119)
        del primals_60
        buf120 = reinterpret_tensor(buf118, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf119, primals_59, buf120, 76800, grid=grid(76800), stream=stream0)
        buf121 = reinterpret_tensor(buf113, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf119, primals_59, buf121, 76800, grid=grid(76800), stream=stream0)
        buf122 = reinterpret_tensor(buf106, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf119, primals_59, buf122, 76800, grid=grid(76800), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf123 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf120, buf121, buf122, None, False)
        buf124 = buf123[0]
        del buf123
        buf128 = reinterpret_tensor(buf122, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf124, buf128, 76800, grid=grid(76800), stream=stream0)
        buf129 = reinterpret_tensor(buf124, (100, 768), (768, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf128, (100, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 768), (1, 768), 0), out=buf129)
        del primals_61
        buf133 = reinterpret_tensor(buf128, (50, 2, 768), (1536, 768, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_24, ret_10], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf114, buf129, primals_62, primals_63, primals_64, buf133, 100, 768, grid=grid(100), stream=stream0)
        del primals_63
        del primals_64
        buf134 = reinterpret_tensor(buf112, (100, 3072), (3072, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf133, (100, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 3072), (1, 768), 0), out=buf134)
        del primals_65
        buf135 = reinterpret_tensor(buf134, (50, 2, 3072), (6144, 3072, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [mul_13, sigmoid_4, input_14], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf135, primals_66, 307200, grid=grid(307200), stream=stream0)
        del primals_66
        buf136 = reinterpret_tensor(buf133, (100, 768), (768, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf135, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_67, (3072, 768), (1, 3072), 0), out=buf136)
        del primals_67
        buf137 = reinterpret_tensor(buf129, (50, 2, 768), (1536, 768, 1), 0); del buf129  # reuse
        buf141 = reinterpret_tensor(buf121, (50, 2, 768), (1536, 768, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_24, x_25, ret_11], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_21.run(buf137, buf114, primals_62, buf136, primals_68, primals_69, primals_70, buf141, 100, 768, grid=grid(100), stream=stream0)
        del primals_62
        del primals_68
        del primals_69
        del primals_70
        buf142 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf141, (100, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 2304), (1, 768), 0), out=buf142)
        del primals_72
        buf143 = reinterpret_tensor(buf141, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf142, primals_71, buf143, 76800, grid=grid(76800), stream=stream0)
        buf144 = reinterpret_tensor(buf136, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf142, primals_71, buf144, 76800, grid=grid(76800), stream=stream0)
        buf145 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf142, primals_71, buf145, 76800, grid=grid(76800), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf146 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf143, buf144, buf145, None, False)
        del buf143
        buf147 = buf146[0]
        del buf146
        buf151 = reinterpret_tensor(buf145, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf147, buf151, 76800, grid=grid(76800), stream=stream0)
        buf152 = reinterpret_tensor(buf147, (100, 768), (768, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf151, (100, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 768), (1, 768), 0), out=buf152)
        del primals_73
        buf156 = reinterpret_tensor(buf151, (50, 2, 768), (1536, 768, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_26, ret_12], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf137, buf152, primals_74, primals_75, primals_76, buf156, 100, 768, grid=grid(100), stream=stream0)
        del primals_75
        del primals_76
        buf157 = reinterpret_tensor(buf135, (100, 3072), (3072, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf156, (100, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 3072), (1, 768), 0), out=buf157)
        del primals_77
        buf158 = reinterpret_tensor(buf157, (50, 2, 3072), (6144, 3072, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [mul_15, sigmoid_5, input_17], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf158, primals_78, 307200, grid=grid(307200), stream=stream0)
        del primals_78
        buf159 = reinterpret_tensor(buf156, (100, 768), (768, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf158, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 768), (1, 3072), 0), out=buf159)
        del primals_79
        buf160 = buf137; del buf137  # reuse
        buf164 = reinterpret_tensor(buf144, (50, 2, 768), (1536, 768, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_26, x_27, ret_13], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf160, buf152, primals_74, buf159, primals_80, primals_81, primals_82, buf164, 100, 768, grid=grid(100), stream=stream0)
        del primals_74
        del primals_80
        del primals_81
        del primals_82
        buf165 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf164, (100, 768), (768, 1), 0), reinterpret_tensor(primals_84, (768, 2304), (1, 768), 0), out=buf165)
        del primals_84
        buf166 = reinterpret_tensor(buf164, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf165, primals_83, buf166, 76800, grid=grid(76800), stream=stream0)
        buf167 = reinterpret_tensor(buf159, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf165, primals_83, buf167, 76800, grid=grid(76800), stream=stream0)
        buf168 = reinterpret_tensor(buf152, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf165, primals_83, buf168, 76800, grid=grid(76800), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf169 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf166, buf167, buf168, None, False)
        del buf166
        buf170 = buf169[0]
        del buf169
        buf174 = reinterpret_tensor(buf168, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf170, buf174, 76800, grid=grid(76800), stream=stream0)
        buf175 = reinterpret_tensor(buf170, (100, 768), (768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf174, (100, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), out=buf175)
        del primals_85
        buf179 = reinterpret_tensor(buf174, (50, 2, 768), (1536, 768, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_28, ret_14], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf160, buf175, primals_86, primals_87, primals_88, buf179, 100, 768, grid=grid(100), stream=stream0)
        del primals_87
        del primals_88
        buf180 = reinterpret_tensor(buf158, (100, 3072), (3072, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf179, (100, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 3072), (1, 768), 0), out=buf180)
        del primals_89
        buf181 = reinterpret_tensor(buf180, (50, 2, 3072), (6144, 3072, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [mul_17, sigmoid_6, input_20], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf181, primals_90, 307200, grid=grid(307200), stream=stream0)
        del primals_90
        buf182 = reinterpret_tensor(buf179, (100, 768), (768, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf181, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (1, 3072), 0), out=buf182)
        del primals_91
        buf183 = buf160; del buf160  # reuse
        buf187 = reinterpret_tensor(buf167, (50, 2, 768), (1536, 768, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_28, x_29, ret_15], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf183, buf175, primals_86, buf182, primals_92, primals_93, primals_94, buf187, 100, 768, grid=grid(100), stream=stream0)
        del primals_86
        del primals_92
        del primals_93
        del primals_94
        buf188 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf187, (100, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 2304), (1, 768), 0), out=buf188)
        del primals_96
        buf189 = reinterpret_tensor(buf187, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf188, primals_95, buf189, 76800, grid=grid(76800), stream=stream0)
        buf190 = reinterpret_tensor(buf182, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf188, primals_95, buf190, 76800, grid=grid(76800), stream=stream0)
        buf191 = reinterpret_tensor(buf175, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf188, primals_95, buf191, 76800, grid=grid(76800), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf192 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf189, buf190, buf191, None, False)
        del buf189
        buf193 = buf192[0]
        del buf192
        buf197 = reinterpret_tensor(buf191, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf193, buf197, 76800, grid=grid(76800), stream=stream0)
        buf198 = reinterpret_tensor(buf193, (100, 768), (768, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf197, (100, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 768), (1, 768), 0), out=buf198)
        del primals_97
        buf202 = reinterpret_tensor(buf197, (50, 2, 768), (1536, 768, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_30, ret_16], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf183, buf198, primals_98, primals_99, primals_100, buf202, 100, 768, grid=grid(100), stream=stream0)
        del primals_100
        del primals_99
        buf203 = reinterpret_tensor(buf181, (100, 3072), (3072, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf202, (100, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), out=buf203)
        del primals_101
        buf204 = reinterpret_tensor(buf203, (50, 2, 3072), (6144, 3072, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [mul_19, sigmoid_7, input_23], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf204, primals_102, 307200, grid=grid(307200), stream=stream0)
        del primals_102
        buf205 = reinterpret_tensor(buf202, (100, 768), (768, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf204, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), out=buf205)
        del primals_103
        buf206 = buf183; del buf183  # reuse
        buf210 = reinterpret_tensor(buf190, (50, 2, 768), (1536, 768, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_30, x_31, ret_17], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf206, buf198, primals_98, buf205, primals_104, primals_105, primals_106, buf210, 100, 768, grid=grid(100), stream=stream0)
        del primals_104
        del primals_105
        del primals_106
        del primals_98
        buf211 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf210, (100, 768), (768, 1), 0), reinterpret_tensor(primals_108, (768, 2304), (1, 768), 0), out=buf211)
        del primals_108
        buf212 = reinterpret_tensor(buf210, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf211, primals_107, buf212, 76800, grid=grid(76800), stream=stream0)
        buf213 = reinterpret_tensor(buf205, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf211, primals_107, buf213, 76800, grid=grid(76800), stream=stream0)
        buf214 = reinterpret_tensor(buf198, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf211, primals_107, buf214, 76800, grid=grid(76800), stream=stream0)
        del primals_107
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf215 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf212, buf213, buf214, None, False)
        buf216 = buf215[0]
        del buf215
        buf220 = reinterpret_tensor(buf214, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf216, buf220, 76800, grid=grid(76800), stream=stream0)
        buf221 = reinterpret_tensor(buf216, (100, 768), (768, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf220, (100, 768), (768, 1), 0), reinterpret_tensor(primals_109, (768, 768), (1, 768), 0), out=buf221)
        del primals_109
        buf225 = reinterpret_tensor(buf220, (50, 2, 768), (1536, 768, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_32, ret_18], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf206, buf221, primals_110, primals_111, primals_112, buf225, 100, 768, grid=grid(100), stream=stream0)
        del primals_111
        del primals_112
        buf226 = reinterpret_tensor(buf204, (100, 3072), (3072, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf225, (100, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 3072), (1, 768), 0), out=buf226)
        del primals_113
        buf227 = reinterpret_tensor(buf226, (50, 2, 3072), (6144, 3072, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [mul_21, sigmoid_8, input_26], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf227, primals_114, 307200, grid=grid(307200), stream=stream0)
        del primals_114
        buf228 = reinterpret_tensor(buf225, (100, 768), (768, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf227, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 768), (1, 3072), 0), out=buf228)
        del primals_115
        buf229 = reinterpret_tensor(buf221, (50, 2, 768), (1536, 768, 1), 0); del buf221  # reuse
        buf233 = reinterpret_tensor(buf213, (50, 2, 768), (1536, 768, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_32, x_33, ret_19], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_21.run(buf229, buf206, primals_110, buf228, primals_116, primals_117, primals_118, buf233, 100, 768, grid=grid(100), stream=stream0)
        del primals_110
        del primals_116
        del primals_117
        del primals_118
        buf234 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf233, (100, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 2304), (1, 768), 0), out=buf234)
        del primals_120
        buf235 = reinterpret_tensor(buf233, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf234, primals_119, buf235, 76800, grid=grid(76800), stream=stream0)
        buf236 = reinterpret_tensor(buf228, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf234, primals_119, buf236, 76800, grid=grid(76800), stream=stream0)
        buf237 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf234, primals_119, buf237, 76800, grid=grid(76800), stream=stream0)
        del primals_119
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf238 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf235, buf236, buf237, None, False)
        del buf235
        buf239 = buf238[0]
        del buf238
        buf243 = reinterpret_tensor(buf237, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf239, buf243, 76800, grid=grid(76800), stream=stream0)
        buf244 = reinterpret_tensor(buf239, (100, 768), (768, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf243, (100, 768), (768, 1), 0), reinterpret_tensor(primals_121, (768, 768), (1, 768), 0), out=buf244)
        del primals_121
        buf248 = reinterpret_tensor(buf243, (50, 2, 768), (1536, 768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_34, ret_20], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf229, buf244, primals_122, primals_123, primals_124, buf248, 100, 768, grid=grid(100), stream=stream0)
        del primals_123
        del primals_124
        buf249 = reinterpret_tensor(buf227, (100, 3072), (3072, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf248, (100, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 3072), (1, 768), 0), out=buf249)
        del primals_125
        buf250 = reinterpret_tensor(buf249, (50, 2, 3072), (6144, 3072, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [mul_23, sigmoid_9, input_29], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf250, primals_126, 307200, grid=grid(307200), stream=stream0)
        del primals_126
        buf251 = reinterpret_tensor(buf248, (100, 768), (768, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf250, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_127, (3072, 768), (1, 3072), 0), out=buf251)
        del primals_127
        buf252 = buf229; del buf229  # reuse
        buf256 = reinterpret_tensor(buf236, (50, 2, 768), (1536, 768, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_34, x_35, ret_21], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf252, buf244, primals_122, buf251, primals_128, primals_129, primals_130, buf256, 100, 768, grid=grid(100), stream=stream0)
        del primals_122
        del primals_128
        del primals_129
        del primals_130
        buf257 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf256, (100, 768), (768, 1), 0), reinterpret_tensor(primals_132, (768, 2304), (1, 768), 0), out=buf257)
        del primals_132
        buf258 = reinterpret_tensor(buf256, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf257, primals_131, buf258, 76800, grid=grid(76800), stream=stream0)
        buf259 = reinterpret_tensor(buf251, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf257, primals_131, buf259, 76800, grid=grid(76800), stream=stream0)
        buf260 = reinterpret_tensor(buf244, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf257, primals_131, buf260, 76800, grid=grid(76800), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf261 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf258, buf259, buf260, None, False)
        del buf258
        buf262 = buf261[0]
        del buf261
        buf266 = reinterpret_tensor(buf260, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf262, buf266, 76800, grid=grid(76800), stream=stream0)
        buf267 = reinterpret_tensor(buf262, (100, 768), (768, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf266, (100, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 768), (1, 768), 0), out=buf267)
        del primals_133
        buf271 = reinterpret_tensor(buf266, (50, 2, 768), (1536, 768, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_36, ret_22], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf252, buf267, primals_134, primals_135, primals_136, buf271, 100, 768, grid=grid(100), stream=stream0)
        del primals_135
        del primals_136
        buf272 = reinterpret_tensor(buf250, (100, 3072), (3072, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf271, (100, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 3072), (1, 768), 0), out=buf272)
        del primals_137
        buf273 = reinterpret_tensor(buf272, (50, 2, 3072), (6144, 3072, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [mul_25, sigmoid_10, input_32], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf273, primals_138, 307200, grid=grid(307200), stream=stream0)
        del primals_138
        buf274 = reinterpret_tensor(buf271, (100, 768), (768, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf273, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_139, (3072, 768), (1, 3072), 0), out=buf274)
        del primals_139
        buf275 = buf252; del buf252  # reuse
        buf279 = reinterpret_tensor(buf259, (50, 2, 768), (1536, 768, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_36, x_37, ret_23], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_20.run(buf275, buf267, primals_134, buf274, primals_140, primals_141, primals_142, buf279, 100, 768, grid=grid(100), stream=stream0)
        del primals_134
        del primals_140
        del primals_141
        del primals_142
        buf280 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf279, (100, 768), (768, 1), 0), reinterpret_tensor(primals_144, (768, 2304), (1, 768), 0), out=buf280)
        del primals_144
        buf281 = reinterpret_tensor(buf279, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_12.run(buf280, primals_143, buf281, 76800, grid=grid(76800), stream=stream0)
        buf282 = reinterpret_tensor(buf274, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_13.run(buf280, primals_143, buf282, 76800, grid=grid(76800), stream=stream0)
        buf283 = reinterpret_tensor(buf267, (2, 12, 50, 64), (768, 64, 1536, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_14.run(buf280, primals_143, buf283, 76800, grid=grid(76800), stream=stream0)
        del buf280
        del primals_143
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf284 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf281, buf282, buf283, None, False)
        del buf281
        del buf282
        buf285 = buf284[0]
        del buf284
        buf289 = reinterpret_tensor(buf283, (50, 2, 12, 64), (1536, 768, 64, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf285, buf289, 76800, grid=grid(76800), stream=stream0)
        buf290 = reinterpret_tensor(buf285, (100, 768), (768, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf289, (100, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 768), (1, 768), 0), out=buf290)
        del primals_145
        buf294 = reinterpret_tensor(buf289, (50, 2, 768), (1536, 768, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_38, ret_24], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_19.run(buf275, buf290, primals_146, primals_147, primals_148, buf294, 100, 768, grid=grid(100), stream=stream0)
        del primals_147
        del primals_148
        buf295 = reinterpret_tensor(buf273, (100, 3072), (3072, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf294, (100, 768), (768, 1), 0), reinterpret_tensor(primals_149, (768, 3072), (1, 768), 0), out=buf295)
        del primals_149
        buf296 = reinterpret_tensor(buf295, (50, 2, 3072), (6144, 3072, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [mul_27, sigmoid_11, input_35], Original ATen: [aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_17.run(buf296, primals_150, 307200, grid=grid(307200), stream=stream0)
        del primals_150
        buf297 = reinterpret_tensor(buf294, (100, 768), (768, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf296, (100, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 768), (1, 3072), 0), out=buf297)
        del buf296
        del primals_151
        buf298 = empty_strided_cuda((2, 768), (768, 1), torch.float32)
        buf302 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_22.run(buf302, buf275, buf290, primals_146, buf297, primals_152, primals_153, primals_154, 2, 768, grid=grid(2), stream=stream0)
        del buf275
        del buf290
        del buf297
        del primals_146
        del primals_152
        del primals_153
        del primals_154
        buf303 = empty_strided_cuda((2, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_25, x_41], Original ATen: [aten.native_layer_norm, aten.mm]
        extern_kernels.mm(buf302, primals_155, out=buf303)
        del buf302
        del primals_155
        buf304 = reinterpret_tensor(buf15, (2, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [reshape_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_23.run(buf114, buf304, 75264, grid=grid(75264), stream=stream0)
        del buf114
        buf305 = empty_strided_cuda((2, 768, 7, 7), (37632, 1, 5376, 768), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_23.run(buf206, buf305, 75264, grid=grid(75264), stream=stream0)
        del buf206
        buf306 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_24.run(primals_156, primals_157, buf306, 6912, 256, grid=grid(6912), stream=stream0)
        buf308 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_25.run(buf306, buf308, primals_158, 1, 6912, grid=grid(1), stream=stream0)
        del primals_158
        buf309 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_1], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_26.run(primals_156, buf308, buf309, 256, 6912, grid=grid(256), stream=stream0)
        buf311 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf312 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u, sigma], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_27.run(buf309, buf311, primals_157, buf312, 1, 256, grid=grid(1), stream=stream0)
        del primals_157
        buf313 = empty_strided_cuda((256, 768, 3, 3), (6912, 9, 3, 1), torch.float32)
        buf314 = empty_strided_cuda((256, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Topologically Sorted Source Nodes: [weight, input_38], Original ATen: [aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_28.run(primals_156, buf312, buf313, buf314, 196608, 9, grid=grid(196608, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf304, buf314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (2, 256, 7, 7), (12544, 1, 1792, 256))
        buf316 = empty_strided_cuda((2, 256, 9, 9), (20736, 1, 2304, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, pad_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29.run(buf315, primals_159, buf316, 41472, grid=grid(41472), stream=stream0)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf317, (2, 256, 6, 6), (9216, 1, 1536, 256))
        buf319 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf320 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf321 = reinterpret_tensor(buf309, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [mv_3, v_2, mv_4, u_2, sigma_1, weight_1], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30.run(primals_161, primals_162, buf319, buf320, buf321, primals_163, primals_162, 1, 256, grid=grid(1), stream=stream0)
        del primals_163
        buf322 = empty_strided_cuda((2, 256, 6, 6), (9216, 6, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf317, buf322, 3072, 6, grid=grid(3072, 6), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, buf321, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (2, 1, 3, 3), (9, 1, 3, 1))
        buf338 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf339 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf340 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mv_9, v_6, mv_10, u_6, sigma_3, weight_3], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30.run(primals_170, primals_171, buf338, buf339, buf340, primals_172, primals_171, 1, 256, grid=grid(1), stream=stream0)
        del primals_172
        buf325 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [mv_6], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_24.run(primals_165, primals_166, buf325, 6912, 256, grid=grid(6912), stream=stream0)
        buf327 = empty_strided_cuda((6912, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_25.run(buf325, buf327, primals_167, 1, 6912, grid=grid(1), stream=stream0)
        del primals_167
        buf328 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_7], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_26.run(primals_165, buf327, buf328, 256, 6912, grid=grid(256), stream=stream0)
        buf330 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf331 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u_4, sigma_2], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_27.run(buf328, buf330, primals_166, buf331, 1, 256, grid=grid(1), stream=stream0)
        del buf325
        del primals_166
        buf332 = reinterpret_tensor(buf314, (256, 768, 3, 3), (6912, 9, 3, 1), 0); del buf314  # reuse
        buf333 = empty_strided_cuda((256, 768, 3, 3), (6912, 1, 2304, 768), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2, input_43], Original ATen: [aten.div, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_28.run(primals_165, buf331, buf332, buf333, 196608, 9, grid=grid(196608, 9), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf305, buf333, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (2, 256, 7, 7), (12544, 1, 1792, 256))
        del buf333
        buf335 = empty_strided_cuda((2, 256, 9, 9), (20736, 1, 2304, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, pad_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convolution_leaky_relu_29.run(buf334, primals_168, buf335, 41472, grid=grid(41472), stream=stream0)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf336, (2, 256, 6, 6), (9216, 1, 1536, 256))
        buf341 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf336, buf341, 3072, 6, grid=grid(3072, 6), stream=stream0)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf340, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (2, 1, 3, 3), (9, 1, 3, 1))
        del buf341
        buf344 = empty_strided_cuda((512, 2), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_red_fused_mv_32.run(primals_174, primals_175, buf344, 1024, 128, grid=grid(1024), stream=stream0)
        buf345 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mv_12], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_33.run(buf344, buf345, 512, 2, grid=grid(512), stream=stream0)
        buf347 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_34.run(buf347, primals_176, 1, 512, grid=grid(1), stream=stream0)
        del primals_176
        buf348 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [mv_13], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_per_fused_mv_35.run(primals_174, buf347, buf348, 256, 512, grid=grid(256), stream=stream0)
        buf350 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf351 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [u_8, sigma_4], Original ATen: [aten.linalg_vector_norm, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_dot_linalg_vector_norm_27.run(buf348, buf350, primals_175, buf351, 1, 256, grid=grid(1), stream=stream0)
        del buf344
        del primals_175
        buf352 = reinterpret_tensor(buf8, (256, 512), (512, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [weight_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_36.run(primals_174, buf351, buf352, 131072, grid=grid(131072), stream=stream0)
        buf353 = empty_strided_cuda((2, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.addmm]
        extern_kernels.mm(buf303, reinterpret_tensor(buf352, (512, 256), (1, 512), 0), out=buf353)
        buf354 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [input_47, input_48], Original ATen: [aten.addmm, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_leaky_relu_37.run(buf354, primals_177, 512, grid=grid(512), stream=stream0)
        del primals_177
        buf356 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf357 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf358 = reinterpret_tensor(buf348, (1, 256), (256, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [mv_15, v_10, mv_16, u_10, sigma_5, weight_5], Original ATen: [aten.mv, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.dot]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_dot_linalg_vector_norm_mv_30.run(primals_178, primals_179, buf356, buf357, buf358, primals_180, primals_179, 1, 256, grid=grid(1), stream=stream0)
        del primals_180
        buf360 = reinterpret_tensor(buf6, (2, 1), (1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, buf354, reinterpret_tensor(buf358, (256, 1), (1, 256), 0), alpha=1, beta=1, out=buf360)
        del primals_181
        buf324 = buf323; del buf323  # reuse
        buf361 = reinterpret_tensor(buf5, (2, ), (1, ), 0); del buf5  # reuse
        buf343 = buf342; del buf342  # reuse
        buf363 = reinterpret_tensor(buf361, (2, 1), (1, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_46, loss_, mean_2, loss, loss__2, mean_3, loss_1, loss__4, loss_2, loss_3], Original ATen: [aten.convolution, aten.binary_cross_entropy_with_logits, aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_binary_cross_entropy_with_logits_convolution_mean_38.run(buf324, buf343, buf363, primals_164, primals_173, buf360, 2, 9, grid=grid(2), stream=stream0)
        del primals_164
        del primals_173
        buf364 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.bool)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39.run(buf334, primals_168, buf364, 25088, grid=grid(25088), stream=stream0)
        del buf334
        del primals_168
        buf365 = empty_strided_cuda((2, 256, 7, 7), (12544, 1, 1792, 256), torch.bool)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_39.run(buf315, primals_159, buf365, 25088, grid=grid(25088), stream=stream0)
        del buf315
        del primals_159
    return (buf363, buf313, buf321, buf332, buf340, buf352, buf358, primals_156, primals_160, primals_161, primals_162, primals_165, primals_169, primals_170, primals_171, primals_174, primals_178, primals_179, buf303, buf304, buf305, buf308, buf311, buf312, buf313, buf316, buf317, buf319, buf320, buf321, buf324, buf327, buf330, buf331, buf332, buf335, buf336, buf338, buf339, buf340, buf343, buf347, buf350, buf351, buf354, buf356, buf357, buf360, buf358, buf364, buf365, )


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
