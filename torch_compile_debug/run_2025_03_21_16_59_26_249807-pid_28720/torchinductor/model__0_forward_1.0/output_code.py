# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_elicer/oj/coj4yox6sdp64pzskzypk3bhibklrnu4rn2ge3nyff45cxj4coij.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/5h/c5hfibvrusqr2bzxoaf2nbgynzg5qvpnp3lxncn3puhycshrh3j3.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ql/cqlpyfijkmtdvuruewsozgtm25ojjgkln2rkgz7ppn5v5ih7r5h5.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/bd/cbdgfe26dgjf6wn55etaemmrptsmkc4kkfyrobuazklfrynopubw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/qt/cqtdl33kwhaithkjrkosarqfws5ai6t6vd7z546tsgamdynapnp7.py
# Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   in0_input => div
#   in1_input => div_1
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
triton_poi_fused_div_sub_4 = async_compile.triton('triton_poi_fused_div_sub_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 65536*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + 65536*y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp6 = tmp5 - tmp1
    tmp7 = tmp6 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 196608*y1), tmp4, ymask)
    tl.store(out_ptr1 + (y0 + 3*x2 + 196608*y1), tmp7, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/6q/c6qcujaxx77twgd6bg34y3qz4ldpbj7zjbzq5umyenmrkdxwadv4.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_5 = async_compile.triton('triton_poi_fused_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cq/ccq6tv7c4elemgdmmgw6txtsxvc7iozr46xwc34qi2pgwugtjoew.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_3, input_4, input_31, input_32, input_33, input_34, pow_1, sum_1, norm_factor, add, truediv_2, pow_2, sum_2, norm_factor_1, add_1, truediv_3, sub_2, pow_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   in1_input => div_1
#   input_3 => convolution_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_4 => relu_1
#   norm_factor => sqrt
#   norm_factor_1 => sqrt_1
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   truediv_2 => div_2
#   truediv_3 => div_3
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, 1e-10), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_1, %add), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_14, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1], True), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_14, %add_1), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_2, %div_3), kwargs = {})
#   %pow_3 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %pow_24 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 1.0), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_24, 2.0), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_2, %add), kwargs = {})
triton_per_fused_add_convolution_div_mul_pow_relu_sqrt_sub_sum_6 = async_compile.triton('triton_per_fused_add_convolution_div_mul_pow_relu_sqrt_sub_sum_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_mul_pow_relu_sqrt_sub_sum_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_mul_pow_relu_sqrt_sub_sum_6(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 64*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (r1 + 64*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = libdevice.sqrt(tmp8)
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp3, tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 1e-10
    tmp18 = tmp9 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp16)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp12 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = 2.0
    tmp26 = tmp23 * tmp25
    tmp27 = tmp19 / tmp18
    tl.store(in_out_ptr0 + (r1 + 64*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp9, None)
    tl.store(in_out_ptr2 + (r1 + 64*x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + 64*x0), tmp24, None)
    tl.store(out_ptr2 + (r1 + 64*x0), tmp26, None)
    tl.store(out_ptr3 + (r1 + 64*x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3u/c3u6x7pdoofrwxb4hsrnsfuxvmmgtfyjn6mu7quyxxvegnub343f.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_5 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 128*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16448 + x0 + 128*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/6c/c6cu63fhqibciym5r44ljrqzeox3i6mwwjncmmogv2z2o75sbp6m.py
# Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_6 => convolution_2
#   input_7 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_poi_fused_convolution_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/tj/ctj7d54l2q6owsvfssxs4bcwfdueafwqilct5i2yttxxsf65oxtj.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in1_input => div_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   sub_1 => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 128)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 128*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16448 + x0 + 128*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/an/canx5w4s2huncndandwe7orvjwnw472wz3a7r3t4nz6j4imn7gtg.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_8, input_9, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, pow_4, sum_3, norm_factor_2, add_2, truediv_4, pow_5, sum_4, norm_factor_3, add_3, truediv_5, sub_3, pow_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   in1_input => div_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_8 => convolution_3
#   input_9 => relu_3
#   norm_factor_2 => sqrt_2
#   norm_factor_3 => sqrt_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   sub_1 => sub_1
#   sub_3 => sub_3
#   sum_3 => sum_3
#   sum_4 => sum_4
#   truediv_4 => div_4
#   truediv_5 => div_5
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_3,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_3, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [1], True), kwargs = {})
#   %sqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_3,), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_2, 1e-10), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_3, %add_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_16, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   %sqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_4,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_3, 1e-10), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_16, %add_3), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %div_5), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %pow_22 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 1.0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_22, 2.0), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_4, %add_2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_10 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_10(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 128*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (r1 + 128*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None]
    tmp9 = libdevice.sqrt(tmp8)
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp3, tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 1e-10
    tmp18 = tmp9 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp16)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp12 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = 2.0
    tmp26 = tmp23 * tmp25
    tmp27 = tmp19 / tmp18
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp9, None)
    tl.store(in_out_ptr2 + (r1 + 128*x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + 128*x0), tmp24, None)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp26, None)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sm/csms5uhj4owwysmjghdu7jeo5xdocmmhjvmiqfdslismeevrhd5z.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
triton_red_fused_mean_11 = async_compile.triton('triton_red_fused_mean_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/sj/csjm4vyhkvxruyvm4pgjndihiai2olosqcafk5sqlsw6ivleq3ww.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
triton_per_fused_mean_12 = async_compile.triton('triton_per_fused_mean_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (r1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/a5/ca5pl43anqowzcsysfthhzm724gg5huhkuksa5s3h4hmo6fwsp4s.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
triton_red_fused_mean_13 = async_compile.triton('triton_red_fused_mean_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/7s/c7saw6ugv7dziybj7kihjy6ipx5z2hde75ji2qqp6jrfeurkchbj.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
triton_per_fused_mean_14 = async_compile.triton('triton_per_fused_mean_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/l5/cl5f6twpvr5cvje5ay7ask2i2qtcirgtbe7c5vmhpay3tdqy6oi7.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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


# kernel path: /tmp/torchinductor_elicer/fh/cfhhgy3ltr7u36sracgpacgkgombcyjeqt5ekbkmg7lupvvouj62.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/r4/cr4ezw57ebmzoiu7l7bme4vjtpi5zmftyyg7qoy6zqlvgp3pwqdu.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_17 = async_compile.triton('triton_poi_fused_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/cd/ccddtdlgihsvojbtgdn73l256vcovoplg7obmjzcwm344brz2coc.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_10 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_18 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 64)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 256*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16512 + x0 + 256*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ik/cikeou5damfledeuorcv54mwyauykp66yeyleqqnu2ywwa7ngos6.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_11 => convolution_4
#   input_12 => relu_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_convolution_relu_19 = async_compile.triton('triton_poi_fused_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/y6/cy64qookstovo732ohmru3s6cemipdhiao5dvsiv74sjdlp64vf4.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in1_input => div_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   sub_1 => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 64)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 256*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16512 + x0 + 256*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/p5/cp5am6fqpyog7dory3pjs4ibn3x45t23xssh3wp33a6zm4whqvcn.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_15, input_16, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, pow_7, sum_5, norm_factor_4, add_4, truediv_6, pow_8, sum_6, norm_factor_5, add_5, truediv_7, sub_4, pow_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_5 => add_5
#   in1_input => div_1
#   input_15 => convolution_6
#   input_16 => relu_6
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   norm_factor_4 => sqrt_4
#   norm_factor_5 => sqrt_5
#   pow_7 => pow_7
#   pow_8 => pow_8
#   pow_9 => pow_9
#   sub_1 => sub_1
#   sub_4 => sub_4
#   sum_5 => sum_5
#   sum_6 => sum_6
#   truediv_6 => div_6
#   truediv_7 => div_7
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_6,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_6, 2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1], True), kwargs = {})
#   %sqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_5,), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_4, 1e-10), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_6, %add_4), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_19, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_8, [1], True), kwargs = {})
#   %sqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_6,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_5, 1e-10), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_19, %add_5), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_6, %div_7), kwargs = {})
#   %pow_9 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %pow_20 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 1.0), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_20, 2.0), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_6, %add_4), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_21 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_21(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (r1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = libdevice.sqrt(tmp8)
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp3, tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1e-10
    tmp18 = tmp9 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp16)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp12 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = 2.0
    tmp26 = tmp23 * tmp25
    tmp27 = tmp19 / tmp18
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp9, None)
    tl.store(in_out_ptr2 + (r1 + 256*x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + 256*x0), tmp24, None)
    tl.store(out_ptr2 + (r1 + 256*x0), tmp26, None)
    tl.store(out_ptr3 + (r1 + 256*x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/r7/cr7ugv4dfhqvxcezjy4yujqipkjo4uedmsg4csh7zz23knvsis3e.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_17 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_22 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_22(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 32)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 512*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16640 + x0 + 512*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/wa/cwag72ncs6k7blpn7bdkwt46amwmfdd3opujpjh6two2xhwatubb.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_18 => convolution_7
#   input_19 => relu_7
# Graph fragment:
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_7,), kwargs = {})
triton_poi_fused_convolution_relu_23 = async_compile.triton('triton_poi_fused_convolution_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gp/cgpxf5dvt3vwx4t2biqizqpwupkbu34s4zwr2xjry32mg33t4hrv.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_24 = async_compile.triton('triton_poi_fused_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_elicer/yd/cydvhyfiyhihor34vba7wivqdoknr66vxmulpvqcl6p35gls3uob.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in1_input => div_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   sub_1 => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 32)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 512*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16640 + x0 + 512*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/gz/cgzpijj6c5som5hpeedjxmnbqvw35repyr2ywp7ig6u7zxpap5hl.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_22, input_23, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, pow_10, sum_7, norm_factor_6, add_6, truediv_8, pow_11, sum_8, norm_factor_7, add_7, truediv_9, sub_5, pow_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_6 => add_6
#   add_7 => add_7
#   in1_input => div_1
#   input_22 => convolution_9
#   input_23 => relu_9
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_50 => convolution_21
#   input_51 => relu_21
#   input_52 => convolution_22
#   input_53 => relu_22
#   norm_factor_6 => sqrt_6
#   norm_factor_7 => sqrt_7
#   pow_10 => pow_10
#   pow_11 => pow_11
#   pow_12 => pow_12
#   sub_1 => sub_1
#   sub_5 => sub_5
#   sum_7 => sum_7
#   sum_8 => sum_8
#   truediv_8 => div_8
#   truediv_9 => div_9
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_9, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_10, [1], True), kwargs = {})
#   %sqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_7,), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_6, 1e-10), kwargs = {})
#   %div_8 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_9, %add_6), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_22, 2), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1], True), kwargs = {})
#   %sqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_8,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_7, 1e-10), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_22, %add_7), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_8, %div_9), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 1.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_18, 2.0), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_8, %add_6), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_26 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_26(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (r1 + 512*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = libdevice.sqrt(tmp8)
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp3, tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1e-10
    tmp18 = tmp9 + tmp17
    tmp19 = tmp4 / tmp18
    tmp20 = libdevice.sqrt(tmp16)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp12 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = 2.0
    tmp26 = tmp23 * tmp25
    tmp27 = tmp19 / tmp18
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp9, None)
    tl.store(in_out_ptr2 + (r1 + 512*x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + 512*x0), tmp24, None)
    tl.store(out_ptr2 + (r1 + 512*x0), tmp26, None)
    tl.store(out_ptr3 + (r1 + 512*x0), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/hn/chncs5m2hwkum4nb53prjfa4q4ygw4lol53u2ab5ff7ichjicv2q.py
# Topologically Sorted Source Nodes: [mean_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_3 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_29, [2, 3], True), kwargs = {})
triton_per_fused_mean_27 = async_compile.triton('triton_per_fused_mean_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_27(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 2
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/6g/c6gvoykhw2sas3ivpcjgoekgubw5b7pzx6ukfheemdw6aeuhwfjn.py
# Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
triton_red_fused_mean_28 = async_compile.triton('triton_red_fused_mean_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_28(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/b6/cb6h2xwckgqiksf3kxaazpnr7f2lvnlriztrv6azazelrjsuenyo.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_24 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_29 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_29(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 16)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 1024*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16896 + x0 + 1024*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/fh/cfhxk2jvu33z36ca37kf7iymvl4ai2f46idwrdda6ldcrvc3vs57.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_25 => convolution_10
#   input_26 => relu_10
# Graph fragment:
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_25, %primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_10,), kwargs = {})
triton_poi_fused_convolution_relu_30 = async_compile.triton('triton_poi_fused_convolution_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/qq/cqqrqxlajovd77x3fcpwebpnrbiasn5wmihigqzxc73rzxus6vah.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   in1_input => div_1
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_50 => convolution_21
#   input_51 => relu_21
#   input_52 => convolution_22
#   input_53 => relu_22
#   input_54 => _low_memory_max_pool2d_with_offsets_7
#   sub_1 => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31 = async_compile.triton('triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 16)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 32768*x2), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 32768*x2), None)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 1024*x1 + 32768*x2), None)
    tmp5 = tl.load(in_ptr0 + (16896 + x0 + 1024*x1 + 32768*x2), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/3v/c3vky7qnxbkhsklmopn6m3lgrmd63a6ictuxncvla7x56rf5tryi.py
# Topologically Sorted Source Nodes: [sub_1, in1_input, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, pow_13, sum_9, norm_factor_8, add_8, truediv_10, pow_14, sum_10, norm_factor_9, add_9, truediv_11, sub_6, pow_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
# Source node to ATen node mapping:
#   add_8 => add_8
#   add_9 => add_9
#   in1_input => div_1
#   input_29 => convolution_12
#   input_30 => relu_12
#   input_31 => convolution_13
#   input_32 => relu_13
#   input_33 => convolution_14
#   input_34 => relu_14
#   input_35 => _low_memory_max_pool2d_with_offsets_4
#   input_36 => convolution_15
#   input_37 => relu_15
#   input_38 => convolution_16
#   input_39 => relu_16
#   input_40 => _low_memory_max_pool2d_with_offsets_5
#   input_41 => convolution_17
#   input_42 => relu_17
#   input_43 => convolution_18
#   input_44 => relu_18
#   input_45 => convolution_19
#   input_46 => relu_19
#   input_47 => _low_memory_max_pool2d_with_offsets_6
#   input_48 => convolution_20
#   input_49 => relu_20
#   input_50 => convolution_21
#   input_51 => relu_21
#   input_52 => convolution_22
#   input_53 => relu_22
#   input_54 => _low_memory_max_pool2d_with_offsets_7
#   input_55 => convolution_23
#   input_56 => relu_23
#   input_57 => convolution_24
#   input_58 => relu_24
#   input_59 => convolution_25
#   input_60 => relu_25
#   norm_factor_8 => sqrt_8
#   norm_factor_9 => sqrt_9
#   pow_13 => pow_13
#   pow_14 => pow_14
#   pow_15 => pow_15
#   sub_1 => sub_1
#   sub_6 => sub_6
#   sum_10 => sum_10
#   sum_9 => sum_9
#   truediv_10 => div_10
#   truediv_11 => div_11
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %primals_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %primals_3), kwargs = {})
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_29, %primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_5, %primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_7, %primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_14,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_8, %primals_9, %primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_11, %primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_10, %primals_13, %primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_15, %primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
#   %convolution_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_17, %primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_19 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_6 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_12, %primals_19, %primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_20,), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_21, %primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_21,), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %primals_23, %primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_22,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_7 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_14, %primals_25, %primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_23,), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %primals_27, %primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %primals_29, %primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_12, 2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1], True), kwargs = {})
#   %sqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_9,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_8, 1e-10), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_12, %add_8), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_25, 2), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_14, [1], True), kwargs = {})
#   %sqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_10,), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_9, 1e-10), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%relu_25, %add_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %div_11), kwargs = {})
#   %pow_15 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_6, 2), kwargs = {})
triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_32 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_32(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, out_ptr0, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (r1 + 512*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = libdevice.sqrt(tmp8)
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp3, tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = 1e-10
    tmp19 = tmp17 + tmp18
    tmp20 = tmp9 + tmp18
    tmp21 = tmp4 / tmp20
    tmp22 = tmp12 / tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp23 * tmp23
    tl.store(in_out_ptr0 + (r1 + 512*x0), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp9, None)
    tl.store(in_out_ptr2 + (r1 + 512*x0), tmp11, None)
    tl.debug_barrier()
    tl.store(in_out_ptr3 + (x0), tmp19, None)
    tl.store(out_ptr0 + (r1 + 512*x0), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/wf/cwfkvgry7q2wkflenyfwqykpebohe3qm4hvbeqcpodtenmg37kfq.py
# Topologically Sorted Source Nodes: [mean, mean_1, mean_2, mean_3, mean_4, val, val_1, val_2, val_3, val_4], Original ATen: [aten.mean, aten.add]
# Source node to ATen node mapping:
#   mean => mean
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mean_4 => mean_4
#   val => add_10
#   val_1 => add_11
#   val_2 => add_12
#   val_3 => add_13
#   val_4 => add_14
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_26, [2, 3], True), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_27, [2, 3], True), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_29, [2, 3], True), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_30, [2, 3], True), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 0), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mean_1), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mean_2), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mean_3), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mean_4), kwargs = {})
triton_per_fused_add_mean_33 = async_compile.triton('triton_per_fused_add_mean_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2
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
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp4 = tl.load(in_out_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp5 = 65536.0
    tmp6 = tmp4 / tmp5
    tmp7 = 0.0
    tmp8 = tmp6 + tmp7
    tmp10 = 16384.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp8 + tmp11
    tmp14 = 4096.0
    tmp15 = tmp13 / tmp14
    tmp16 = tmp12 + tmp15
    tmp18 = 1024.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = 256.0
    tmp22 = tmp3 / tmp21
    tmp23 = tmp20 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (2, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_4, (2, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(primals_5, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_33, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_34, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_35, (1, 512, 1, 1), (512, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_11, buf3, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_11
        buf2 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_9, buf2, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del primals_9
        buf1 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_7
        buf0 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_5, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_5
        buf13 = empty_strided_cuda((2, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        buf48 = empty_strided_cuda((2, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, in0_input, sub_1, in1_input], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_4.run(primals_2, primals_1, primals_3, primals_4, buf13, buf48, 6, 65536, grid=grid(6, 65536), stream=stream0)
        del primals_1
        del primals_2
        del primals_4
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (2, 64, 256, 256), (4194304, 1, 16384, 64))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_5.run(buf15, primals_6, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (2, 64, 256, 256), (4194304, 1, 16384, 64))
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31], Original ATen: [aten.sub, aten.div, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (2, 64, 256, 256), (4194304, 1, 16384, 64))
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_5.run(buf50, primals_6, 8388608, grid=grid(8388608), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu]
        buf51 = extern_kernels.convolution(buf50, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (2, 64, 256, 256), (4194304, 1, 16384, 64))
        buf17 = buf16; del buf16  # reuse
        buf79 = empty_strided_cuda((2, 1, 256, 256), (65536, 131072, 256, 1), torch.float32)
        buf80 = reinterpret_tensor(buf79, (2, 1, 256, 256), (65536, 1, 256, 1), 0); del buf79  # reuse
        buf52 = buf51; del buf51  # reuse
        buf82 = buf50; del buf50  # reuse
        buf119 = empty_strided_cuda((2, 64, 256, 256), (4194304, 1, 16384, 64), torch.float32)
        buf120 = empty_strided_cuda((2, 64, 256, 256), (4194304, 1, 16384, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_3, input_4, input_31, input_32, input_33, input_34, pow_1, sum_1, norm_factor, add, truediv_2, pow_2, sum_2, norm_factor_1, add_1, truediv_3, sub_2, pow_3], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_mul_pow_relu_sqrt_sub_sum_6.run(buf17, buf80, buf52, primals_8, buf82, buf119, buf120, 131072, 64, grid=grid(131072), stream=stream0)
        del primals_8
        buf18 = empty_strided_cuda((2, 64, 128, 128), (1048576, 1, 8192, 64), torch.float32)
        buf19 = empty_strided_cuda((2, 64, 128, 128), (1048576, 1, 8192, 64), torch.int8)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf17, buf18, buf19, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf18, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (2, 128, 128, 128), (2097152, 1, 16384, 128))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf21, primals_10, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (2, 128, 128, 128), (2097152, 1, 16384, 128))
        buf53 = empty_strided_cuda((2, 64, 128, 128), (1048576, 1, 8192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_9.run(buf52, buf53, 2097152, grid=grid(2097152), stream=stream0)
        del buf52
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf54 = extern_kernels.convolution(buf53, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (2, 128, 128, 128), (2097152, 1, 16384, 128))
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf55, primals_10, 4194304, grid=grid(4194304), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf56 = extern_kernels.convolution(buf55, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (2, 128, 128, 128), (2097152, 1, 16384, 128))
        buf23 = buf22; del buf22  # reuse
        buf83 = empty_strided_cuda((2, 1, 128, 128), (16384, 32768, 128, 1), torch.float32)
        buf84 = reinterpret_tensor(buf83, (2, 1, 128, 128), (16384, 1, 128, 1), 0); del buf83  # reuse
        buf57 = buf56; del buf56  # reuse
        buf86 = buf55; del buf55  # reuse
        buf117 = empty_strided_cuda((2, 128, 128, 128), (2097152, 1, 16384, 128), torch.float32)
        buf118 = empty_strided_cuda((2, 128, 128, 128), (2097152, 1, 16384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_8, input_9, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, pow_4, sum_3, norm_factor_2, add_2, truediv_4, pow_5, sum_4, norm_factor_3, add_3, truediv_5, sub_3, pow_6], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_10.run(buf23, buf84, buf57, primals_12, buf86, buf117, buf118, 32768, 128, grid=grid(32768), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf86, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (2, 1, 128, 128), (16384, 1, 128, 1))
        buf104 = empty_strided_cuda((2, 1, 1, 1, 2), (2, 4, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_11.run(buf103, buf104, 4, 8192, grid=grid(4), stream=stream0)
        del buf103
        buf105 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_12.run(buf104, buf105, 2, 2, grid=grid(2), stream=stream0)
        del buf104
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf82, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (2, 1, 256, 256), (65536, 1, 256, 1))
        buf101 = empty_strided_cuda((2, 1, 1, 1, 8), (8, 16, 16, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_13.run(buf100, buf101, 16, 8192, grid=grid(16), stream=stream0)
        del buf100
        buf102 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_14.run(buf101, buf102, 2, 8, grid=grid(2), stream=stream0)
        del buf101
        buf7 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(primals_19, buf7, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del primals_19
        buf6 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_17, buf6, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_17
        buf5 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(primals_15, buf5, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_15
        buf4 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(primals_13, buf4, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del primals_13
        buf24 = empty_strided_cuda((2, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        buf25 = empty_strided_cuda((2, 128, 64, 64), (524288, 1, 8192, 128), torch.int8)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_18.run(buf23, buf24, buf25, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf27, primals_14, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf29, primals_16, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf58 = empty_strided_cuda((2, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_20.run(buf57, buf58, 1048576, grid=grid(1048576), stream=stream0)
        del buf57
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf59 = extern_kernels.convolution(buf58, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf60, primals_14, 2097152, grid=grid(2097152), stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf61 = extern_kernels.convolution(buf60, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf62, primals_16, 2097152, grid=grid(2097152), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf63 = extern_kernels.convolution(buf62, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (2, 256, 64, 64), (1048576, 1, 16384, 256))
        buf31 = buf30; del buf30  # reuse
        buf87 = empty_strided_cuda((2, 1, 64, 64), (4096, 8192, 64, 1), torch.float32)
        buf88 = reinterpret_tensor(buf87, (2, 1, 64, 64), (4096, 1, 64, 1), 0); del buf87  # reuse
        buf64 = buf63; del buf63  # reuse
        buf90 = buf62; del buf62  # reuse
        buf115 = buf60; del buf60  # reuse
        buf116 = reinterpret_tensor(buf53, (2, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_15, input_16, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, pow_7, sum_5, norm_factor_4, add_4, truediv_6, pow_8, sum_6, norm_factor_5, add_5, truediv_7, sub_4, pow_9], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_21.run(buf31, buf88, buf64, primals_18, buf90, buf115, buf116, 8192, 256, grid=grid(8192), stream=stream0)
        del primals_18
        buf32 = empty_strided_cuda((2, 256, 32, 32), (262144, 1, 8192, 256), torch.float32)
        buf33 = empty_strided_cuda((2, 256, 32, 32), (262144, 1, 8192, 256), torch.int8)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_22.run(buf31, buf32, buf33, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (2, 512, 32, 32), (524288, 1, 16384, 512))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_23.run(buf35, primals_20, 1048576, grid=grid(1048576), stream=stream0)
        buf8 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_21, buf8, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (2, 512, 32, 32), (524288, 1, 16384, 512))
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_23.run(buf37, primals_22, 1048576, grid=grid(1048576), stream=stream0)
        buf9 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_23, buf9, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (2, 512, 32, 32), (524288, 1, 16384, 512))
        buf65 = empty_strided_cuda((2, 256, 32, 32), (262144, 1, 8192, 256), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_25.run(buf64, buf65, 524288, grid=grid(524288), stream=stream0)
        del buf64
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf66 = extern_kernels.convolution(buf65, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (2, 512, 32, 32), (524288, 1, 16384, 512))
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_23.run(buf67, primals_20, 1048576, grid=grid(1048576), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf68 = extern_kernels.convolution(buf67, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (2, 512, 32, 32), (524288, 1, 16384, 512))
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_23.run(buf69, primals_22, 1048576, grid=grid(1048576), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf70 = extern_kernels.convolution(buf69, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (2, 512, 32, 32), (524288, 1, 16384, 512))
        buf39 = buf38; del buf38  # reuse
        buf91 = empty_strided_cuda((2, 1, 32, 32), (1024, 2048, 32, 1), torch.float32)
        buf92 = reinterpret_tensor(buf91, (2, 1, 32, 32), (1024, 1, 32, 1), 0); del buf91  # reuse
        buf71 = buf70; del buf70  # reuse
        buf94 = buf69; del buf69  # reuse
        buf113 = buf67; del buf67  # reuse
        buf114 = reinterpret_tensor(buf58, (2, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_22, input_23, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, pow_10, sum_7, norm_factor_6, add_6, truediv_8, pow_11, sum_8, norm_factor_7, add_7, truediv_9, sub_5, pow_12], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_mul_pow_relu_sqrt_sub_sum_26.run(buf39, buf92, buf71, primals_24, buf94, buf113, buf114, 2048, 512, grid=grid(2048), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf94, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (2, 1, 32, 32), (1024, 1, 32, 1))
        buf109 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [mean_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_27.run(buf108, buf109, 2, 1024, grid=grid(2), stream=stream0)
        del buf108
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf90, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (2, 1, 64, 64), (4096, 1, 64, 1))
        buf107 = empty_strided_cuda((2, 1, 1, 1), (1, 2, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [mean_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_28.run(buf106, buf107, 2, 4096, grid=grid(2), stream=stream0)
        del buf106
        buf40 = empty_strided_cuda((2, 512, 16, 16), (131072, 1, 8192, 512), torch.float32)
        buf41 = empty_strided_cuda((2, 512, 16, 16), (131072, 1, 8192, 512), torch.int8)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_29.run(buf39, buf40, buf41, 262144, grid=grid(262144), stream=stream0)
        buf10 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_25, buf10, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (2, 512, 16, 16), (131072, 1, 8192, 512))
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_30.run(buf43, primals_26, 262144, grid=grid(262144), stream=stream0)
        buf11 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_27, buf11, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (2, 512, 16, 16), (131072, 1, 8192, 512))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_30.run(buf45, primals_28, 262144, grid=grid(262144), stream=stream0)
        buf12 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(primals_29, buf12, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (2, 512, 16, 16), (131072, 1, 8192, 512))
        buf72 = empty_strided_cuda((2, 512, 16, 16), (131072, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_max_pool2d_with_indices_relu_sub_31.run(buf71, buf72, 262144, grid=grid(262144), stream=stream0)
        del buf71
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf73 = extern_kernels.convolution(buf72, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (2, 512, 16, 16), (131072, 1, 8192, 512))
        del buf72
        buf74 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_30.run(buf74, primals_26, 262144, grid=grid(262144), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf75 = extern_kernels.convolution(buf74, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (2, 512, 16, 16), (131072, 1, 8192, 512))
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_30.run(buf76, primals_28, 262144, grid=grid(262144), stream=stream0)
        del primals_28
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        buf77 = extern_kernels.convolution(buf76, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (2, 512, 16, 16), (131072, 1, 8192, 512))
        buf47 = buf46; del buf46  # reuse
        buf95 = empty_strided_cuda((2, 1, 16, 16), (256, 512, 16, 1), torch.float32)
        buf96 = reinterpret_tensor(buf95, (2, 1, 16, 16), (256, 1, 16, 1), 0); del buf95  # reuse
        buf78 = buf77; del buf77  # reuse
        buf97 = empty_strided_cuda((2, 1, 16, 16), (256, 512, 16, 1), torch.float32)
        buf98 = reinterpret_tensor(buf97, (2, 1, 16, 16), (256, 1, 16, 1), 0); del buf97  # reuse
        buf99 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [sub_1, in1_input, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, pow_13, sum_9, norm_factor_8, add_8, truediv_10, pow_14, sum_10, norm_factor_9, add_9, truediv_11, sub_6, pow_15], Original ATen: [aten.sub, aten.div, aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.pow, aten.sum, aten.sqrt, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool2d_with_indices_pow_relu_sqrt_sub_sum_32.run(buf47, buf96, buf78, buf98, primals_30, buf99, 512, 512, grid=grid(512), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf99, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (2, 1, 16, 16), (256, 1, 16, 1))
        buf112 = reinterpret_tensor(buf102, (2, 1, 1, 1), (1, 1, 1, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [mean, mean_1, mean_2, mean_3, mean_4, val, val_1, val_2, val_3, val_4], Original ATen: [aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_33.run(buf112, buf110, buf105, buf107, buf109, 2, 256, grid=grid(2), stream=stream0)
        del buf105
        del buf107
        del buf109
        del buf110
    return (buf112, primals_3, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, primals_31, primals_32, primals_33, primals_34, primals_35, buf13, buf15, buf17, buf18, buf19, buf21, buf23, buf24, buf25, buf27, buf29, buf31, buf32, buf33, buf35, buf37, buf39, buf40, buf41, buf43, buf45, buf47, buf78, buf80, buf82, buf84, buf86, buf88, buf90, buf92, buf94, buf96, buf98, buf99, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
