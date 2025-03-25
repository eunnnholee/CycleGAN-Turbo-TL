class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[1, 3, 1, 1]", primals_5: "f32[64, 3, 3, 3]", primals_7: "f32[64, 64, 3, 3]", primals_9: "f32[128, 64, 3, 3]", primals_11: "f32[128, 128, 3, 3]", primals_13: "f32[256, 128, 3, 3]", primals_15: "f32[256, 256, 3, 3]", primals_17: "f32[256, 256, 3, 3]", primals_19: "f32[512, 256, 3, 3]", primals_21: "f32[512, 512, 3, 3]", primals_23: "f32[512, 512, 3, 3]", primals_25: "f32[512, 512, 3, 3]", primals_27: "f32[512, 512, 3, 3]", primals_29: "f32[512, 512, 3, 3]", primals_31: "f32[1, 64, 1, 1]", primals_32: "f32[1, 128, 1, 1]", primals_33: "f32[1, 256, 1, 1]", primals_34: "f32[1, 512, 1, 1]", primals_35: "f32[1, 512, 1, 1]", div: "f32[2, 3, 256, 256]", relu: "f32[2, 64, 256, 256]", relu_1: "f32[2, 64, 256, 256]", getitem: "f32[2, 64, 128, 128]", getitem_1: "i8[2, 64, 128, 128]", relu_2: "f32[2, 128, 128, 128]", relu_3: "f32[2, 128, 128, 128]", getitem_2: "f32[2, 128, 64, 64]", getitem_3: "i8[2, 128, 64, 64]", relu_4: "f32[2, 256, 64, 64]", relu_5: "f32[2, 256, 64, 64]", relu_6: "f32[2, 256, 64, 64]", getitem_4: "f32[2, 256, 32, 32]", getitem_5: "i8[2, 256, 32, 32]", relu_7: "f32[2, 512, 32, 32]", relu_8: "f32[2, 512, 32, 32]", relu_9: "f32[2, 512, 32, 32]", getitem_6: "f32[2, 512, 16, 16]", getitem_7: "i8[2, 512, 16, 16]", relu_10: "f32[2, 512, 16, 16]", relu_11: "f32[2, 512, 16, 16]", convolution_12: "f32[2, 512, 16, 16]", convolution_25: "f32[2, 512, 16, 16]", sqrt: "f32[2, 1, 256, 256]", pow_3: "f32[2, 64, 256, 256]", sqrt_2: "f32[2, 1, 128, 128]", pow_6: "f32[2, 128, 128, 128]", sqrt_4: "f32[2, 1, 64, 64]", pow_9: "f32[2, 256, 64, 64]", sqrt_6: "f32[2, 1, 32, 32]", pow_12: "f32[2, 512, 32, 32]", sqrt_8: "f32[2, 1, 16, 16]", add_9: "f32[2, 1, 16, 16]", pow_15: "f32[2, 512, 16, 16]", mul_6: "f32[2, 512, 32, 32]", div_22: "f32[2, 512, 32, 32]", mul_12: "f32[2, 256, 64, 64]", div_26: "f32[2, 256, 64, 64]", mul_18: "f32[2, 128, 128, 128]", div_30: "f32[2, 128, 128, 128]", mul_24: "f32[2, 64, 256, 256]", div_34: "f32[2, 64, 256, 256]", tangents_1: "f32[2, 1, 1, 1]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        expand: "f32[2, 1, 16, 16]" = torch.ops.aten.expand.default(tangents_1, [2, 1, 16, 16])
        div_12: "f32[2, 1, 16, 16]" = torch.ops.aten.div.Scalar(expand, 256);  expand = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(div_12, pow_15, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_12 = pow_15 = primals_35 = None
        getitem_16: "f32[2, 512, 16, 16]" = convolution_backward[0];  convolution_backward = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        expand_1: "f32[2, 1, 32, 32]" = torch.ops.aten.expand.default(tangents_1, [2, 1, 32, 32])
        div_13: "f32[2, 1, 32, 32]" = torch.ops.aten.div.Scalar(expand_1, 1024);  expand_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(div_13, pow_12, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_13 = pow_12 = primals_34 = None
        getitem_19: "f32[2, 512, 32, 32]" = convolution_backward_1[0];  convolution_backward_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        expand_2: "f32[2, 1, 64, 64]" = torch.ops.aten.expand.default(tangents_1, [2, 1, 64, 64])
        div_14: "f32[2, 1, 64, 64]" = torch.ops.aten.div.Scalar(expand_2, 4096);  expand_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(div_14, pow_9, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_14 = pow_9 = primals_33 = None
        getitem_22: "f32[2, 256, 64, 64]" = convolution_backward_2[0];  convolution_backward_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        expand_3: "f32[2, 1, 128, 128]" = torch.ops.aten.expand.default(tangents_1, [2, 1, 128, 128])
        div_15: "f32[2, 1, 128, 128]" = torch.ops.aten.div.Scalar(expand_3, 16384);  expand_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(div_15, pow_6, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_15 = pow_6 = primals_32 = None
        getitem_25: "f32[2, 128, 128, 128]" = convolution_backward_3[0];  convolution_backward_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        expand_4: "f32[2, 1, 256, 256]" = torch.ops.aten.expand.default(tangents_1, [2, 1, 256, 256]);  tangents_1 = None
        div_16: "f32[2, 1, 256, 256]" = torch.ops.aten.div.Scalar(expand_4, 65536);  expand_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(div_16, pow_3, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]);  div_16 = pow_3 = primals_31 = None
        getitem_28: "f32[2, 64, 256, 256]" = convolution_backward_4[0];  convolution_backward_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:129 in forward, code: h = self.slice5(h)
        relu_12: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        relu_25: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_8: "f32[2, 1, 16, 16]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-10)
        div_10: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(relu_12, add_8)
        div_11: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(relu_25, add_9);  relu_25 = add_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_6: "f32[2, 512, 16, 16]" = torch.ops.aten.sub.Tensor(div_10, div_11);  div_11 = None
        pow_16: "f32[2, 512, 16, 16]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 1.0);  sub_6 = None
        mul: "f32[2, 512, 16, 16]" = torch.ops.aten.mul.Scalar(pow_16, 2.0);  pow_16 = None
        mul_1: "f32[2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_16, mul);  getitem_16 = mul = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        div_18: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(div_10, add_8);  div_10 = None
        neg: "f32[2, 512, 16, 16]" = torch.ops.aten.neg.default(mul_1)
        mul_2: "f32[2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(neg, div_18);  neg = div_18 = None
        div_19: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(mul_1, add_8);  mul_1 = add_8 = None
        sum_11: "f32[2, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_2, [1], True);  mul_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        mul_3: "f32[2, 1, 16, 16]" = torch.ops.aten.mul.Scalar(sqrt_8, 2);  sqrt_8 = None
        div_20: "f32[2, 1, 16, 16]" = torch.ops.aten.div.Tensor(sum_11, mul_3);  sum_11 = mul_3 = None
        expand_5: "f32[2, 512, 16, 16]" = torch.ops.aten.expand.default(div_20, [2, 512, 16, 16]);  div_20 = None
        pow_17: "f32[2, 512, 16, 16]" = torch.ops.aten.pow.Tensor_Scalar(relu_12, 1.0)
        mul_4: "f32[2, 512, 16, 16]" = torch.ops.aten.mul.Scalar(pow_17, 2.0);  pow_17 = None
        mul_5: "f32[2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(expand_5, mul_4);  expand_5 = mul_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        add_15: "f32[2, 512, 16, 16]" = torch.ops.aten.add.Tensor(div_19, mul_5);  div_19 = mul_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        mul_7: "f32[2, 512, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_19, mul_6);  getitem_19 = mul_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        neg_1: "f32[2, 512, 32, 32]" = torch.ops.aten.neg.default(mul_7)
        mul_8: "f32[2, 512, 32, 32]" = torch.ops.aten.mul.Tensor(neg_1, div_22);  neg_1 = div_22 = None
        add_6: "f32[2, 1, 32, 32]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-10)
        div_23: "f32[2, 512, 32, 32]" = torch.ops.aten.div.Tensor(mul_7, add_6);  mul_7 = add_6 = None
        sum_12: "f32[2, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_8, [1], True);  mul_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        mul_9: "f32[2, 1, 32, 32]" = torch.ops.aten.mul.Scalar(sqrt_6, 2);  sqrt_6 = None
        div_24: "f32[2, 1, 32, 32]" = torch.ops.aten.div.Tensor(sum_12, mul_9);  sum_12 = mul_9 = None
        expand_6: "f32[2, 512, 32, 32]" = torch.ops.aten.expand.default(div_24, [2, 512, 32, 32]);  div_24 = None
        pow_19: "f32[2, 512, 32, 32]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 1.0)
        mul_10: "f32[2, 512, 32, 32]" = torch.ops.aten.mul.Scalar(pow_19, 2.0);  pow_19 = None
        mul_11: "f32[2, 512, 32, 32]" = torch.ops.aten.mul.Tensor(expand_6, mul_10);  expand_6 = mul_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        add_16: "f32[2, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_23, mul_11);  div_23 = mul_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        mul_13: "f32[2, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_22, mul_12);  getitem_22 = mul_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        neg_2: "f32[2, 256, 64, 64]" = torch.ops.aten.neg.default(mul_13)
        mul_14: "f32[2, 256, 64, 64]" = torch.ops.aten.mul.Tensor(neg_2, div_26);  neg_2 = div_26 = None
        add_4: "f32[2, 1, 64, 64]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-10)
        div_27: "f32[2, 256, 64, 64]" = torch.ops.aten.div.Tensor(mul_13, add_4);  mul_13 = add_4 = None
        sum_13: "f32[2, 1, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_14, [1], True);  mul_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        mul_15: "f32[2, 1, 64, 64]" = torch.ops.aten.mul.Scalar(sqrt_4, 2);  sqrt_4 = None
        div_28: "f32[2, 1, 64, 64]" = torch.ops.aten.div.Tensor(sum_13, mul_15);  sum_13 = mul_15 = None
        expand_7: "f32[2, 256, 64, 64]" = torch.ops.aten.expand.default(div_28, [2, 256, 64, 64]);  div_28 = None
        pow_21: "f32[2, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 1.0)
        mul_16: "f32[2, 256, 64, 64]" = torch.ops.aten.mul.Scalar(pow_21, 2.0);  pow_21 = None
        mul_17: "f32[2, 256, 64, 64]" = torch.ops.aten.mul.Tensor(expand_7, mul_16);  expand_7 = mul_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        add_17: "f32[2, 256, 64, 64]" = torch.ops.aten.add.Tensor(div_27, mul_17);  div_27 = mul_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        mul_19: "f32[2, 128, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_25, mul_18);  getitem_25 = mul_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        neg_3: "f32[2, 128, 128, 128]" = torch.ops.aten.neg.default(mul_19)
        mul_20: "f32[2, 128, 128, 128]" = torch.ops.aten.mul.Tensor(neg_3, div_30);  neg_3 = div_30 = None
        add_2: "f32[2, 1, 128, 128]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-10)
        div_31: "f32[2, 128, 128, 128]" = torch.ops.aten.div.Tensor(mul_19, add_2);  mul_19 = add_2 = None
        sum_14: "f32[2, 1, 128, 128]" = torch.ops.aten.sum.dim_IntList(mul_20, [1], True);  mul_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        mul_21: "f32[2, 1, 128, 128]" = torch.ops.aten.mul.Scalar(sqrt_2, 2);  sqrt_2 = None
        div_32: "f32[2, 1, 128, 128]" = torch.ops.aten.div.Tensor(sum_14, mul_21);  sum_14 = mul_21 = None
        expand_8: "f32[2, 128, 128, 128]" = torch.ops.aten.expand.default(div_32, [2, 128, 128, 128]);  div_32 = None
        pow_23: "f32[2, 128, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 1.0)
        mul_22: "f32[2, 128, 128, 128]" = torch.ops.aten.mul.Scalar(pow_23, 2.0);  pow_23 = None
        mul_23: "f32[2, 128, 128, 128]" = torch.ops.aten.mul.Tensor(expand_8, mul_22);  expand_8 = mul_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        add_18: "f32[2, 128, 128, 128]" = torch.ops.aten.add.Tensor(div_31, mul_23);  div_31 = mul_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        mul_25: "f32[2, 64, 256, 256]" = torch.ops.aten.mul.Tensor(getitem_28, mul_24);  getitem_28 = mul_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        neg_4: "f32[2, 64, 256, 256]" = torch.ops.aten.neg.default(mul_25)
        mul_26: "f32[2, 64, 256, 256]" = torch.ops.aten.mul.Tensor(neg_4, div_34);  neg_4 = div_34 = None
        add: "f32[2, 1, 256, 256]" = torch.ops.aten.add.Tensor(sqrt, 1e-10)
        div_35: "f32[2, 64, 256, 256]" = torch.ops.aten.div.Tensor(mul_25, add);  mul_25 = add = None
        sum_15: "f32[2, 1, 256, 256]" = torch.ops.aten.sum.dim_IntList(mul_26, [1], True);  mul_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        mul_27: "f32[2, 1, 256, 256]" = torch.ops.aten.mul.Scalar(sqrt, 2);  sqrt = None
        div_36: "f32[2, 1, 256, 256]" = torch.ops.aten.div.Tensor(sum_15, mul_27);  sum_15 = mul_27 = None
        expand_9: "f32[2, 64, 256, 256]" = torch.ops.aten.expand.default(div_36, [2, 64, 256, 256]);  div_36 = None
        pow_25: "f32[2, 64, 256, 256]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 1.0)
        mul_28: "f32[2, 64, 256, 256]" = torch.ops.aten.mul.Scalar(pow_25, 2.0);  pow_25 = None
        mul_29: "f32[2, 64, 256, 256]" = torch.ops.aten.mul.Tensor(expand_9, mul_28);  expand_9 = mul_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        add_19: "f32[2, 64, 256, 256]" = torch.ops.aten.add.Tensor(div_35, mul_29);  div_35 = mul_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:129 in forward, code: h = self.slice5(h)
        le: "b8[2, 512, 16, 16]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[2, 512, 16, 16]" = torch.ops.aten.where.self(le, full_default, add_15);  le = add_15 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where, relu_11, primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where = primals_29 = None
        getitem_31: "f32[2, 512, 16, 16]" = convolution_backward_5[0];  convolution_backward_5 = None
        le_1: "b8[2, 512, 16, 16]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_1: "f32[2, 512, 16, 16]" = torch.ops.aten.where.self(le_1, full_default, getitem_31);  le_1 = getitem_31 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_1, relu_10, primals_27, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_1 = primals_27 = None
        getitem_34: "f32[2, 512, 16, 16]" = convolution_backward_6[0];  convolution_backward_6 = None
        le_2: "b8[2, 512, 16, 16]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_2: "f32[2, 512, 16, 16]" = torch.ops.aten.where.self(le_2, full_default, getitem_34);  le_2 = getitem_34 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_2, getitem_6, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_2 = getitem_6 = primals_25 = None
        getitem_37: "f32[2, 512, 16, 16]" = convolution_backward_7[0];  convolution_backward_7 = None
        _low_memory_max_pool2d_offsets_to_indices_3: "i64[2, 512, 16, 16]" = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_7, 2, 32, [2, 2], [0, 0]);  getitem_7 = None
        max_pool2d_with_indices_backward: "f32[2, 512, 32, 32]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_37, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_3);  getitem_37 = _low_memory_max_pool2d_offsets_to_indices_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:129 in forward, code: h = self.slice5(h)
        add_20: "f32[2, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_16, max_pool2d_with_indices_backward);  add_16 = max_pool2d_with_indices_backward = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:127 in forward, code: h = self.slice4(h)
        le_3: "b8[2, 512, 32, 32]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_3: "f32[2, 512, 32, 32]" = torch.ops.aten.where.self(le_3, full_default, add_20);  le_3 = add_20 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_3, relu_8, primals_23, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_3 = primals_23 = None
        getitem_40: "f32[2, 512, 32, 32]" = convolution_backward_8[0];  convolution_backward_8 = None
        le_4: "b8[2, 512, 32, 32]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_4: "f32[2, 512, 32, 32]" = torch.ops.aten.where.self(le_4, full_default, getitem_40);  le_4 = getitem_40 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_4, relu_7, primals_21, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_4 = primals_21 = None
        getitem_43: "f32[2, 512, 32, 32]" = convolution_backward_9[0];  convolution_backward_9 = None
        le_5: "b8[2, 512, 32, 32]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_5: "f32[2, 512, 32, 32]" = torch.ops.aten.where.self(le_5, full_default, getitem_43);  le_5 = getitem_43 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_5, getitem_4, primals_19, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_5 = getitem_4 = primals_19 = None
        getitem_46: "f32[2, 256, 32, 32]" = convolution_backward_10[0];  convolution_backward_10 = None
        _low_memory_max_pool2d_offsets_to_indices_2: "i64[2, 256, 32, 32]" = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_5, 2, 64, [2, 2], [0, 0]);  getitem_5 = None
        max_pool2d_with_indices_backward_1: "f32[2, 256, 64, 64]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_46, relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_2);  getitem_46 = _low_memory_max_pool2d_offsets_to_indices_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:127 in forward, code: h = self.slice4(h)
        add_21: "f32[2, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_17, max_pool2d_with_indices_backward_1);  add_17 = max_pool2d_with_indices_backward_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:125 in forward, code: h = self.slice3(h)
        le_6: "b8[2, 256, 64, 64]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_6: "f32[2, 256, 64, 64]" = torch.ops.aten.where.self(le_6, full_default, add_21);  le_6 = add_21 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_6, relu_5, primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_6 = primals_17 = None
        getitem_49: "f32[2, 256, 64, 64]" = convolution_backward_11[0];  convolution_backward_11 = None
        le_7: "b8[2, 256, 64, 64]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_7: "f32[2, 256, 64, 64]" = torch.ops.aten.where.self(le_7, full_default, getitem_49);  le_7 = getitem_49 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_7, relu_4, primals_15, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_7 = primals_15 = None
        getitem_52: "f32[2, 256, 64, 64]" = convolution_backward_12[0];  convolution_backward_12 = None
        le_8: "b8[2, 256, 64, 64]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_8: "f32[2, 256, 64, 64]" = torch.ops.aten.where.self(le_8, full_default, getitem_52);  le_8 = getitem_52 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_8, getitem_2, primals_13, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_8 = getitem_2 = primals_13 = None
        getitem_55: "f32[2, 128, 64, 64]" = convolution_backward_13[0];  convolution_backward_13 = None
        _low_memory_max_pool2d_offsets_to_indices_1: "i64[2, 128, 64, 64]" = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_3, 2, 128, [2, 2], [0, 0]);  getitem_3 = None
        max_pool2d_with_indices_backward_2: "f32[2, 128, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_55, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices_1);  getitem_55 = _low_memory_max_pool2d_offsets_to_indices_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:125 in forward, code: h = self.slice3(h)
        add_22: "f32[2, 128, 128, 128]" = torch.ops.aten.add.Tensor(add_18, max_pool2d_with_indices_backward_2);  add_18 = max_pool2d_with_indices_backward_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:123 in forward, code: h = self.slice2(h)
        le_9: "b8[2, 128, 128, 128]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_9: "f32[2, 128, 128, 128]" = torch.ops.aten.where.self(le_9, full_default, add_22);  le_9 = add_22 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_9, relu_2, primals_11, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_9 = primals_11 = None
        getitem_58: "f32[2, 128, 128, 128]" = convolution_backward_14[0];  convolution_backward_14 = None
        le_10: "b8[2, 128, 128, 128]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_10: "f32[2, 128, 128, 128]" = torch.ops.aten.where.self(le_10, full_default, getitem_58);  le_10 = getitem_58 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_10, getitem, primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_10 = getitem = primals_9 = None
        getitem_61: "f32[2, 64, 128, 128]" = convolution_backward_15[0];  convolution_backward_15 = None
        _low_memory_max_pool2d_offsets_to_indices: "i64[2, 64, 128, 128]" = torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default(getitem_1, 2, 256, [2, 2], [0, 0]);  getitem_1 = None
        max_pool2d_with_indices_backward_3: "f32[2, 64, 256, 256]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_61, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, _low_memory_max_pool2d_offsets_to_indices);  getitem_61 = _low_memory_max_pool2d_offsets_to_indices = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:123 in forward, code: h = self.slice2(h)
        add_23: "f32[2, 64, 256, 256]" = torch.ops.aten.add.Tensor(add_19, max_pool2d_with_indices_backward_3);  add_19 = max_pool2d_with_indices_backward_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:121 in forward, code: h = self.slice1(X)
        le_11: "b8[2, 64, 256, 256]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_11: "f32[2, 64, 256, 256]" = torch.ops.aten.where.self(le_11, full_default, add_23);  le_11 = add_23 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_11, relu, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_11 = primals_7 = None
        getitem_64: "f32[2, 64, 256, 256]" = convolution_backward_16[0];  convolution_backward_16 = None
        le_12: "b8[2, 64, 256, 256]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_12: "f32[2, 64, 256, 256]" = torch.ops.aten.where.self(le_12, full_default, getitem_64);  le_12 = full_default = getitem_64 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_12, div, primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False]);  where_12 = div = primals_5 = None
        getitem_67: "f32[2, 3, 256, 256]" = convolution_backward_17[0];  convolution_backward_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:154 in forward, code: return (inp - self.shift) / self.scale
        div_37: "f32[2, 3, 256, 256]" = torch.ops.aten.div.Tensor(getitem_67, primals_3);  getitem_67 = primals_3 = None
        return (None, div_37, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        