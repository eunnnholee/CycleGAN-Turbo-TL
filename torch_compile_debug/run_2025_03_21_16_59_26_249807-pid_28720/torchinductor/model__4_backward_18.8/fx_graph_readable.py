class GraphModule(torch.nn.Module):
    def forward(self, primals_156: "f32[256, 768, 3, 3]", primals_160: "f32[256, 1, 4, 4]", primals_161: "f32[1, 256, 1, 1]", primals_162: "f32[1]", primals_165: "f32[256, 768, 3, 3]", primals_169: "f32[256, 1, 4, 4]", primals_170: "f32[1, 256, 1, 1]", primals_171: "f32[1]", primals_174: "f32[256, 512]", primals_178: "f32[1, 256]", primals_179: "f32[1]", mm: "f32[2, 512]", view_187: "f32[2, 768, 7, 7]", view_188: "f32[2, 768, 7, 7]", div_1: "f32[6912]", div_2: "f32[256]", sum_6: "f32[]", div_3: "f32[256, 768, 3, 3]", constant_pad_nd_1: "f32[2, 256, 9, 9]", convolution_2: "f32[2, 256, 6, 6]", clamp_min_6: "f32[1]", sum_9: "f32[1]", div_6: "f32[1, 256, 1, 1]", convolution_3: "f32[2, 1, 3, 3]", div_7: "f32[6912]", div_8: "f32[256]", sum_18: "f32[]", div_9: "f32[256, 768, 3, 3]", constant_pad_nd_2: "f32[2, 256, 9, 9]", convolution_5: "f32[2, 256, 6, 6]", clamp_min_10: "f32[1]", sum_21: "f32[1]", div_12: "f32[1, 256, 1, 1]", convolution_6: "f32[2, 1, 3, 3]", div_13: "f32[512]", div_14: "f32[256]", sum_30: "f32[]", where_2: "f32[2, 256]", clamp_min_14: "f32[1]", sum_33: "f32[1]", addmm_49: "f32[2, 1]", permute_125: "f32[1, 256]", gt_4: "b8[2, 256, 7, 7]", gt_5: "b8[2, 256, 7, 7]", tangents_1: "f32[2, 1]", tangents_2: "f32[256, 768, 3, 3]", tangents_3: "f32[1, 256, 1, 1]", tangents_4: "f32[256, 768, 3, 3]", tangents_5: "f32[1, 256, 1, 1]", tangents_6: "f32[256, 512]", tangents_7: "f32[1, 256]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:43 in forward, code: loss_ = self.lossfn(each, target_)
        sigmoid_12: "f32[2, 1]" = torch.ops.aten.sigmoid.default(addmm_49);  addmm_49 = None
        mul_111: "f32[2, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, tangents_1);  sigmoid_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:45 in forward, code: loss_ = loss_.mean([1, 2]).reshape(-1, 1)
        view_197: "f32[2]" = torch.ops.aten.view.default(tangents_1, [2]);  tangents_1 = None
        unsqueeze_18: "f32[2, 1]" = torch.ops.aten.unsqueeze.default(view_197, 1);  view_197 = None
        unsqueeze_19: "f32[2, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        expand_33: "f32[2, 3, 3]" = torch.ops.aten.expand.default(unsqueeze_19, [2, 3, 3]);  unsqueeze_19 = None
        div_19: "f32[2, 3, 3]" = torch.ops.aten.div.Scalar(expand_33, 9);  expand_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_losses.py:43 in forward, code: loss_ = self.lossfn(each, target_)
        squeeze_13: "f32[2, 3, 3]" = torch.ops.aten.squeeze.dim(convolution_6, 1);  convolution_6 = None
        sigmoid_13: "f32[2, 3, 3]" = torch.ops.aten.sigmoid.default(squeeze_13);  squeeze_13 = None
        mul_112: "f32[2, 3, 3]" = torch.ops.aten.mul.Tensor(sigmoid_13, div_19);  sigmoid_13 = None
        squeeze_12: "f32[2, 3, 3]" = torch.ops.aten.squeeze.dim(convolution_3, 1);  convolution_3 = None
        sigmoid_14: "f32[2, 3, 3]" = torch.ops.aten.sigmoid.default(squeeze_12);  squeeze_12 = None
        mul_113: "f32[2, 3, 3]" = torch.ops.aten.mul.Tensor(sigmoid_14, div_19);  sigmoid_14 = div_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:38 in forward, code: out = self.out(h)
        mm_1: "f32[2, 256]" = torch.ops.aten.mm.default(mul_111, permute_125);  permute_125 = None
        permute_126: "f32[1, 2]" = torch.ops.aten.permute.default(mul_111, [1, 0])
        mm_2: "f32[1, 256]" = torch.ops.aten.mm.default(permute_126, where_2);  permute_126 = None
        sum_37: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [0], True);  mul_111 = None
        view_199: "f32[1]" = torch.ops.aten.view.default(sum_37, [1]);  sum_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:38 in forward, code: out = self.out(h)
        add_93: "f32[1, 256]" = torch.ops.aten.add.Tensor(tangents_7, mm_2);  tangents_7 = mm_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        pow_23: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_33, 2.0)
        sum_34: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_23, [0], True);  pow_23 = None
        pow_24: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
        clamp_min_15: "f32[1]" = torch.ops.aten.clamp_min.default(pow_24, 1e-12);  pow_24 = None
        expand_29: "f32[1]" = torch.ops.aten.expand.default(clamp_min_15, [1]);  clamp_min_15 = None
        div_17: "f32[1]" = torch.ops.aten.div.Tensor(sum_33, expand_29);  expand_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_107: "f32[1]" = torch.ops.aten.mul.Tensor(div_17, sum_33);  sum_33 = None
        sum_36: "f32[]" = torch.ops.aten.sum.default(mul_107);  mul_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_21: "f32[1, 256]" = torch.ops.aten.div.Tensor(primals_178, sum_36)
        div_22: "f32[1, 256]" = torch.ops.aten.div.Tensor(div_21, sum_36);  div_21 = None
        neg_3: "f32[1, 256]" = torch.ops.aten.neg.default(add_93)
        mul_114: "f32[1, 256]" = torch.ops.aten.mul.Tensor(neg_3, div_22);  neg_3 = div_22 = None
        div_23: "f32[1, 256]" = torch.ops.aten.div.Tensor(add_93, sum_36);  add_93 = sum_36 = None
        sum_38: "f32[]" = torch.ops.aten.sum.default(mul_114);  mul_114 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_115: "f32[1]" = torch.ops.aten.mul.Tensor(sum_38, div_17);  sum_38 = None
        view_200: "f32[1, 1]" = torch.ops.aten.view.default(mul_115, [1, 1]);  mul_115 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_194: "f32[1, 256]" = torch.ops.aten.view.default(primals_178, [1, -1]);  primals_178 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_123: "f32[256, 1]" = torch.ops.aten.permute.default(view_194, [1, 0]);  view_194 = None
        mul_104: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_123, primals_179);  permute_123 = None
        sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_104, [1]);  mul_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        expand_27: "f32[256]" = torch.ops.aten.expand.default(clamp_min_14, [256]);  clamp_min_14 = None
        div_16: "f32[256]" = torch.ops.aten.div.Tensor(sum_31, expand_27);  sum_31 = expand_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_116: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_200, div_16);  view_200 = div_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_94: "f32[1, 256]" = torch.ops.aten.add.Tensor(div_23, mul_116);  div_23 = mul_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:37 in forward, code: h = self.decoder[-1](x[-1].float())
        gt_3: "b8[2, 256]" = torch.ops.aten.gt.Scalar(where_2, 0);  where_2 = None
        mul_117: "f32[2, 256]" = torch.ops.aten.mul.Tensor(mm_1, 0.2)
        where_3: "f32[2, 256]" = torch.ops.aten.where.self(gt_3, mm_1, mul_117);  gt_3 = mm_1 = mul_117 = None
        permute_129: "f32[256, 2]" = torch.ops.aten.permute.default(where_3, [1, 0])
        mm_3: "f32[256, 512]" = torch.ops.aten.mm.default(permute_129, mm);  permute_129 = mm = None
        sum_39: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(where_3, [0], True);  where_3 = None
        view_202: "f32[256]" = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:37 in forward, code: h = self.decoder[-1](x[-1].float())
        add_95: "f32[256, 512]" = torch.ops.aten.add.Tensor(tangents_6, mm_3);  tangents_6 = mm_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_24: "f32[256, 512]" = torch.ops.aten.div.Tensor(primals_174, sum_30);  primals_174 = None
        div_25: "f32[256, 512]" = torch.ops.aten.div.Tensor(div_24, sum_30);  div_24 = None
        neg_4: "f32[256, 512]" = torch.ops.aten.neg.default(add_95)
        mul_118: "f32[256, 512]" = torch.ops.aten.mul.Tensor(neg_4, div_25);  neg_4 = div_25 = None
        div_26: "f32[256, 512]" = torch.ops.aten.div.Tensor(add_95, sum_30);  add_95 = sum_30 = None
        sum_40: "f32[]" = torch.ops.aten.sum.default(mul_118);  mul_118 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_119: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, div_14);  sum_40 = div_14 = None
        view_203: "f32[256, 1]" = torch.ops.aten.view.default(mul_119, [256, 1]);  mul_119 = None
        mul_120: "f32[256, 512]" = torch.ops.aten.mul.Tensor(view_203, div_13);  view_203 = div_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_96: "f32[256, 512]" = torch.ops.aten.add.Tensor(div_26, mul_120);  div_26 = mul_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        unsqueeze_22: "f32[2, 1, 3, 3]" = torch.ops.aten.unsqueeze.default(mul_112, 1);  mul_112 = None
        sum_41: "f32[1]" = torch.ops.aten.sum.dim_IntList(unsqueeze_22, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(unsqueeze_22, convolution_5, div_12, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  unsqueeze_22 = convolution_5 = div_12 = None
        getitem_100: "f32[2, 256, 6, 6]" = convolution_backward[0]
        getitem_101: "f32[1, 256, 1, 1]" = convolution_backward[1];  convolution_backward = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        add_97: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(tangents_5, getitem_101);  tangents_5 = getitem_101 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        pow_15: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_21, 2.0)
        sum_22: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_15, [0], True);  pow_15 = None
        pow_16: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
        clamp_min_11: "f32[1]" = torch.ops.aten.clamp_min.default(pow_16, 1e-12);  pow_16 = None
        expand_21: "f32[1]" = torch.ops.aten.expand.default(clamp_min_11, [1]);  clamp_min_11 = None
        div_11: "f32[1]" = torch.ops.aten.div.Tensor(sum_21, expand_21);  expand_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_98: "f32[1]" = torch.ops.aten.mul.Tensor(div_11, sum_21);  sum_21 = None
        sum_24: "f32[]" = torch.ops.aten.sum.default(mul_98);  mul_98 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_27: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(primals_170, sum_24)
        div_28: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(div_27, sum_24);  div_27 = None
        neg_5: "f32[1, 256, 1, 1]" = torch.ops.aten.neg.default(add_97)
        mul_121: "f32[1, 256, 1, 1]" = torch.ops.aten.mul.Tensor(neg_5, div_28);  neg_5 = div_28 = None
        div_29: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(add_97, sum_24);  add_97 = sum_24 = None
        sum_42: "f32[]" = torch.ops.aten.sum.default(mul_121);  mul_121 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_122: "f32[1]" = torch.ops.aten.mul.Tensor(sum_42, div_11);  sum_42 = None
        view_205: "f32[1, 1]" = torch.ops.aten.view.default(mul_122, [1, 1]);  mul_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_192: "f32[1, 256]" = torch.ops.aten.view.default(primals_170, [1, -1]);  primals_170 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_120: "f32[256, 1]" = torch.ops.aten.permute.default(view_192, [1, 0]);  view_192 = None
        mul_95: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_120, primals_171);  permute_120 = None
        sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_95, [1]);  mul_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        expand_19: "f32[256]" = torch.ops.aten.expand.default(clamp_min_10, [256]);  clamp_min_10 = None
        div_10: "f32[256]" = torch.ops.aten.div.Tensor(sum_19, expand_19);  sum_19 = expand_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_123: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_205, div_10);  view_205 = div_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_206: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(mul_123, [1, 256, 1, 1]);  mul_123 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_98: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(div_29, view_206);  div_29 = view_206 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/blurpool.py:53 in forward, code: return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_100, constant_pad_nd_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False]);  getitem_100 = constant_pad_nd_2 = primals_169 = None
        getitem_103: "f32[2, 256, 9, 9]" = convolution_backward_1[0];  convolution_backward_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[2, 256, 7, 7]" = torch.ops.aten.constant_pad_nd.default(getitem_103, [-1, -1, -1, -1]);  getitem_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        mul_124: "f32[2, 256, 7, 7]" = torch.ops.aten.mul.Tensor(constant_pad_nd_3, 0.2)
        where_4: "f32[2, 256, 7, 7]" = torch.ops.aten.where.self(gt_4, constant_pad_nd_3, mul_124);  gt_4 = constant_pad_nd_3 = mul_124 = None
        sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, view_188, div_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_4 = view_188 = div_9 = None
        getitem_107: "f32[256, 768, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        add_99: "f32[256, 768, 3, 3]" = torch.ops.aten.add.Tensor(tangents_4, getitem_107);  tangents_4 = getitem_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_30: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(primals_165, sum_18);  primals_165 = None
        div_31: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(div_30, sum_18);  div_30 = None
        neg_6: "f32[256, 768, 3, 3]" = torch.ops.aten.neg.default(add_99)
        mul_125: "f32[256, 768, 3, 3]" = torch.ops.aten.mul.Tensor(neg_6, div_31);  neg_6 = div_31 = None
        div_32: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(add_99, sum_18);  add_99 = sum_18 = None
        sum_44: "f32[]" = torch.ops.aten.sum.default(mul_125);  mul_125 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_126: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, div_8);  sum_44 = div_8 = None
        view_207: "f32[256, 1]" = torch.ops.aten.view.default(mul_126, [256, 1]);  mul_126 = None
        mul_127: "f32[256, 6912]" = torch.ops.aten.mul.Tensor(view_207, div_7);  view_207 = div_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_208: "f32[256, 768, 3, 3]" = torch.ops.aten.view.default(mul_127, [256, 768, 3, 3]);  mul_127 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_100: "f32[256, 768, 3, 3]" = torch.ops.aten.add.Tensor(div_32, view_208);  div_32 = view_208 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        unsqueeze_23: "f32[2, 1, 3, 3]" = torch.ops.aten.unsqueeze.default(mul_113, 1);  mul_113 = None
        sum_45: "f32[1]" = torch.ops.aten.sum.dim_IntList(unsqueeze_23, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(unsqueeze_23, convolution_2, div_6, [1], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  unsqueeze_23 = convolution_2 = div_6 = None
        getitem_109: "f32[2, 256, 6, 6]" = convolution_backward_3[0]
        getitem_110: "f32[1, 256, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        add_101: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(tangents_3, getitem_110);  tangents_3 = getitem_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:105 in compute_weight, code: u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
        pow_7: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 2.0)
        sum_10: "f32[1]" = torch.ops.aten.sum.dim_IntList(pow_7, [0], True);  pow_7 = None
        pow_8: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
        clamp_min_7: "f32[1]" = torch.ops.aten.clamp_min.default(pow_8, 1e-12);  pow_8 = None
        expand_13: "f32[1]" = torch.ops.aten.expand.default(clamp_min_7, [1]);  clamp_min_7 = None
        div_5: "f32[1]" = torch.ops.aten.div.Tensor(sum_9, expand_13);  expand_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_89: "f32[1]" = torch.ops.aten.mul.Tensor(div_5, sum_9);  sum_9 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(mul_89);  mul_89 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_33: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(primals_161, sum_12)
        div_34: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(div_33, sum_12);  div_33 = None
        neg_7: "f32[1, 256, 1, 1]" = torch.ops.aten.neg.default(add_101)
        mul_128: "f32[1, 256, 1, 1]" = torch.ops.aten.mul.Tensor(neg_7, div_34);  neg_7 = div_34 = None
        div_35: "f32[1, 256, 1, 1]" = torch.ops.aten.div.Tensor(add_101, sum_12);  add_101 = sum_12 = None
        sum_46: "f32[]" = torch.ops.aten.sum.default(mul_128);  mul_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_129: "f32[1]" = torch.ops.aten.mul.Tensor(sum_46, div_5);  sum_46 = None
        view_209: "f32[1, 1]" = torch.ops.aten.view.default(mul_129, [1, 1]);  mul_129 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_190: "f32[1, 256]" = torch.ops.aten.view.default(primals_161, [1, -1]);  primals_161 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:103 in compute_weight, code: torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
        permute_118: "f32[256, 1]" = torch.ops.aten.permute.default(view_190, [1, 0]);  view_190 = None
        mul_86: "f32[256, 1]" = torch.ops.aten.mul.Tensor(permute_118, primals_162);  permute_118 = None
        sum_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_86, [1]);  mul_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:102 in compute_weight, code: v = F.normalize(
        expand_11: "f32[256]" = torch.ops.aten.expand.default(clamp_min_6, [256]);  clamp_min_6 = None
        div_4: "f32[256]" = torch.ops.aten.div.Tensor(sum_7, expand_11);  sum_7 = expand_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_130: "f32[1, 256]" = torch.ops.aten.mul.Tensor(view_209, div_4);  view_209 = div_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_210: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(mul_130, [1, 256, 1, 1]);  mul_130 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_102: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(div_35, view_210);  div_35 = view_210 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/blurpool.py:53 in forward, code: return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(getitem_109, constant_pad_nd_1, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 256, [True, False, False]);  getitem_109 = constant_pad_nd_1 = primals_160 = None
        getitem_112: "f32[2, 256, 9, 9]" = convolution_backward_4[0];  convolution_backward_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f32[2, 256, 7, 7]" = torch.ops.aten.constant_pad_nd.default(getitem_112, [-1, -1, -1, -1]);  getitem_112 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        mul_131: "f32[2, 256, 7, 7]" = torch.ops.aten.mul.Tensor(constant_pad_nd_4, 0.2)
        where_5: "f32[2, 256, 7, 7]" = torch.ops.aten.where.self(gt_5, constant_pad_nd_4, mul_131);  gt_5 = constant_pad_nd_4 = mul_131 = None
        sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, view_187, div_3, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  where_5 = view_187 = div_3 = None
        getitem_116: "f32[256, 768, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/vision_aided_loss/cv_discriminator.py:35 in forward, code: final_pred.append(self.decoder[i](x[i]).squeeze(1))
        add_103: "f32[256, 768, 3, 3]" = torch.ops.aten.add.Tensor(tangents_2, getitem_116);  tangents_2 = getitem_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:112 in compute_weight, code: weight = weight / sigma
        div_36: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(primals_156, sum_6);  primals_156 = None
        div_37: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(div_36, sum_6);  div_36 = None
        neg_8: "f32[256, 768, 3, 3]" = torch.ops.aten.neg.default(add_103)
        mul_132: "f32[256, 768, 3, 3]" = torch.ops.aten.mul.Tensor(neg_8, div_37);  neg_8 = div_37 = None
        div_38: "f32[256, 768, 3, 3]" = torch.ops.aten.div.Tensor(add_103, sum_6);  add_103 = sum_6 = None
        sum_48: "f32[]" = torch.ops.aten.sum.default(mul_132);  mul_132 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:111 in compute_weight, code: sigma = torch.dot(u, torch.mv(weight_mat, v))
        mul_133: "f32[256]" = torch.ops.aten.mul.Tensor(sum_48, div_2);  sum_48 = div_2 = None
        view_211: "f32[256, 1]" = torch.ops.aten.view.default(mul_133, [256, 1]);  mul_133 = None
        mul_134: "f32[256, 6912]" = torch.ops.aten.mul.Tensor(view_211, div_1);  view_211 = div_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        view_212: "f32[256, 768, 3, 3]" = torch.ops.aten.view.default(mul_134, [256, 768, 3, 3]);  mul_134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/utils/spectral_norm.py:59 in reshape_weight_to_matrix, code: return weight_mat.reshape(height, -1)
        add_104: "f32[256, 768, 3, 3]" = torch.ops.aten.add.Tensor(div_38, view_212);  div_38 = view_212 = None
        
        # No stacktrace found for following nodes
        copy__2: "f32[1]" = torch.ops.aten.copy_.default(primals_162, div_5);  primals_162 = div_5 = copy__2 = None
        copy__6: "f32[1]" = torch.ops.aten.copy_.default(primals_171, div_11);  primals_171 = div_11 = copy__6 = None
        copy__10: "f32[1]" = torch.ops.aten.copy_.default(primals_179, div_17);  primals_179 = div_17 = copy__10 = None
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, add_104, None, None, sum_47, None, add_102, None, None, sum_45, add_100, None, None, sum_43, None, add_98, None, None, sum_41, add_96, None, None, view_202, add_94, None, None, view_199)
        