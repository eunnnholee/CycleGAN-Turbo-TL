class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3, 256, 256]", primals_2: "f32[128, 3, 3, 3]", primals_3: "f32[128]", primals_4: "f32[4, 3, 3, 3]", primals_5: "f32[128, 4, 1, 1]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128, 128, 3, 3]", primals_9: "f32[128]", primals_10: "f32[4, 128, 3, 3]", primals_11: "f32[128, 4, 1, 1]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128, 128, 3, 3]", primals_15: "f32[128]", primals_16: "f32[4, 128, 3, 3]", primals_17: "f32[128, 4, 1, 1]", primals_18: "f32[128]", primals_19: "f32[128]", primals_20: "f32[128, 128, 3, 3]", primals_21: "f32[128]", primals_22: "f32[4, 128, 3, 3]", primals_23: "f32[128, 4, 1, 1]", primals_24: "f32[128]", primals_25: "f32[128]", primals_26: "f32[128, 128, 3, 3]", primals_27: "f32[128]", primals_28: "f32[4, 128, 3, 3]", primals_29: "f32[128, 4, 1, 1]", primals_30: "f32[128, 128, 3, 3]", primals_31: "f32[128]", primals_32: "f32[4, 128, 3, 3]", primals_33: "f32[128, 4, 1, 1]", primals_34: "f32[128]", primals_35: "f32[128]", primals_36: "f32[256, 128, 3, 3]", primals_37: "f32[256]", primals_38: "f32[4, 128, 3, 3]", primals_39: "f32[256, 4, 1, 1]", primals_40: "f32[256]", primals_41: "f32[256]", primals_42: "f32[256, 256, 3, 3]", primals_43: "f32[256]", primals_44: "f32[4, 256, 3, 3]", primals_45: "f32[256, 4, 1, 1]", primals_46: "f32[256, 128, 1, 1]", primals_47: "f32[256]", primals_48: "f32[4, 128, 1, 1]", primals_49: "f32[256, 4, 1, 1]", primals_50: "f32[256]", primals_51: "f32[256]", primals_52: "f32[256, 256, 3, 3]", primals_53: "f32[256]", primals_54: "f32[4, 256, 3, 3]", primals_55: "f32[256, 4, 1, 1]", primals_56: "f32[256]", primals_57: "f32[256]", primals_58: "f32[256, 256, 3, 3]", primals_59: "f32[256]", primals_60: "f32[4, 256, 3, 3]", primals_61: "f32[256, 4, 1, 1]", primals_62: "f32[256, 256, 3, 3]", primals_63: "f32[256]", primals_64: "f32[4, 256, 3, 3]", primals_65: "f32[256, 4, 1, 1]", primals_66: "f32[256]", primals_67: "f32[256]", primals_68: "f32[512, 256, 3, 3]", primals_69: "f32[512]", primals_70: "f32[4, 256, 3, 3]", primals_71: "f32[512, 4, 1, 1]", primals_72: "f32[512]", primals_73: "f32[512]", primals_74: "f32[512, 512, 3, 3]", primals_75: "f32[512]", primals_76: "f32[4, 512, 3, 3]", primals_77: "f32[512, 4, 1, 1]", primals_78: "f32[512, 256, 1, 1]", primals_79: "f32[512]", primals_80: "f32[4, 256, 1, 1]", primals_81: "f32[512, 4, 1, 1]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512, 512, 3, 3]", primals_85: "f32[512]", primals_86: "f32[4, 512, 3, 3]", primals_87: "f32[512, 4, 1, 1]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512, 512, 3, 3]", primals_91: "f32[512]", primals_92: "f32[4, 512, 3, 3]", primals_93: "f32[512, 4, 1, 1]", primals_94: "f32[512, 512, 3, 3]", primals_95: "f32[512]", primals_96: "f32[4, 512, 3, 3]", primals_97: "f32[512, 4, 1, 1]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512, 512, 3, 3]", primals_101: "f32[512]", primals_102: "f32[4, 512, 3, 3]", primals_103: "f32[512, 4, 1, 1]", primals_104: "f32[512]", primals_105: "f32[512]", primals_106: "f32[512, 512, 3, 3]", primals_107: "f32[512]", primals_108: "f32[4, 512, 3, 3]", primals_109: "f32[512, 4, 1, 1]", primals_110: "f32[512]", primals_111: "f32[512]", primals_112: "f32[512, 512, 3, 3]", primals_113: "f32[512]", primals_114: "f32[4, 512, 3, 3]", primals_115: "f32[512, 4, 1, 1]", primals_116: "f32[512]", primals_117: "f32[512]", primals_118: "f32[512, 512, 3, 3]", primals_119: "f32[512]", primals_120: "f32[4, 512, 3, 3]", primals_121: "f32[512, 4, 1, 1]", primals_122: "f32[512]", primals_123: "f32[512]", primals_124: "f32[512, 512, 3, 3]", primals_125: "f32[512]", primals_126: "f32[4, 512, 3, 3]", primals_127: "f32[512, 4, 1, 1]", primals_128: "f32[512]", primals_129: "f32[512]", primals_130: "f32[512, 512, 3, 3]", primals_131: "f32[512]", primals_132: "f32[4, 512, 3, 3]", primals_133: "f32[512, 4, 1, 1]", primals_134: "f32[512]", primals_135: "f32[512]", primals_136: "f32[512, 512]", primals_137: "f32[512]", primals_138: "f32[4, 512]", primals_139: "f32[512, 4]", primals_140: "f32[512, 512]", primals_141: "f32[512]", primals_142: "f32[4, 512]", primals_143: "f32[512, 4]", primals_144: "f32[512, 512]", primals_145: "f32[512]", primals_146: "f32[4, 512]", primals_147: "f32[512, 4]", primals_148: "f32[512, 512]", primals_149: "f32[512]", primals_150: "f32[4, 512]", primals_151: "f32[512, 4]", primals_152: "f32[512]", primals_153: "f32[512]", primals_154: "f32[512, 512, 3, 3]", primals_155: "f32[512]", primals_156: "f32[4, 512, 3, 3]", primals_157: "f32[512, 4, 1, 1]", primals_158: "f32[512]", primals_159: "f32[512]", primals_160: "f32[512, 512, 3, 3]", primals_161: "f32[512]", primals_162: "f32[4, 512, 3, 3]", primals_163: "f32[512, 4, 1, 1]", primals_164: "f32[512]", primals_165: "f32[512]", primals_166: "f32[8, 512, 3, 3]", primals_167: "f32[8]", primals_168: "f32[4, 512, 3, 3]", primals_169: "f32[8, 4, 1, 1]", primals_170: "f32[8, 8, 1, 1]", primals_171: "f32[8]", primals_172: "i64[1]", primals_173: "f32[1280, 320]", primals_174: "f32[1280]", primals_175: "f32[1280, 1280]", primals_176: "f32[1280]", primals_177: "f32[4, 77, 1024]", primals_178: "f32[320, 4, 3, 3]", primals_179: "f32[320]", primals_180: "f32[80, 4, 3, 3]", primals_181: "f32[320, 80, 1, 1]", primals_182: "f32[320]", primals_183: "f32[320]", primals_184: "f32[320, 320, 3, 3]", primals_185: "f32[320]", primals_186: "f32[80, 320, 3, 3]", primals_187: "f32[320, 80, 1, 1]", primals_188: "f32[320, 1280]", primals_189: "f32[320]", primals_190: "f32[320]", primals_191: "f32[320]", primals_192: "f32[320, 320, 3, 3]", primals_193: "f32[320]", primals_194: "f32[80, 320, 3, 3]", primals_195: "f32[320, 80, 1, 1]", primals_196: "f32[320]", primals_197: "f32[320]", primals_198: "f32[320, 320]", primals_199: "f32[320]", primals_200: "f32[80, 320]", primals_201: "f32[320, 80]", primals_202: "f32[320]", primals_203: "f32[320]", primals_204: "f32[320, 320]", primals_205: "f32[80, 320]", primals_206: "f32[320, 80]", primals_207: "f32[320, 320]", primals_208: "f32[80, 320]", primals_209: "f32[320, 80]", primals_210: "f32[320, 320]", primals_211: "f32[80, 320]", primals_212: "f32[320, 80]", primals_213: "f32[320, 320]", primals_214: "f32[320]", primals_215: "f32[80, 320]", primals_216: "f32[320, 80]", primals_217: "f32[320]", primals_218: "f32[320]", primals_219: "f32[320, 320]", primals_220: "f32[80, 320]", primals_221: "f32[320, 80]", primals_222: "f32[320, 1024]", primals_223: "f32[80, 1024]", primals_224: "f32[320, 80]", primals_225: "f32[320, 1024]", primals_226: "f32[80, 1024]", primals_227: "f32[320, 80]", primals_228: "f32[320, 320]", primals_229: "f32[320]", primals_230: "f32[80, 320]", primals_231: "f32[320, 80]", primals_232: "f32[320]", primals_233: "f32[320]", primals_234: "f32[2560, 320]", primals_235: "f32[2560]", primals_236: "f32[80, 320]", primals_237: "f32[2560, 80]", primals_238: "f32[320, 1280]", primals_239: "f32[320]", primals_240: "f32[80, 1280]", primals_241: "f32[320, 80]", primals_242: "f32[320, 320]", primals_243: "f32[320]", primals_244: "f32[80, 320]", primals_245: "f32[320, 80]", primals_246: "f32[320]", primals_247: "f32[320]", primals_248: "f32[320, 320, 3, 3]", primals_249: "f32[320]", primals_250: "f32[80, 320, 3, 3]", primals_251: "f32[320, 80, 1, 1]", primals_252: "f32[320, 1280]", primals_253: "f32[320]", primals_254: "f32[320]", primals_255: "f32[320]", primals_256: "f32[320, 320, 3, 3]", primals_257: "f32[320]", primals_258: "f32[80, 320, 3, 3]", primals_259: "f32[320, 80, 1, 1]", primals_260: "f32[320]", primals_261: "f32[320]", primals_262: "f32[320, 320]", primals_263: "f32[320]", primals_264: "f32[80, 320]", primals_265: "f32[320, 80]", primals_266: "f32[320]", primals_267: "f32[320]", primals_268: "f32[320, 320]", primals_269: "f32[80, 320]", primals_270: "f32[320, 80]", primals_271: "f32[320, 320]", primals_272: "f32[80, 320]", primals_273: "f32[320, 80]", primals_274: "f32[320, 320]", primals_275: "f32[80, 320]", primals_276: "f32[320, 80]", primals_277: "f32[320, 320]", primals_278: "f32[320]", primals_279: "f32[80, 320]", primals_280: "f32[320, 80]", primals_281: "f32[320]", primals_282: "f32[320]", primals_283: "f32[320, 320]", primals_284: "f32[80, 320]", primals_285: "f32[320, 80]", primals_286: "f32[320, 1024]", primals_287: "f32[80, 1024]", primals_288: "f32[320, 80]", primals_289: "f32[320, 1024]", primals_290: "f32[80, 1024]", primals_291: "f32[320, 80]", primals_292: "f32[320, 320]", primals_293: "f32[320]", primals_294: "f32[80, 320]", primals_295: "f32[320, 80]", primals_296: "f32[320]", primals_297: "f32[320]", primals_298: "f32[2560, 320]", primals_299: "f32[2560]", primals_300: "f32[80, 320]", primals_301: "f32[2560, 80]", primals_302: "f32[320, 1280]", primals_303: "f32[320]", primals_304: "f32[80, 1280]", primals_305: "f32[320, 80]", primals_306: "f32[320, 320]", primals_307: "f32[320]", primals_308: "f32[80, 320]", primals_309: "f32[320, 80]", primals_310: "f32[320, 320, 3, 3]", primals_311: "f32[320]", primals_312: "f32[80, 320, 3, 3]", primals_313: "f32[320, 80, 1, 1]", primals_314: "f32[320]", primals_315: "f32[320]", primals_316: "f32[640, 320, 3, 3]", primals_317: "f32[640]", primals_318: "f32[80, 320, 3, 3]", primals_319: "f32[640, 80, 1, 1]", primals_320: "f32[640, 1280]", primals_321: "f32[640]", primals_322: "f32[640]", primals_323: "f32[640]", primals_324: "f32[640, 640, 3, 3]", primals_325: "f32[640]", primals_326: "f32[80, 640, 3, 3]", primals_327: "f32[640, 80, 1, 1]", primals_328: "f32[640, 320, 1, 1]", primals_329: "f32[640]", primals_330: "f32[80, 320, 1, 1]", primals_331: "f32[640, 80, 1, 1]", primals_332: "f32[640]", primals_333: "f32[640]", primals_334: "f32[640, 640]", primals_335: "f32[640]", primals_336: "f32[80, 640]", primals_337: "f32[640, 80]", primals_338: "f32[640]", primals_339: "f32[640]", primals_340: "f32[640, 640]", primals_341: "f32[80, 640]", primals_342: "f32[640, 80]", primals_343: "f32[640, 640]", primals_344: "f32[80, 640]", primals_345: "f32[640, 80]", primals_346: "f32[640, 640]", primals_347: "f32[80, 640]", primals_348: "f32[640, 80]", primals_349: "f32[640, 640]", primals_350: "f32[640]", primals_351: "f32[80, 640]", primals_352: "f32[640, 80]", primals_353: "f32[640]", primals_354: "f32[640]", primals_355: "f32[640, 640]", primals_356: "f32[80, 640]", primals_357: "f32[640, 80]", primals_358: "f32[640, 1024]", primals_359: "f32[80, 1024]", primals_360: "f32[640, 80]", primals_361: "f32[640, 1024]", primals_362: "f32[80, 1024]", primals_363: "f32[640, 80]", primals_364: "f32[640, 640]", primals_365: "f32[640]", primals_366: "f32[80, 640]", primals_367: "f32[640, 80]", primals_368: "f32[640]", primals_369: "f32[640]", primals_370: "f32[5120, 640]", primals_371: "f32[5120]", primals_372: "f32[80, 640]", primals_373: "f32[5120, 80]", primals_374: "f32[640, 2560]", primals_375: "f32[640]", primals_376: "f32[80, 2560]", primals_377: "f32[640, 80]", primals_378: "f32[640, 640]", primals_379: "f32[640]", primals_380: "f32[80, 640]", primals_381: "f32[640, 80]", primals_382: "f32[640]", primals_383: "f32[640]", primals_384: "f32[640, 640, 3, 3]", primals_385: "f32[640]", primals_386: "f32[80, 640, 3, 3]", primals_387: "f32[640, 80, 1, 1]", primals_388: "f32[640, 1280]", primals_389: "f32[640]", primals_390: "f32[640]", primals_391: "f32[640]", primals_392: "f32[640, 640, 3, 3]", primals_393: "f32[640]", primals_394: "f32[80, 640, 3, 3]", primals_395: "f32[640, 80, 1, 1]", primals_396: "f32[640]", primals_397: "f32[640]", primals_398: "f32[640, 640]", primals_399: "f32[640]", primals_400: "f32[80, 640]", primals_401: "f32[640, 80]", primals_402: "f32[640]", primals_403: "f32[640]", primals_404: "f32[640, 640]", primals_405: "f32[80, 640]", primals_406: "f32[640, 80]", primals_407: "f32[640, 640]", primals_408: "f32[80, 640]", primals_409: "f32[640, 80]", primals_410: "f32[640, 640]", primals_411: "f32[80, 640]", primals_412: "f32[640, 80]", primals_413: "f32[640, 640]", primals_414: "f32[640]", primals_415: "f32[80, 640]", primals_416: "f32[640, 80]", primals_417: "f32[640]", primals_418: "f32[640]", primals_419: "f32[640, 640]", primals_420: "f32[80, 640]", primals_421: "f32[640, 80]", primals_422: "f32[640, 1024]", primals_423: "f32[80, 1024]", primals_424: "f32[640, 80]", primals_425: "f32[640, 1024]", primals_426: "f32[80, 1024]", primals_427: "f32[640, 80]", primals_428: "f32[640, 640]", primals_429: "f32[640]", primals_430: "f32[80, 640]", primals_431: "f32[640, 80]", primals_432: "f32[640]", primals_433: "f32[640]", primals_434: "f32[5120, 640]", primals_435: "f32[5120]", primals_436: "f32[80, 640]", primals_437: "f32[5120, 80]", primals_438: "f32[640, 2560]", primals_439: "f32[640]", primals_440: "f32[80, 2560]", primals_441: "f32[640, 80]", primals_442: "f32[640, 640]", primals_443: "f32[640]", primals_444: "f32[80, 640]", primals_445: "f32[640, 80]", primals_446: "f32[640, 640, 3, 3]", primals_447: "f32[640]", primals_448: "f32[80, 640, 3, 3]", primals_449: "f32[640, 80, 1, 1]", primals_450: "f32[640]", primals_451: "f32[640]", primals_452: "f32[1280, 640, 3, 3]", primals_453: "f32[1280]", primals_454: "f32[80, 640, 3, 3]", primals_455: "f32[1280, 80, 1, 1]", primals_456: "f32[1280, 1280]", primals_457: "f32[1280]", primals_458: "f32[1280]", primals_459: "f32[1280]", primals_460: "f32[1280, 1280, 3, 3]", primals_461: "f32[1280]", primals_462: "f32[80, 1280, 3, 3]", primals_463: "f32[1280, 80, 1, 1]", primals_464: "f32[1280, 640, 1, 1]", primals_465: "f32[1280]", primals_466: "f32[80, 640, 1, 1]", primals_467: "f32[1280, 80, 1, 1]", primals_468: "f32[1280]", primals_469: "f32[1280]", primals_470: "f32[1280, 1280]", primals_471: "f32[1280]", primals_472: "f32[80, 1280]", primals_473: "f32[1280, 80]", primals_474: "f32[1280]", primals_475: "f32[1280]", primals_476: "f32[1280, 1280]", primals_477: "f32[80, 1280]", primals_478: "f32[1280, 80]", primals_479: "f32[1280, 1280]", primals_480: "f32[80, 1280]", primals_481: "f32[1280, 80]", primals_482: "f32[1280, 1280]", primals_483: "f32[80, 1280]", primals_484: "f32[1280, 80]", primals_485: "f32[1280, 1280]", primals_486: "f32[1280]", primals_487: "f32[80, 1280]", primals_488: "f32[1280, 80]", primals_489: "f32[1280]", primals_490: "f32[1280]", primals_491: "f32[1280, 1280]", primals_492: "f32[80, 1280]", primals_493: "f32[1280, 80]", primals_494: "f32[1280, 1024]", primals_495: "f32[80, 1024]", primals_496: "f32[1280, 80]", primals_497: "f32[1280, 1024]", primals_498: "f32[80, 1024]", primals_499: "f32[1280, 80]", primals_500: "f32[1280, 1280]", primals_501: "f32[1280]", primals_502: "f32[80, 1280]", primals_503: "f32[1280, 80]", primals_504: "f32[1280]", primals_505: "f32[1280]", primals_506: "f32[10240, 1280]", primals_507: "f32[10240]", primals_508: "f32[80, 1280]", primals_509: "f32[10240, 80]", primals_510: "f32[1280, 5120]", primals_511: "f32[1280]", primals_512: "f32[80, 5120]", primals_513: "f32[1280, 80]", primals_514: "f32[1280, 1280]", primals_515: "f32[1280]", primals_516: "f32[80, 1280]", primals_517: "f32[1280, 80]", primals_518: "f32[1280]", primals_519: "f32[1280]", primals_520: "f32[1280, 1280, 3, 3]", primals_521: "f32[1280]", primals_522: "f32[80, 1280, 3, 3]", primals_523: "f32[1280, 80, 1, 1]", primals_524: "f32[1280, 1280]", primals_525: "f32[1280]", primals_526: "f32[1280]", primals_527: "f32[1280]", primals_528: "f32[1280, 1280, 3, 3]", primals_529: "f32[1280]", primals_530: "f32[80, 1280, 3, 3]", primals_531: "f32[1280, 80, 1, 1]", primals_532: "f32[1280]", primals_533: "f32[1280]", primals_534: "f32[1280, 1280]", primals_535: "f32[1280]", primals_536: "f32[80, 1280]", primals_537: "f32[1280, 80]", primals_538: "f32[1280]", primals_539: "f32[1280]", primals_540: "f32[1280, 1280]", primals_541: "f32[80, 1280]", primals_542: "f32[1280, 80]", primals_543: "f32[1280, 1280]", primals_544: "f32[80, 1280]", primals_545: "f32[1280, 80]", primals_546: "f32[1280, 1280]", primals_547: "f32[80, 1280]", primals_548: "f32[1280, 80]", primals_549: "f32[1280, 1280]", primals_550: "f32[1280]", primals_551: "f32[80, 1280]", primals_552: "f32[1280, 80]", primals_553: "f32[1280]", primals_554: "f32[1280]", primals_555: "f32[1280, 1280]", primals_556: "f32[80, 1280]", primals_557: "f32[1280, 80]", primals_558: "f32[1280, 1024]", primals_559: "f32[80, 1024]", primals_560: "f32[1280, 80]", primals_561: "f32[1280, 1024]", primals_562: "f32[80, 1024]", primals_563: "f32[1280, 80]", primals_564: "f32[1280, 1280]", primals_565: "f32[1280]", primals_566: "f32[80, 1280]", primals_567: "f32[1280, 80]", primals_568: "f32[1280]", primals_569: "f32[1280]", primals_570: "f32[10240, 1280]", primals_571: "f32[10240]", primals_572: "f32[80, 1280]", primals_573: "f32[10240, 80]", primals_574: "f32[1280, 5120]", primals_575: "f32[1280]", primals_576: "f32[80, 5120]", primals_577: "f32[1280, 80]", primals_578: "f32[1280, 1280]", primals_579: "f32[1280]", primals_580: "f32[80, 1280]", primals_581: "f32[1280, 80]", primals_582: "f32[1280, 1280, 3, 3]", primals_583: "f32[1280]", primals_584: "f32[80, 1280, 3, 3]", primals_585: "f32[1280, 80, 1, 1]", primals_586: "f32[1280]", primals_587: "f32[1280]", primals_588: "f32[1280, 1280, 3, 3]", primals_589: "f32[1280]", primals_590: "f32[80, 1280, 3, 3]", primals_591: "f32[1280, 80, 1, 1]", primals_592: "f32[1280, 1280]", primals_593: "f32[1280]", primals_594: "f32[1280]", primals_595: "f32[1280]", primals_596: "f32[1280, 1280, 3, 3]", primals_597: "f32[1280]", primals_598: "f32[80, 1280, 3, 3]", primals_599: "f32[1280, 80, 1, 1]", primals_600: "f32[1280]", primals_601: "f32[1280]", primals_602: "f32[1280, 1280, 3, 3]", primals_603: "f32[1280]", primals_604: "f32[80, 1280, 3, 3]", primals_605: "f32[1280, 80, 1, 1]", primals_606: "f32[1280, 1280]", primals_607: "f32[1280]", primals_608: "f32[1280]", primals_609: "f32[1280]", primals_610: "f32[1280, 1280, 3, 3]", primals_611: "f32[1280]", primals_612: "f32[80, 1280, 3, 3]", primals_613: "f32[1280, 80, 1, 1]", primals_614: "f32[1280]", primals_615: "f32[1280]", primals_616: "f32[1280, 1280, 3, 3]", primals_617: "f32[1280]", primals_618: "f32[80, 1280, 3, 3]", primals_619: "f32[1280, 80, 1, 1]", primals_620: "f32[1280, 1280]", primals_621: "f32[1280]", primals_622: "f32[1280]", primals_623: "f32[1280]", primals_624: "f32[1280, 1280, 3, 3]", primals_625: "f32[1280]", primals_626: "f32[80, 1280, 3, 3]", primals_627: "f32[1280, 80, 1, 1]", primals_628: "f32[1280]", primals_629: "f32[1280]", primals_630: "f32[1280, 1280]", primals_631: "f32[1280]", primals_632: "f32[80, 1280]", primals_633: "f32[1280, 80]", primals_634: "f32[1280]", primals_635: "f32[1280]", primals_636: "f32[1280, 1280]", primals_637: "f32[80, 1280]", primals_638: "f32[1280, 80]", primals_639: "f32[1280, 1280]", primals_640: "f32[80, 1280]", primals_641: "f32[1280, 80]", primals_642: "f32[1280, 1280]", primals_643: "f32[80, 1280]", primals_644: "f32[1280, 80]", primals_645: "f32[1280, 1280]", primals_646: "f32[1280]", primals_647: "f32[80, 1280]", primals_648: "f32[1280, 80]", primals_649: "f32[1280]", primals_650: "f32[1280]", primals_651: "f32[1280, 1280]", primals_652: "f32[80, 1280]", primals_653: "f32[1280, 80]", primals_654: "f32[1280, 1024]", primals_655: "f32[80, 1024]", primals_656: "f32[1280, 80]", primals_657: "f32[1280, 1024]", primals_658: "f32[80, 1024]", primals_659: "f32[1280, 80]", primals_660: "f32[1280, 1280]", primals_661: "f32[1280]", primals_662: "f32[80, 1280]", primals_663: "f32[1280, 80]", primals_664: "f32[1280]", primals_665: "f32[1280]", primals_666: "f32[10240, 1280]", primals_667: "f32[10240]", primals_668: "f32[80, 1280]", primals_669: "f32[10240, 80]", primals_670: "f32[1280, 5120]", primals_671: "f32[1280]", primals_672: "f32[80, 5120]", primals_673: "f32[1280, 80]", primals_674: "f32[1280, 1280]", primals_675: "f32[1280]", primals_676: "f32[80, 1280]", primals_677: "f32[1280, 80]", primals_678: "f32[1280]", primals_679: "f32[1280]", primals_680: "f32[1280, 1280, 3, 3]", primals_681: "f32[1280]", primals_682: "f32[80, 1280, 3, 3]", primals_683: "f32[1280, 80, 1, 1]", primals_684: "f32[1280, 1280]", primals_685: "f32[1280]", primals_686: "f32[1280]", primals_687: "f32[1280]", primals_688: "f32[1280, 1280, 3, 3]", primals_689: "f32[1280]", primals_690: "f32[80, 1280, 3, 3]", primals_691: "f32[1280, 80, 1, 1]", primals_692: "f32[2560]", primals_693: "f32[2560]", primals_694: "f32[1280, 2560, 3, 3]", primals_695: "f32[1280]", primals_696: "f32[80, 2560, 3, 3]", primals_697: "f32[1280, 80, 1, 1]", primals_698: "f32[1280, 1280]", primals_699: "f32[1280]", primals_700: "f32[1280]", primals_701: "f32[1280]", primals_702: "f32[1280, 1280, 3, 3]", primals_703: "f32[1280]", primals_704: "f32[80, 1280, 3, 3]", primals_705: "f32[1280, 80, 1, 1]", primals_706: "f32[1280, 2560, 1, 1]", primals_707: "f32[1280]", primals_708: "f32[80, 2560, 1, 1]", primals_709: "f32[1280, 80, 1, 1]", primals_710: "f32[2560]", primals_711: "f32[2560]", primals_712: "f32[1280, 2560, 3, 3]", primals_713: "f32[1280]", primals_714: "f32[80, 2560, 3, 3]", primals_715: "f32[1280, 80, 1, 1]", primals_716: "f32[1280, 1280]", primals_717: "f32[1280]", primals_718: "f32[1280]", primals_719: "f32[1280]", primals_720: "f32[1280, 1280, 3, 3]", primals_721: "f32[1280]", primals_722: "f32[80, 1280, 3, 3]", primals_723: "f32[1280, 80, 1, 1]", primals_724: "f32[1280, 2560, 1, 1]", primals_725: "f32[1280]", primals_726: "f32[80, 2560, 1, 1]", primals_727: "f32[1280, 80, 1, 1]", primals_728: "f32[2560]", primals_729: "f32[2560]", primals_730: "f32[1280, 2560, 3, 3]", primals_731: "f32[1280]", primals_732: "f32[80, 2560, 3, 3]", primals_733: "f32[1280, 80, 1, 1]", primals_734: "f32[1280, 1280]", primals_735: "f32[1280]", primals_736: "f32[1280]", primals_737: "f32[1280]", primals_738: "f32[1280, 1280, 3, 3]", primals_739: "f32[1280]", primals_740: "f32[80, 1280, 3, 3]", primals_741: "f32[1280, 80, 1, 1]", primals_742: "f32[1280, 2560, 1, 1]", primals_743: "f32[1280]", primals_744: "f32[80, 2560, 1, 1]", primals_745: "f32[1280, 80, 1, 1]", primals_746: "f32[1280, 1280, 3, 3]", primals_747: "f32[1280]", primals_748: "f32[80, 1280, 3, 3]", primals_749: "f32[1280, 80, 1, 1]", primals_750: "f32[2560]", primals_751: "f32[2560]", primals_752: "f32[1280, 2560, 3, 3]", primals_753: "f32[1280]", primals_754: "f32[80, 2560, 3, 3]", primals_755: "f32[1280, 80, 1, 1]", primals_756: "f32[1280, 1280]", primals_757: "f32[1280]", primals_758: "f32[1280]", primals_759: "f32[1280]", primals_760: "f32[1280, 1280, 3, 3]", primals_761: "f32[1280]", primals_762: "f32[80, 1280, 3, 3]", primals_763: "f32[1280, 80, 1, 1]", primals_764: "f32[1280, 2560, 1, 1]", primals_765: "f32[1280]", primals_766: "f32[80, 2560, 1, 1]", primals_767: "f32[1280, 80, 1, 1]", primals_768: "f32[1280]", primals_769: "f32[1280]", primals_770: "f32[1280, 1280]", primals_771: "f32[1280]", primals_772: "f32[80, 1280]", primals_773: "f32[1280, 80]", primals_774: "f32[1280]", primals_775: "f32[1280]", primals_776: "f32[1280, 1280]", primals_777: "f32[80, 1280]", primals_778: "f32[1280, 80]", primals_779: "f32[1280, 1280]", primals_780: "f32[80, 1280]", primals_781: "f32[1280, 80]", primals_782: "f32[1280, 1280]", primals_783: "f32[80, 1280]", primals_784: "f32[1280, 80]", primals_785: "f32[1280, 1280]", primals_786: "f32[1280]", primals_787: "f32[80, 1280]", primals_788: "f32[1280, 80]", primals_789: "f32[1280]", primals_790: "f32[1280]", primals_791: "f32[1280, 1280]", primals_792: "f32[80, 1280]", primals_793: "f32[1280, 80]", primals_794: "f32[1280, 1024]", primals_795: "f32[80, 1024]", primals_796: "f32[1280, 80]", primals_797: "f32[1280, 1024]", primals_798: "f32[80, 1024]", primals_799: "f32[1280, 80]", primals_800: "f32[1280, 1280]", primals_801: "f32[1280]", primals_802: "f32[80, 1280]", primals_803: "f32[1280, 80]", primals_804: "f32[1280]", primals_805: "f32[1280]", primals_806: "f32[10240, 1280]", primals_807: "f32[10240]", primals_808: "f32[80, 1280]", primals_809: "f32[10240, 80]", primals_810: "f32[1280, 5120]", primals_811: "f32[1280]", primals_812: "f32[80, 5120]", primals_813: "f32[1280, 80]", primals_814: "f32[1280, 1280]", primals_815: "f32[1280]", primals_816: "f32[80, 1280]", primals_817: "f32[1280, 80]", primals_818: "f32[2560]", primals_819: "f32[2560]", primals_820: "f32[1280, 2560, 3, 3]", primals_821: "f32[1280]", primals_822: "f32[80, 2560, 3, 3]", primals_823: "f32[1280, 80, 1, 1]", primals_824: "f32[1280, 1280]", primals_825: "f32[1280]", primals_826: "f32[1280]", primals_827: "f32[1280]", primals_828: "f32[1280, 1280, 3, 3]", primals_829: "f32[1280]", primals_830: "f32[80, 1280, 3, 3]", primals_831: "f32[1280, 80, 1, 1]", primals_832: "f32[1280, 2560, 1, 1]", primals_833: "f32[1280]", primals_834: "f32[80, 2560, 1, 1]", primals_835: "f32[1280, 80, 1, 1]", primals_836: "f32[1280]", primals_837: "f32[1280]", primals_838: "f32[1280, 1280]", primals_839: "f32[1280]", primals_840: "f32[80, 1280]", primals_841: "f32[1280, 80]", primals_842: "f32[1280]", primals_843: "f32[1280]", primals_844: "f32[1280, 1280]", primals_845: "f32[80, 1280]", primals_846: "f32[1280, 80]", primals_847: "f32[1280, 1280]", primals_848: "f32[80, 1280]", primals_849: "f32[1280, 80]", primals_850: "f32[1280, 1280]", primals_851: "f32[80, 1280]", primals_852: "f32[1280, 80]", primals_853: "f32[1280, 1280]", primals_854: "f32[1280]", primals_855: "f32[80, 1280]", primals_856: "f32[1280, 80]", primals_857: "f32[1280]", primals_858: "f32[1280]", primals_859: "f32[1280, 1280]", primals_860: "f32[80, 1280]", primals_861: "f32[1280, 80]", primals_862: "f32[1280, 1024]", primals_863: "f32[80, 1024]", primals_864: "f32[1280, 80]", primals_865: "f32[1280, 1024]", primals_866: "f32[80, 1024]", primals_867: "f32[1280, 80]", primals_868: "f32[1280, 1280]", primals_869: "f32[1280]", primals_870: "f32[80, 1280]", primals_871: "f32[1280, 80]", primals_872: "f32[1280]", primals_873: "f32[1280]", primals_874: "f32[10240, 1280]", primals_875: "f32[10240]", primals_876: "f32[80, 1280]", primals_877: "f32[10240, 80]", primals_878: "f32[1280, 5120]", primals_879: "f32[1280]", primals_880: "f32[80, 5120]", primals_881: "f32[1280, 80]", primals_882: "f32[1280, 1280]", primals_883: "f32[1280]", primals_884: "f32[80, 1280]", primals_885: "f32[1280, 80]", primals_886: "f32[1920]", primals_887: "f32[1920]", primals_888: "f32[1280, 1920, 3, 3]", primals_889: "f32[1280]", primals_890: "f32[80, 1920, 3, 3]", primals_891: "f32[1280, 80, 1, 1]", primals_892: "f32[1280, 1280]", primals_893: "f32[1280]", primals_894: "f32[1280]", primals_895: "f32[1280]", primals_896: "f32[1280, 1280, 3, 3]", primals_897: "f32[1280]", primals_898: "f32[80, 1280, 3, 3]", primals_899: "f32[1280, 80, 1, 1]", primals_900: "f32[1280, 1920, 1, 1]", primals_901: "f32[1280]", primals_902: "f32[80, 1920, 1, 1]", primals_903: "f32[1280, 80, 1, 1]", primals_904: "f32[1280]", primals_905: "f32[1280]", primals_906: "f32[1280, 1280]", primals_907: "f32[1280]", primals_908: "f32[80, 1280]", primals_909: "f32[1280, 80]", primals_910: "f32[1280]", primals_911: "f32[1280]", primals_912: "f32[1280, 1280]", primals_913: "f32[80, 1280]", primals_914: "f32[1280, 80]", primals_915: "f32[1280, 1280]", primals_916: "f32[80, 1280]", primals_917: "f32[1280, 80]", primals_918: "f32[1280, 1280]", primals_919: "f32[80, 1280]", primals_920: "f32[1280, 80]", primals_921: "f32[1280, 1280]", primals_922: "f32[1280]", primals_923: "f32[80, 1280]", primals_924: "f32[1280, 80]", primals_925: "f32[1280]", primals_926: "f32[1280]", primals_927: "f32[1280, 1280]", primals_928: "f32[80, 1280]", primals_929: "f32[1280, 80]", primals_930: "f32[1280, 1024]", primals_931: "f32[80, 1024]", primals_932: "f32[1280, 80]", primals_933: "f32[1280, 1024]", primals_934: "f32[80, 1024]", primals_935: "f32[1280, 80]", primals_936: "f32[1280, 1280]", primals_937: "f32[1280]", primals_938: "f32[80, 1280]", primals_939: "f32[1280, 80]", primals_940: "f32[1280]", primals_941: "f32[1280]", primals_942: "f32[10240, 1280]", primals_943: "f32[10240]", primals_944: "f32[80, 1280]", primals_945: "f32[10240, 80]", primals_946: "f32[1280, 5120]", primals_947: "f32[1280]", primals_948: "f32[80, 5120]", primals_949: "f32[1280, 80]", primals_950: "f32[1280, 1280]", primals_951: "f32[1280]", primals_952: "f32[80, 1280]", primals_953: "f32[1280, 80]", primals_954: "f32[1280, 1280, 3, 3]", primals_955: "f32[1280]", primals_956: "f32[80, 1280, 3, 3]", primals_957: "f32[1280, 80, 1, 1]", primals_958: "f32[1920]", primals_959: "f32[1920]", primals_960: "f32[640, 1920, 3, 3]", primals_961: "f32[640]", primals_962: "f32[80, 1920, 3, 3]", primals_963: "f32[640, 80, 1, 1]", primals_964: "f32[640, 1280]", primals_965: "f32[640]", primals_966: "f32[640]", primals_967: "f32[640]", primals_968: "f32[640, 640, 3, 3]", primals_969: "f32[640]", primals_970: "f32[80, 640, 3, 3]", primals_971: "f32[640, 80, 1, 1]", primals_972: "f32[640, 1920, 1, 1]", primals_973: "f32[640]", primals_974: "f32[80, 1920, 1, 1]", primals_975: "f32[640, 80, 1, 1]", primals_976: "f32[640]", primals_977: "f32[640]", primals_978: "f32[640, 640]", primals_979: "f32[640]", primals_980: "f32[80, 640]", primals_981: "f32[640, 80]", primals_982: "f32[640]", primals_983: "f32[640]", primals_984: "f32[640, 640]", primals_985: "f32[80, 640]", primals_986: "f32[640, 80]", primals_987: "f32[640, 640]", primals_988: "f32[80, 640]", primals_989: "f32[640, 80]", primals_990: "f32[640, 640]", primals_991: "f32[80, 640]", primals_992: "f32[640, 80]", primals_993: "f32[640, 640]", primals_994: "f32[640]", primals_995: "f32[80, 640]", primals_996: "f32[640, 80]", primals_997: "f32[640]", primals_998: "f32[640]", primals_999: "f32[640, 640]", primals_1000: "f32[80, 640]", primals_1001: "f32[640, 80]", primals_1002: "f32[640, 1024]", primals_1003: "f32[80, 1024]", primals_1004: "f32[640, 80]", primals_1005: "f32[640, 1024]", primals_1006: "f32[80, 1024]", primals_1007: "f32[640, 80]", primals_1008: "f32[640, 640]", primals_1009: "f32[640]", primals_1010: "f32[80, 640]", primals_1011: "f32[640, 80]", primals_1012: "f32[640]", primals_1013: "f32[640]", primals_1014: "f32[5120, 640]", primals_1015: "f32[5120]", primals_1016: "f32[80, 640]", primals_1017: "f32[5120, 80]", primals_1018: "f32[640, 2560]", primals_1019: "f32[640]", primals_1020: "f32[80, 2560]", primals_1021: "f32[640, 80]", primals_1022: "f32[640, 640]", primals_1023: "f32[640]", primals_1024: "f32[80, 640]", primals_1025: "f32[640, 80]", primals_1026: "f32[1280]", primals_1027: "f32[1280]", primals_1028: "f32[640, 1280, 3, 3]", primals_1029: "f32[640]", primals_1030: "f32[80, 1280, 3, 3]", primals_1031: "f32[640, 80, 1, 1]", primals_1032: "f32[640, 1280]", primals_1033: "f32[640]", primals_1034: "f32[640]", primals_1035: "f32[640]", primals_1036: "f32[640, 640, 3, 3]", primals_1037: "f32[640]", primals_1038: "f32[80, 640, 3, 3]", primals_1039: "f32[640, 80, 1, 1]", primals_1040: "f32[640, 1280, 1, 1]", primals_1041: "f32[640]", primals_1042: "f32[80, 1280, 1, 1]", primals_1043: "f32[640, 80, 1, 1]", primals_1044: "f32[640]", primals_1045: "f32[640]", primals_1046: "f32[640, 640]", primals_1047: "f32[640]", primals_1048: "f32[80, 640]", primals_1049: "f32[640, 80]", primals_1050: "f32[640]", primals_1051: "f32[640]", primals_1052: "f32[640, 640]", primals_1053: "f32[80, 640]", primals_1054: "f32[640, 80]", primals_1055: "f32[640, 640]", primals_1056: "f32[80, 640]", primals_1057: "f32[640, 80]", primals_1058: "f32[640, 640]", primals_1059: "f32[80, 640]", primals_1060: "f32[640, 80]", primals_1061: "f32[640, 640]", primals_1062: "f32[640]", primals_1063: "f32[80, 640]", primals_1064: "f32[640, 80]", primals_1065: "f32[640]", primals_1066: "f32[640]", primals_1067: "f32[640, 640]", primals_1068: "f32[80, 640]", primals_1069: "f32[640, 80]", primals_1070: "f32[640, 1024]", primals_1071: "f32[80, 1024]", primals_1072: "f32[640, 80]", primals_1073: "f32[640, 1024]", primals_1074: "f32[80, 1024]", primals_1075: "f32[640, 80]", primals_1076: "f32[640, 640]", primals_1077: "f32[640]", primals_1078: "f32[80, 640]", primals_1079: "f32[640, 80]", primals_1080: "f32[640]", primals_1081: "f32[640]", primals_1082: "f32[5120, 640]", primals_1083: "f32[5120]", primals_1084: "f32[80, 640]", primals_1085: "f32[5120, 80]", primals_1086: "f32[640, 2560]", primals_1087: "f32[640]", primals_1088: "f32[80, 2560]", primals_1089: "f32[640, 80]", primals_1090: "f32[640, 640]", primals_1091: "f32[640]", primals_1092: "f32[80, 640]", primals_1093: "f32[640, 80]", primals_1094: "f32[960]", primals_1095: "f32[960]", primals_1096: "f32[640, 960, 3, 3]", primals_1097: "f32[640]", primals_1098: "f32[80, 960, 3, 3]", primals_1099: "f32[640, 80, 1, 1]", primals_1100: "f32[640, 1280]", primals_1101: "f32[640]", primals_1102: "f32[640]", primals_1103: "f32[640]", primals_1104: "f32[640, 640, 3, 3]", primals_1105: "f32[640]", primals_1106: "f32[80, 640, 3, 3]", primals_1107: "f32[640, 80, 1, 1]", primals_1108: "f32[640, 960, 1, 1]", primals_1109: "f32[640]", primals_1110: "f32[80, 960, 1, 1]", primals_1111: "f32[640, 80, 1, 1]", primals_1112: "f32[640]", primals_1113: "f32[640]", primals_1114: "f32[640, 640]", primals_1115: "f32[640]", primals_1116: "f32[80, 640]", primals_1117: "f32[640, 80]", primals_1118: "f32[640]", primals_1119: "f32[640]", primals_1120: "f32[640, 640]", primals_1121: "f32[80, 640]", primals_1122: "f32[640, 80]", primals_1123: "f32[640, 640]", primals_1124: "f32[80, 640]", primals_1125: "f32[640, 80]", primals_1126: "f32[640, 640]", primals_1127: "f32[80, 640]", primals_1128: "f32[640, 80]", primals_1129: "f32[640, 640]", primals_1130: "f32[640]", primals_1131: "f32[80, 640]", primals_1132: "f32[640, 80]", primals_1133: "f32[640]", primals_1134: "f32[640]", primals_1135: "f32[640, 640]", primals_1136: "f32[80, 640]", primals_1137: "f32[640, 80]", primals_1138: "f32[640, 1024]", primals_1139: "f32[80, 1024]", primals_1140: "f32[640, 80]", primals_1141: "f32[640, 1024]", primals_1142: "f32[80, 1024]", primals_1143: "f32[640, 80]", primals_1144: "f32[640, 640]", primals_1145: "f32[640]", primals_1146: "f32[80, 640]", primals_1147: "f32[640, 80]", primals_1148: "f32[640]", primals_1149: "f32[640]", primals_1150: "f32[5120, 640]", primals_1151: "f32[5120]", primals_1152: "f32[80, 640]", primals_1153: "f32[5120, 80]", primals_1154: "f32[640, 2560]", primals_1155: "f32[640]", primals_1156: "f32[80, 2560]", primals_1157: "f32[640, 80]", primals_1158: "f32[640, 640]", primals_1159: "f32[640]", primals_1160: "f32[80, 640]", primals_1161: "f32[640, 80]", primals_1162: "f32[640, 640, 3, 3]", primals_1163: "f32[640]", primals_1164: "f32[80, 640, 3, 3]", primals_1165: "f32[640, 80, 1, 1]", primals_1166: "f32[960]", primals_1167: "f32[960]", primals_1168: "f32[320, 960, 3, 3]", primals_1169: "f32[320]", primals_1170: "f32[80, 960, 3, 3]", primals_1171: "f32[320, 80, 1, 1]", primals_1172: "f32[320, 1280]", primals_1173: "f32[320]", primals_1174: "f32[320]", primals_1175: "f32[320]", primals_1176: "f32[320, 320, 3, 3]", primals_1177: "f32[320]", primals_1178: "f32[80, 320, 3, 3]", primals_1179: "f32[320, 80, 1, 1]", primals_1180: "f32[320, 960, 1, 1]", primals_1181: "f32[320]", primals_1182: "f32[80, 960, 1, 1]", primals_1183: "f32[320, 80, 1, 1]", primals_1184: "f32[320]", primals_1185: "f32[320]", primals_1186: "f32[320, 320]", primals_1187: "f32[320]", primals_1188: "f32[80, 320]", primals_1189: "f32[320, 80]", primals_1190: "f32[320]", primals_1191: "f32[320]", primals_1192: "f32[320, 320]", primals_1193: "f32[80, 320]", primals_1194: "f32[320, 80]", primals_1195: "f32[320, 320]", primals_1196: "f32[80, 320]", primals_1197: "f32[320, 80]", primals_1198: "f32[320, 320]", primals_1199: "f32[80, 320]", primals_1200: "f32[320, 80]", primals_1201: "f32[320, 320]", primals_1202: "f32[320]", primals_1203: "f32[80, 320]", primals_1204: "f32[320, 80]", primals_1205: "f32[320]", primals_1206: "f32[320]", primals_1207: "f32[320, 320]", primals_1208: "f32[80, 320]", primals_1209: "f32[320, 80]", primals_1210: "f32[320, 1024]", primals_1211: "f32[80, 1024]", primals_1212: "f32[320, 80]", primals_1213: "f32[320, 1024]", primals_1214: "f32[80, 1024]", primals_1215: "f32[320, 80]", primals_1216: "f32[320, 320]", primals_1217: "f32[320]", primals_1218: "f32[80, 320]", primals_1219: "f32[320, 80]", primals_1220: "f32[320]", primals_1221: "f32[320]", primals_1222: "f32[2560, 320]", primals_1223: "f32[2560]", primals_1224: "f32[80, 320]", primals_1225: "f32[2560, 80]", primals_1226: "f32[320, 1280]", primals_1227: "f32[320]", primals_1228: "f32[80, 1280]", primals_1229: "f32[320, 80]", primals_1230: "f32[320, 320]", primals_1231: "f32[320]", primals_1232: "f32[80, 320]", primals_1233: "f32[320, 80]", primals_1234: "f32[640]", primals_1235: "f32[640]", primals_1236: "f32[320, 640, 3, 3]", primals_1237: "f32[320]", primals_1238: "f32[80, 640, 3, 3]", primals_1239: "f32[320, 80, 1, 1]", primals_1240: "f32[320, 1280]", primals_1241: "f32[320]", primals_1242: "f32[320]", primals_1243: "f32[320]", primals_1244: "f32[320, 320, 3, 3]", primals_1245: "f32[320]", primals_1246: "f32[80, 320, 3, 3]", primals_1247: "f32[320, 80, 1, 1]", primals_1248: "f32[320, 640, 1, 1]", primals_1249: "f32[320]", primals_1250: "f32[80, 640, 1, 1]", primals_1251: "f32[320, 80, 1, 1]", primals_1252: "f32[320]", primals_1253: "f32[320]", primals_1254: "f32[320, 320]", primals_1255: "f32[320]", primals_1256: "f32[80, 320]", primals_1257: "f32[320, 80]", primals_1258: "f32[320]", primals_1259: "f32[320]", primals_1260: "f32[320, 320]", primals_1261: "f32[80, 320]", primals_1262: "f32[320, 80]", primals_1263: "f32[320, 320]", primals_1264: "f32[80, 320]", primals_1265: "f32[320, 80]", primals_1266: "f32[320, 320]", primals_1267: "f32[80, 320]", primals_1268: "f32[320, 80]", primals_1269: "f32[320, 320]", primals_1270: "f32[320]", primals_1271: "f32[80, 320]", primals_1272: "f32[320, 80]", primals_1273: "f32[320]", primals_1274: "f32[320]", primals_1275: "f32[320, 320]", primals_1276: "f32[80, 320]", primals_1277: "f32[320, 80]", primals_1278: "f32[320, 1024]", primals_1279: "f32[80, 1024]", primals_1280: "f32[320, 80]", primals_1281: "f32[320, 1024]", primals_1282: "f32[80, 1024]", primals_1283: "f32[320, 80]", primals_1284: "f32[320, 320]", primals_1285: "f32[320]", primals_1286: "f32[80, 320]", primals_1287: "f32[320, 80]", primals_1288: "f32[320]", primals_1289: "f32[320]", primals_1290: "f32[2560, 320]", primals_1291: "f32[2560]", primals_1292: "f32[80, 320]", primals_1293: "f32[2560, 80]", primals_1294: "f32[320, 1280]", primals_1295: "f32[320]", primals_1296: "f32[80, 1280]", primals_1297: "f32[320, 80]", primals_1298: "f32[320, 320]", primals_1299: "f32[320]", primals_1300: "f32[80, 320]", primals_1301: "f32[320, 80]", primals_1302: "f32[640]", primals_1303: "f32[640]", primals_1304: "f32[320, 640, 3, 3]", primals_1305: "f32[320]", primals_1306: "f32[80, 640, 3, 3]", primals_1307: "f32[320, 80, 1, 1]", primals_1308: "f32[320, 1280]", primals_1309: "f32[320]", primals_1310: "f32[320]", primals_1311: "f32[320]", primals_1312: "f32[320, 320, 3, 3]", primals_1313: "f32[320]", primals_1314: "f32[80, 320, 3, 3]", primals_1315: "f32[320, 80, 1, 1]", primals_1316: "f32[320, 640, 1, 1]", primals_1317: "f32[320]", primals_1318: "f32[80, 640, 1, 1]", primals_1319: "f32[320, 80, 1, 1]", primals_1320: "f32[320]", primals_1321: "f32[320]", primals_1322: "f32[320, 320]", primals_1323: "f32[320]", primals_1324: "f32[80, 320]", primals_1325: "f32[320, 80]", primals_1326: "f32[320]", primals_1327: "f32[320]", primals_1328: "f32[320, 320]", primals_1329: "f32[80, 320]", primals_1330: "f32[320, 80]", primals_1331: "f32[320, 320]", primals_1332: "f32[80, 320]", primals_1333: "f32[320, 80]", primals_1334: "f32[320, 320]", primals_1335: "f32[80, 320]", primals_1336: "f32[320, 80]", primals_1337: "f32[320, 320]", primals_1338: "f32[320]", primals_1339: "f32[80, 320]", primals_1340: "f32[320, 80]", primals_1341: "f32[320]", primals_1342: "f32[320]", primals_1343: "f32[320, 320]", primals_1344: "f32[80, 320]", primals_1345: "f32[320, 80]", primals_1346: "f32[320, 1024]", primals_1347: "f32[80, 1024]", primals_1348: "f32[320, 80]", primals_1349: "f32[320, 1024]", primals_1350: "f32[80, 1024]", primals_1351: "f32[320, 80]", primals_1352: "f32[320, 320]", primals_1353: "f32[320]", primals_1354: "f32[80, 320]", primals_1355: "f32[320, 80]", primals_1356: "f32[320]", primals_1357: "f32[320]", primals_1358: "f32[2560, 320]", primals_1359: "f32[2560]", primals_1360: "f32[80, 320]", primals_1361: "f32[2560, 80]", primals_1362: "f32[320, 1280]", primals_1363: "f32[320]", primals_1364: "f32[80, 1280]", primals_1365: "f32[320, 80]", primals_1366: "f32[320, 320]", primals_1367: "f32[320]", primals_1368: "f32[80, 320]", primals_1369: "f32[320, 80]", primals_1370: "f32[320]", primals_1371: "f32[320]", primals_1372: "f32[4, 320, 3, 3]", primals_1373: "f32[4]", primals_1374: "f32[80, 320, 3, 3]", primals_1375: "f32[4, 80, 1, 1]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(primals_1, primals_2, primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = primals_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_1: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(primals_1, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_2: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_1, primals_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_2, 2.0);  convolution_2 = None
        add: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution, mul);  convolution = mul = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view: "f32[4, 32, 4, 65536]" = torch.ops.aten.view.default(add, [4, 32, 4, 65536])
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[4, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[4, 32, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul_1: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1: "f32[4, 128, 256, 256]" = torch.ops.aten.view.default(mul_1, [4, 128, 256, 256]);  mul_1 = None
        unsqueeze: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_7, 0)
        unsqueeze_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_6, 0)
        unsqueeze_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_2: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_5);  view_1 = unsqueeze_5 = None
        add_2: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_2);  mul_2 = unsqueeze_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_2)
        mul_3: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_2, sigmoid);  add_2 = sigmoid = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_3: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_3, primals_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_4: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_3, primals_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_5: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_4, primals_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_4: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_5, 2.0);  convolution_5 = None
        add_3: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_3, mul_4);  convolution_3 = mul_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_2: "f32[4, 32, 4, 65536]" = torch.ops.aten.view.default(add_3, [4, 32, 4, 65536])
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[4, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_2, getitem_3);  view_2 = None
        mul_5: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        view_3: "f32[4, 128, 256, 256]" = torch.ops.aten.view.default(mul_5, [4, 128, 256, 256]);  mul_5 = None
        unsqueeze_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_13, 0)
        unsqueeze_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_12, 0)
        unsqueeze_10: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_6: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_11);  view_3 = unsqueeze_11 = None
        add_5: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_8);  mul_6 = unsqueeze_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_1: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_5)
        mul_7: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_5, sigmoid_1);  add_5 = sigmoid_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_6: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_7, primals_14, primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_15 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_7: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_7, primals_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_8: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_7, primals_17, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_8: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_8, 2.0);  convolution_8 = None
        add_6: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_6, mul_8);  convolution_6 = mul_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_7: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(add, add_6);  add_6 = None
        div: "f32[4, 128, 256, 256]" = torch.ops.aten.div.Tensor(add_7, 1.0);  add_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_4: "f32[4, 32, 4, 65536]" = torch.ops.aten.view.default(div, [4, 32, 4, 65536])
        var_mean_2 = torch.ops.aten.var_mean.correction(view_4, [2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[4, 32, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[4, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_4, getitem_5);  view_4 = None
        mul_9: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        view_5: "f32[4, 128, 256, 256]" = torch.ops.aten.view.default(mul_9, [4, 128, 256, 256]);  mul_9 = None
        unsqueeze_12: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_19, 0)
        unsqueeze_13: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_18, 0)
        unsqueeze_16: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_10: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_5, unsqueeze_17);  view_5 = unsqueeze_17 = None
        add_9: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_14);  mul_10 = unsqueeze_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_2: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_9)
        mul_11: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_2);  add_9 = sigmoid_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_9: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_11, primals_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_10: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_11, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_11: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_10, primals_23, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_12: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_11, 2.0);  convolution_11 = None
        add_10: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_9, mul_12);  convolution_9 = mul_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_6: "f32[4, 32, 4, 65536]" = torch.ops.aten.view.default(add_10, [4, 32, 4, 65536])
        var_mean_3 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[4, 32, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[4, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_3: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_6, getitem_7);  view_6 = None
        mul_13: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        view_7: "f32[4, 128, 256, 256]" = torch.ops.aten.view.default(mul_13, [4, 128, 256, 256]);  mul_13 = None
        unsqueeze_18: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_25, 0)
        unsqueeze_19: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        unsqueeze_21: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_24, 0)
        unsqueeze_22: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        mul_14: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_23);  view_7 = unsqueeze_23 = None
        add_12: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_20);  mul_14 = unsqueeze_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_3: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_12)
        mul_15: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_12, sigmoid_3);  add_12 = sigmoid_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_12: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_15, primals_26, primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_13: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_15, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_14: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_13, primals_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_16: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_14, 2.0);  convolution_14 = None
        add_13: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_12, mul_16);  convolution_12 = mul_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_14: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(div, add_13);  add_13 = None
        div_1: "f32[4, 128, 256, 256]" = torch.ops.aten.div.Tensor(add_14, 1.0);  add_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[4, 128, 257, 257]" = torch.ops.aten.constant_pad_nd.default(div_1, [0, 1, 0, 1], 0.0);  div_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_15: "f32[4, 128, 128, 128]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_30, primals_31, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_16: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_32, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_17: "f32[4, 128, 128, 128]" = torch.ops.aten.convolution.default(convolution_16, primals_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_17: "f32[4, 128, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_17, 2.0);  convolution_17 = None
        add_15: "f32[4, 128, 128, 128]" = torch.ops.aten.add.Tensor(convolution_15, mul_17);  convolution_15 = mul_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_8: "f32[4, 32, 4, 16384]" = torch.ops.aten.view.default(add_15, [4, 32, 4, 16384])
        var_mean_4 = torch.ops.aten.var_mean.correction(view_8, [2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[4, 32, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[4, 32, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_16: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_4: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_4: "f32[4, 32, 4, 16384]" = torch.ops.aten.sub.Tensor(view_8, getitem_9);  view_8 = None
        mul_18: "f32[4, 32, 4, 16384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        view_9: "f32[4, 128, 128, 128]" = torch.ops.aten.view.default(mul_18, [4, 128, 128, 128]);  mul_18 = None
        unsqueeze_24: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_35, 0)
        unsqueeze_25: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        unsqueeze_26: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
        unsqueeze_27: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_34, 0)
        unsqueeze_28: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        unsqueeze_29: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
        mul_19: "f32[4, 128, 128, 128]" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_29);  view_9 = unsqueeze_29 = None
        add_17: "f32[4, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_26);  mul_19 = unsqueeze_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_4: "f32[4, 128, 128, 128]" = torch.ops.aten.sigmoid.default(add_17)
        mul_20: "f32[4, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_17, sigmoid_4);  add_17 = sigmoid_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_18: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_20, primals_36, primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_19: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_20, primals_38, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_20: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_19, primals_39, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_21: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_20, 2.0);  convolution_20 = None
        add_18: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_18, mul_21);  convolution_18 = mul_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_10: "f32[4, 32, 8, 16384]" = torch.ops.aten.view.default(add_18, [4, 32, 8, 16384])
        var_mean_5 = torch.ops.aten.var_mean.correction(view_10, [2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[4, 32, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[4, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_19: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_5: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_5: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_10, getitem_11);  view_10 = None
        mul_22: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        view_11: "f32[4, 256, 128, 128]" = torch.ops.aten.view.default(mul_22, [4, 256, 128, 128]);  mul_22 = None
        unsqueeze_30: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_41, 0)
        unsqueeze_31: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        unsqueeze_32: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
        unsqueeze_33: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_40, 0)
        unsqueeze_34: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
        unsqueeze_35: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        mul_23: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_11, unsqueeze_35);  view_11 = unsqueeze_35 = None
        add_20: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_32);  mul_23 = unsqueeze_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_5: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_20)
        mul_24: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_5);  add_20 = sigmoid_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_21: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_24, primals_42, primals_43, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_22: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_24, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_23: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_22, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_25: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_23, 2.0);  convolution_23 = None
        add_21: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_21, mul_25);  convolution_21 = mul_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_24: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(add_15, primals_46, primals_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_47 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_25: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(add_15, primals_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_26: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_25, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_26: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_26, 2.0);  convolution_26 = None
        add_22: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_24, mul_26);  convolution_24 = mul_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_23: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(add_22, add_21);  add_22 = add_21 = None
        div_2: "f32[4, 256, 128, 128]" = torch.ops.aten.div.Tensor(add_23, 1.0);  add_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_12: "f32[4, 32, 8, 16384]" = torch.ops.aten.view.default(div_2, [4, 32, 8, 16384])
        var_mean_6 = torch.ops.aten.var_mean.correction(view_12, [2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[4, 32, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[4, 32, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_24: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_6: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_6: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_12, getitem_13);  view_12 = None
        mul_27: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        view_13: "f32[4, 256, 128, 128]" = torch.ops.aten.view.default(mul_27, [4, 256, 128, 128]);  mul_27 = None
        unsqueeze_36: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_51, 0)
        unsqueeze_37: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
        unsqueeze_39: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_50, 0)
        unsqueeze_40: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        mul_28: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_13, unsqueeze_41);  view_13 = unsqueeze_41 = None
        add_25: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_38);  mul_28 = unsqueeze_38 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_6: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_25)
        mul_29: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, sigmoid_6);  add_25 = sigmoid_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_27: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_29, primals_52, primals_53, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_53 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_28: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_29, primals_54, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_29: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_28, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_30: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_29, 2.0);  convolution_29 = None
        add_26: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_27, mul_30);  convolution_27 = mul_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_14: "f32[4, 32, 8, 16384]" = torch.ops.aten.view.default(add_26, [4, 32, 8, 16384])
        var_mean_7 = torch.ops.aten.var_mean.correction(view_14, [2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[4, 32, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[4, 32, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_27: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_7: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_7: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_14, getitem_15);  view_14 = None
        mul_31: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        view_15: "f32[4, 256, 128, 128]" = torch.ops.aten.view.default(mul_31, [4, 256, 128, 128]);  mul_31 = None
        unsqueeze_42: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_57, 0)
        unsqueeze_43: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        unsqueeze_45: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_56, 0)
        unsqueeze_46: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        mul_32: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_15, unsqueeze_47);  view_15 = unsqueeze_47 = None
        add_28: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_44);  mul_32 = unsqueeze_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_7: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_28)
        mul_33: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_28, sigmoid_7);  add_28 = sigmoid_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_30: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_33, primals_58, primals_59, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_59 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_31: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_33, primals_60, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_32: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_31, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_34: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_32, 2.0);  convolution_32 = None
        add_29: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_30, mul_34);  convolution_30 = mul_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_30: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(div_2, add_29);  add_29 = None
        div_3: "f32[4, 256, 128, 128]" = torch.ops.aten.div.Tensor(add_30, 1.0);  add_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[4, 256, 129, 129]" = torch.ops.aten.constant_pad_nd.default(div_3, [0, 1, 0, 1], 0.0);  div_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_33: "f32[4, 256, 64, 64]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_62, primals_63, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_63 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_34: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_64, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_35: "f32[4, 256, 64, 64]" = torch.ops.aten.convolution.default(convolution_34, primals_65, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_35: "f32[4, 256, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_35, 2.0);  convolution_35 = None
        add_31: "f32[4, 256, 64, 64]" = torch.ops.aten.add.Tensor(convolution_33, mul_35);  convolution_33 = mul_35 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_16: "f32[4, 32, 8, 4096]" = torch.ops.aten.view.default(add_31, [4, 32, 8, 4096])
        var_mean_8 = torch.ops.aten.var_mean.correction(view_16, [2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[4, 32, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[4, 32, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_32: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_8: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_8: "f32[4, 32, 8, 4096]" = torch.ops.aten.sub.Tensor(view_16, getitem_17);  view_16 = None
        mul_36: "f32[4, 32, 8, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        view_17: "f32[4, 256, 64, 64]" = torch.ops.aten.view.default(mul_36, [4, 256, 64, 64]);  mul_36 = None
        unsqueeze_48: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_67, 0)
        unsqueeze_49: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
        unsqueeze_50: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
        unsqueeze_51: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_66, 0)
        unsqueeze_52: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
        unsqueeze_53: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 3);  unsqueeze_52 = None
        mul_37: "f32[4, 256, 64, 64]" = torch.ops.aten.mul.Tensor(view_17, unsqueeze_53);  view_17 = unsqueeze_53 = None
        add_33: "f32[4, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_50);  mul_37 = unsqueeze_50 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_8: "f32[4, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_33)
        mul_38: "f32[4, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_33, sigmoid_8);  add_33 = sigmoid_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_36: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_38, primals_68, primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_69 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_37: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_38, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_38: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_37, primals_71, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_39: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_38, 2.0);  convolution_38 = None
        add_34: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_36, mul_39);  convolution_36 = mul_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_18: "f32[4, 32, 16, 4096]" = torch.ops.aten.view.default(add_34, [4, 32, 16, 4096])
        var_mean_9 = torch.ops.aten.var_mean.correction(view_18, [2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[4, 32, 1, 1]" = var_mean_9[0]
        getitem_19: "f32[4, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_35: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_9: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_9: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_18, getitem_19);  view_18 = None
        mul_40: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        view_19: "f32[4, 512, 64, 64]" = torch.ops.aten.view.default(mul_40, [4, 512, 64, 64]);  mul_40 = None
        unsqueeze_54: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_73, 0)
        unsqueeze_55: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 2);  unsqueeze_54 = None
        unsqueeze_56: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
        unsqueeze_57: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_72, 0)
        unsqueeze_58: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
        unsqueeze_59: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
        mul_41: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_19, unsqueeze_59);  view_19 = unsqueeze_59 = None
        add_36: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_56);  mul_41 = unsqueeze_56 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_9: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_36)
        mul_42: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_36, sigmoid_9);  add_36 = sigmoid_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_39: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_42, primals_74, primals_75, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_40: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_42, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_41: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_40, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_43: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_41, 2.0);  convolution_41 = None
        add_37: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_39, mul_43);  convolution_39 = mul_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_42: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(add_31, primals_78, primals_79, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_43: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(add_31, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_44: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_43, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_44: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_44, 2.0);  convolution_44 = None
        add_38: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_42, mul_44);  convolution_42 = mul_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_39: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(add_38, add_37);  add_38 = add_37 = None
        div_4: "f32[4, 512, 64, 64]" = torch.ops.aten.div.Tensor(add_39, 1.0);  add_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_20: "f32[4, 32, 16, 4096]" = torch.ops.aten.view.default(div_4, [4, 32, 16, 4096])
        var_mean_10 = torch.ops.aten.var_mean.correction(view_20, [2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[4, 32, 1, 1]" = var_mean_10[0]
        getitem_21: "f32[4, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_40: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_10: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_10: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_20, getitem_21);  view_20 = None
        mul_45: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        view_21: "f32[4, 512, 64, 64]" = torch.ops.aten.view.default(mul_45, [4, 512, 64, 64]);  mul_45 = None
        unsqueeze_60: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_83, 0)
        unsqueeze_61: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
        unsqueeze_62: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 3);  unsqueeze_61 = None
        unsqueeze_63: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
        unsqueeze_64: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
        unsqueeze_65: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
        mul_46: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_21, unsqueeze_65);  view_21 = unsqueeze_65 = None
        add_41: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_62);  mul_46 = unsqueeze_62 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_10: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_41)
        mul_47: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_10);  add_41 = sigmoid_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_45: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_47, primals_84, primals_85, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_46: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_47, primals_86, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_47: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_46, primals_87, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_48: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_47, 2.0);  convolution_47 = None
        add_42: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_45, mul_48);  convolution_45 = mul_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_22: "f32[4, 32, 16, 4096]" = torch.ops.aten.view.default(add_42, [4, 32, 16, 4096])
        var_mean_11 = torch.ops.aten.var_mean.correction(view_22, [2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[4, 32, 1, 1]" = var_mean_11[0]
        getitem_23: "f32[4, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_43: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_11: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_11: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_22, getitem_23);  view_22 = None
        mul_49: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        view_23: "f32[4, 512, 64, 64]" = torch.ops.aten.view.default(mul_49, [4, 512, 64, 64]);  mul_49 = None
        unsqueeze_66: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_89, 0)
        unsqueeze_67: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 2);  unsqueeze_66 = None
        unsqueeze_68: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
        unsqueeze_69: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_88, 0)
        unsqueeze_70: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
        unsqueeze_71: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 3);  unsqueeze_70 = None
        mul_50: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_23, unsqueeze_71);  view_23 = unsqueeze_71 = None
        add_44: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_68);  mul_50 = unsqueeze_68 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_11: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_44)
        mul_51: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_44, sigmoid_11);  add_44 = sigmoid_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_48: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_51, primals_90, primals_91, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_49: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_51, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_50: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_49, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_52: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_50, 2.0);  convolution_50 = None
        add_45: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_48, mul_52);  convolution_48 = mul_52 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_46: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(div_4, add_45);  add_45 = None
        div_5: "f32[4, 512, 64, 64]" = torch.ops.aten.div.Tensor(add_46, 1.0);  add_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/torch/nn/functional.py:5209 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[4, 512, 65, 65]" = torch.ops.aten.constant_pad_nd.default(div_5, [0, 1, 0, 1], 0.0);  div_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_51: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_94, primals_95, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_52: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_96, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_53: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_52, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_53: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_53, 2.0);  convolution_53 = None
        add_47: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_51, mul_53);  convolution_51 = mul_53 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_24: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(add_47, [4, 32, 16, 1024])
        var_mean_12 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[4, 32, 1, 1]" = var_mean_12[0]
        getitem_25: "f32[4, 32, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_48: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_12: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_12: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_24, getitem_25);  view_24 = None
        mul_54: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        view_25: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_54, [4, 512, 32, 32]);  mul_54 = None
        unsqueeze_72: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_99, 0)
        unsqueeze_73: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
        unsqueeze_74: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
        unsqueeze_75: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_98, 0)
        unsqueeze_76: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        unsqueeze_77: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
        mul_55: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_25, unsqueeze_77);  view_25 = unsqueeze_77 = None
        add_49: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_74);  mul_55 = unsqueeze_74 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_12: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_49)
        mul_56: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_49, sigmoid_12);  add_49 = sigmoid_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_54: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_56, primals_100, primals_101, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_101 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_55: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_56, primals_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_56: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_55, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_57: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_56, 2.0);  convolution_56 = None
        add_50: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_54, mul_57);  convolution_54 = mul_57 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_26: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(add_50, [4, 32, 16, 1024])
        var_mean_13 = torch.ops.aten.var_mean.correction(view_26, [2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[4, 32, 1, 1]" = var_mean_13[0]
        getitem_27: "f32[4, 32, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_51: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_13: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_13: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_26, getitem_27);  view_26 = None
        mul_58: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        view_27: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_58, [4, 512, 32, 32]);  mul_58 = None
        unsqueeze_78: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_105, 0)
        unsqueeze_79: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
        unsqueeze_80: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
        unsqueeze_81: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_104, 0)
        unsqueeze_82: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        unsqueeze_83: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
        mul_59: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_27, unsqueeze_83);  view_27 = unsqueeze_83 = None
        add_52: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_80);  mul_59 = unsqueeze_80 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_13: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_52)
        mul_60: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_52, sigmoid_13);  add_52 = sigmoid_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_57: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_60, primals_106, primals_107, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_58: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_60, primals_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_59: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_58, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_61: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_59, 2.0);  convolution_59 = None
        add_53: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_57, mul_61);  convolution_57 = mul_61 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_54: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_47, add_53);  add_53 = None
        div_6: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_54, 1.0);  add_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_28: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(div_6, [4, 32, 16, 1024])
        var_mean_14 = torch.ops.aten.var_mean.correction(view_28, [2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[4, 32, 1, 1]" = var_mean_14[0]
        getitem_29: "f32[4, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_55: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_14: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_14: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_28, getitem_29);  view_28 = None
        mul_62: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        view_29: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_62, [4, 512, 32, 32]);  mul_62 = None
        unsqueeze_84: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_111, 0)
        unsqueeze_85: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
        unsqueeze_87: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_110, 0)
        unsqueeze_88: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        mul_63: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_29, unsqueeze_89);  view_29 = unsqueeze_89 = None
        add_56: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_86);  mul_63 = unsqueeze_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_14: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_56)
        mul_64: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_14);  add_56 = sigmoid_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_60: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_64, primals_112, primals_113, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_113 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_61: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_64, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_62: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_61, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_65: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_62, 2.0);  convolution_62 = None
        add_57: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_60, mul_65);  convolution_60 = mul_65 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_30: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(add_57, [4, 32, 16, 1024])
        var_mean_15 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[4, 32, 1, 1]" = var_mean_15[0]
        getitem_31: "f32[4, 32, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_58: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_15: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_15: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_30, getitem_31);  view_30 = None
        mul_66: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        view_31: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_66, [4, 512, 32, 32]);  mul_66 = None
        unsqueeze_90: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_117, 0)
        unsqueeze_91: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        unsqueeze_93: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_116, 0)
        unsqueeze_94: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        mul_67: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_31, unsqueeze_95);  view_31 = unsqueeze_95 = None
        add_59: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_92);  mul_67 = unsqueeze_92 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_15: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_59)
        mul_68: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_59, sigmoid_15);  add_59 = sigmoid_15 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_63: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_68, primals_118, primals_119, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_119 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_64: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_68, primals_120, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_65: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_64, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_69: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_65, 2.0);  convolution_65 = None
        add_60: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_63, mul_69);  convolution_63 = mul_69 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_61: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_6, add_60);  add_60 = None
        div_7: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_61, 1.0);  add_61 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_32: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(div_7, [4, 32, 16, 1024])
        var_mean_16 = torch.ops.aten.var_mean.correction(view_32, [2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[4, 32, 1, 1]" = var_mean_16[0]
        getitem_33: "f32[4, 32, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_62: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_16: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_16: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_32, getitem_33);  view_32 = None
        mul_70: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        view_33: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_70, [4, 512, 32, 32]);  mul_70 = None
        unsqueeze_96: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_123, 0)
        unsqueeze_97: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        unsqueeze_98: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
        unsqueeze_99: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_122, 0)
        unsqueeze_100: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        unsqueeze_101: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
        mul_71: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_33, unsqueeze_101);  view_33 = unsqueeze_101 = None
        add_63: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_98);  mul_71 = unsqueeze_98 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_16: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_63)
        mul_72: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_63, sigmoid_16);  add_63 = sigmoid_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_66: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_72, primals_124, primals_125, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_125 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_67: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_72, primals_126, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_68: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_67, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_73: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_68, 2.0);  convolution_68 = None
        add_64: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_66, mul_73);  convolution_66 = mul_73 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_34: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(add_64, [4, 32, 16, 1024])
        var_mean_17 = torch.ops.aten.var_mean.correction(view_34, [2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[4, 32, 1, 1]" = var_mean_17[0]
        getitem_35: "f32[4, 32, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_65: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_17: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_17: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_34, getitem_35);  view_34 = None
        mul_74: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        view_35: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_74, [4, 512, 32, 32]);  mul_74 = None
        unsqueeze_102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_129, 0)
        unsqueeze_103: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        unsqueeze_105: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_128, 0)
        unsqueeze_106: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        mul_75: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_35, unsqueeze_107);  view_35 = unsqueeze_107 = None
        add_66: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_104);  mul_75 = unsqueeze_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_17: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_66)
        mul_76: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_17);  add_66 = sigmoid_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_69: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_76, primals_130, primals_131, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_131 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_70: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_76, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_71: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_70, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_77: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_71, 2.0);  convolution_71 = None
        add_67: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_69, mul_77);  convolution_69 = mul_77 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_68: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_7, add_67);  add_67 = None
        div_8: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_68, 1);  add_68 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3246 in __call__, code: hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        view_36: "f32[4, 512, 1024]" = torch.ops.aten.view.default(div_8, [4, 512, 1024])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3259 in __call__, code: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        view_37: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(view_36, [4, 32, 16, 1024])
        var_mean_18 = torch.ops.aten.var_mean.correction(view_37, [2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[4, 32, 1, 1]" = var_mean_18[0]
        getitem_37: "f32[4, 32, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_69: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_18: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_18: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_37, getitem_37);  view_37 = None
        mul_78: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        view_38: "f32[4, 512, 1024]" = torch.ops.aten.view.default(mul_78, [4, 512, 1024]);  mul_78 = None
        unsqueeze_108: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_135, 0);  primals_135 = None
        unsqueeze_109: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_134, 0)
        unsqueeze_111: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
        mul_79: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_38, unsqueeze_111);  view_38 = unsqueeze_111 = None
        add_70: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_109);  mul_79 = unsqueeze_109 = None
        squeeze_36: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_37, [2, 3]);  getitem_37 = None
        squeeze_37: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_18, [2, 3]);  rsqrt_18 = None
        permute_2: "f32[4, 1024, 512]" = torch.ops.aten.permute.default(add_70, [0, 2, 1]);  add_70 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        expand: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(permute_2, [4, 1024, 512])
        expand_1: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_3, [4, 512, 512]);  permute_3 = None
        bmm: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_1);  expand_1 = None
        add_71: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm, primals_137);  bmm = primals_137 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_4: "f32[512, 4]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        clone_9: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_42: "f32[4096, 512]" = torch.ops.aten.view.default(clone_9, [4096, 512]);  clone_9 = None
        mm: "f32[4096, 4]" = torch.ops.aten.mm.default(view_42, permute_4)
        permute_5: "f32[4, 512]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        mm_1: "f32[4096, 512]" = torch.ops.aten.mm.default(mm, permute_5)
        view_45: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_1, [4, 1024, 512]);  mm_1 = None
        mul_80: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_45, 2.0);  view_45 = None
        add_72: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_71, mul_80);  add_71 = mul_80 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_6: "f32[512, 512]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        expand_3: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_6, [4, 512, 512]);  permute_6 = None
        bmm_1: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_3);  expand_3 = None
        add_73: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm_1, primals_141);  bmm_1 = primals_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_7: "f32[512, 4]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        mm_2: "f32[4096, 4]" = torch.ops.aten.mm.default(view_42, permute_7)
        permute_8: "f32[4, 512]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        mm_3: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_2, permute_8)
        view_52: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_3, [4, 1024, 512]);  mm_3 = None
        mul_81: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_52, 2.0);  view_52 = None
        add_74: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, mul_81);  add_73 = mul_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        expand_5: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_9, [4, 512, 512]);  permute_9 = None
        bmm_2: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_5);  expand = expand_5 = None
        add_75: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm_2, primals_145);  bmm_2 = primals_145 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_10: "f32[512, 4]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        mm_4: "f32[4096, 4]" = torch.ops.aten.mm.default(view_42, permute_10)
        permute_11: "f32[4, 512]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        mm_5: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_4, permute_11)
        view_59: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_5, [4, 1024, 512]);  mm_5 = None
        mul_82: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_59, 2.0);  view_59 = None
        add_76: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_75, mul_82);  add_75 = mul_82 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_63: "f32[4, 1024, 1, 512]" = torch.ops.aten.view.default(add_72, [4, -1, 1, 512]);  add_72 = None
        permute_15: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        view_64: "f32[4, 1024, 1, 512]" = torch.ops.aten.view.default(add_74, [4, -1, 1, 512]);  add_74 = None
        permute_16: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        view_65: "f32[4, 1024, 1, 512]" = torch.ops.aten.view.default(add_76, [4, -1, 1, 512]);  add_76 = None
        permute_17: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_15, permute_16, permute_17, None, True)
        getitem_38: "f32[4, 1, 1024, 512]" = _scaled_dot_product_efficient_attention[0]
        getitem_39: "f32[4, 1, 1024]" = _scaled_dot_product_efficient_attention[1]
        getitem_40: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_18: "f32[4, 1024, 1, 512]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_66: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_18, [4, -1, 512]);  permute_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_67: "f32[4096, 512]" = torch.ops.aten.view.default(view_66, [4096, 512]);  view_66 = None
        permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        addmm: "f32[4096, 512]" = torch.ops.aten.addmm.default(primals_149, view_67, permute_19);  primals_149 = None
        view_68: "f32[4, 1024, 512]" = torch.ops.aten.view.default(addmm, [4, 1024, 512]);  addmm = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_20: "f32[512, 4]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        mm_6: "f32[4096, 4]" = torch.ops.aten.mm.default(view_67, permute_20);  view_67 = None
        permute_21: "f32[4, 512]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        mm_7: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_6, permute_21)
        view_72: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_7, [4, 1024, 512]);  mm_7 = None
        mul_83: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_72, 2.0);  view_72 = None
        add_77: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_68, mul_83);  view_68 = mul_83 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3299 in __call__, code: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        permute_22: "f32[4, 512, 1024]" = torch.ops.aten.permute.default(add_77, [0, 2, 1]);  add_77 = None
        view_76: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(permute_22, [4, 512, 32, 32]);  permute_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3302 in __call__, code: hidden_states = hidden_states + residual
        add_78: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(view_76, div_8);  view_76 = div_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_9: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_78, 1);  add_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_13: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(div_9, memory_format = torch.contiguous_format)
        view_77: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(clone_13, [4, 32, 16, 1024])
        var_mean_19 = torch.ops.aten.var_mean.correction(view_77, [2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[4, 32, 1, 1]" = var_mean_19[0]
        getitem_43: "f32[4, 32, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_79: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_19: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_19: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_77, getitem_43);  view_77 = None
        mul_84: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        view_78: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_84, [4, 512, 32, 32]);  mul_84 = None
        unsqueeze_112: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_153, 0)
        unsqueeze_113: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None
        unsqueeze_114: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 3);  unsqueeze_113 = None
        unsqueeze_115: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_152, 0)
        unsqueeze_116: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
        unsqueeze_117: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 3);  unsqueeze_116 = None
        mul_85: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_78, unsqueeze_117);  view_78 = unsqueeze_117 = None
        add_80: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_114);  mul_85 = unsqueeze_114 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_18: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_80)
        mul_86: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_80, sigmoid_18);  add_80 = sigmoid_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_72: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_86, primals_154, primals_155, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_155 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_73: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_86, primals_156, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_74: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_73, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_87: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_74, 2.0);  convolution_74 = None
        add_81: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_72, mul_87);  convolution_72 = mul_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_79: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(add_81, [4, 32, 16, 1024])
        var_mean_20 = torch.ops.aten.var_mean.correction(view_79, [2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[4, 32, 1, 1]" = var_mean_20[0]
        getitem_45: "f32[4, 32, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_20: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_79, getitem_45);  view_79 = None
        mul_88: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        view_80: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_88, [4, 512, 32, 32]);  mul_88 = None
        unsqueeze_118: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_159, 0)
        unsqueeze_119: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 2);  unsqueeze_118 = None
        unsqueeze_120: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 3);  unsqueeze_119 = None
        unsqueeze_121: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_158, 0)
        unsqueeze_122: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
        unsqueeze_123: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
        mul_89: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_80, unsqueeze_123);  view_80 = unsqueeze_123 = None
        add_83: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_120);  mul_89 = unsqueeze_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_19: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_83)
        mul_90: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_83, sigmoid_19);  add_83 = sigmoid_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_75: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_90, primals_160, primals_161, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_161 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_76: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_90, primals_162, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_77: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_76, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_91: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_77, 2.0);  convolution_77 = None
        add_84: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_75, mul_91);  convolution_75 = mul_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_85: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_9, add_84);  div_9 = add_84 = None
        div_10: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_85, 1);  add_85 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:23 in my_vae_encoder_fwd, code: sample = self.conv_norm_out(sample)
        clone_15: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(div_10, memory_format = torch.contiguous_format);  div_10 = None
        view_81: "f32[4, 32, 16, 1024]" = torch.ops.aten.view.default(clone_15, [4, 32, 16, 1024])
        var_mean_21 = torch.ops.aten.var_mean.correction(view_81, [2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[4, 32, 1, 1]" = var_mean_21[0]
        getitem_47: "f32[4, 32, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_86: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_21: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_21: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_81, getitem_47);  view_81 = None
        mul_92: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        view_82: "f32[4, 512, 32, 32]" = torch.ops.aten.view.default(mul_92, [4, 512, 32, 32]);  mul_92 = None
        unsqueeze_124: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_165, 0)
        unsqueeze_125: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
        unsqueeze_126: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
        unsqueeze_127: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_164, 0)
        unsqueeze_128: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        unsqueeze_129: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
        mul_93: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_82, unsqueeze_129);  view_82 = unsqueeze_129 = None
        add_87: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_126);  mul_93 = unsqueeze_126 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:24 in my_vae_encoder_fwd, code: sample = self.conv_act(sample)
        sigmoid_20: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_87)
        mul_94: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_87, sigmoid_20);  add_87 = sigmoid_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_78: "f32[4, 8, 32, 32]" = torch.ops.aten.convolution.default(mul_94, primals_166, primals_167, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_167 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_79: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_94, primals_168, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_80: "f32[4, 8, 32, 32]" = torch.ops.aten.convolution.default(convolution_79, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_95: "f32[4, 8, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_80, 2.0);  convolution_80 = None
        add_88: "f32[4, 8, 32, 32]" = torch.ops.aten.add.Tensor(convolution_78, mul_95);  convolution_78 = mul_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py:258 in _encode, code: enc = self.quant_conv(enc)
        convolution_81: "f32[4, 8, 32, 32]" = torch.ops.aten.convolution.default(add_88, primals_170, primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_171 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/vae.py:783 in __init__, code: self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        split = torch.ops.aten.split.Tensor(convolution_81, 4, 1);  convolution_81 = None
        getitem_48: "f32[4, 4, 32, 32]" = split[0]
        getitem_49: "f32[4, 4, 32, 32]" = split[1];  split = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/vae.py:784 in __init__, code: self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        clamp_min: "f32[4, 4, 32, 32]" = torch.ops.aten.clamp_min.default(getitem_49, -30.0)
        clamp_max: "f32[4, 4, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min, 20.0);  clamp_min = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/vae.py:786 in __init__, code: self.std = torch.exp(0.5 * self.logvar)
        mul_96: "f32[4, 4, 32, 32]" = torch.ops.aten.mul.Tensor(clamp_max, 0.5);  clamp_max = None
        exp: "f32[4, 4, 32, 32]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 4, 32, 32]" = torch.ops.prims.inductor_random.default([4, 4, 32, 32], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/vae.py:801 in sample, code: x = self.mean + self.std * sample
        mul_97: "f32[4, 4, 32, 32]" = torch.ops.aten.mul.Tensor(exp, inductor_random_default);  exp = None
        add_89: "f32[4, 4, 32, 32]" = torch.ops.aten.add.Tensor(getitem_48, mul_97);  getitem_48 = mul_97 = None
        
         # File: /home/elicer/cyclegan-turbo/src/cyclegan_turbo.py:27 in forward, code: return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor
        mul_98: "f32[4, 4, 32, 32]" = torch.ops.aten.mul.Tensor(add_89, 0.18215);  add_89 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py:927 in get_time_embed, code: timesteps = timesteps.expand(sample.shape[0])
        expand_6: "i64[4]" = torch.ops.aten.expand.default(primals_172, [4]);  primals_172 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:57 in get_timestep_embedding, code: exponent = -math.log(max_period) * torch.arange(
        iota: "i64[160]" = torch.ops.prims.iota.default(160, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_99: "i64[160]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_90: "i64[160]" = torch.ops.aten.add.Tensor(mul_99, 0);  mul_99 = None
        convert_element_type: "f32[160]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32);  add_90 = None
        mul_100: "f32[160]" = torch.ops.aten.mul.Tensor(convert_element_type, -9.210340371976184);  convert_element_type = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:60 in get_timestep_embedding, code: exponent = exponent / (half_dim - downscale_freq_shift)
        div_11: "f32[160]" = torch.ops.aten.div.Tensor(mul_100, 160);  mul_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:62 in get_timestep_embedding, code: emb = torch.exp(exponent)
        exp_2: "f32[160]" = torch.ops.aten.exp.default(div_11);  div_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:63 in get_timestep_embedding, code: emb = timesteps[:, None].float() * emb[None, :]
        unsqueeze_130: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(expand_6, 1);  expand_6 = None
        convert_element_type_1: "f32[4, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_130, torch.float32);  unsqueeze_130 = None
        unsqueeze_131: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(exp_2, 0);  exp_2 = None
        mul_101: "f32[4, 160]" = torch.ops.aten.mul.Tensor(convert_element_type_1, unsqueeze_131);  convert_element_type_1 = unsqueeze_131 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:66 in get_timestep_embedding, code: emb = scale * emb
        mul_102: "f32[4, 160]" = torch.ops.aten.mul.Tensor(mul_101, 1);  mul_101 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:69 in get_timestep_embedding, code: emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        sin: "f32[4, 160]" = torch.ops.aten.sin.default(mul_102)
        cos: "f32[4, 160]" = torch.ops.aten.cos.default(mul_102);  mul_102 = None
        cat: "f32[4, 320]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:73 in get_timestep_embedding, code: emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        slice_4: "f32[4, 160]" = torch.ops.aten.slice.Tensor(cat, 1, 160, 9223372036854775807)
        slice_6: "f32[4, 160]" = torch.ops.aten.slice.Tensor(cat, 1, 0, 160);  cat = None
        cat_1: "f32[4, 320]" = torch.ops.aten.cat.default([slice_4, slice_6], -1);  slice_4 = slice_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:1304 in forward, code: sample = self.linear_1(sample)
        permute_23: "f32[320, 1280]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        addmm_1: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_174, cat_1, permute_23);  primals_174 = cat_1 = permute_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:1307 in forward, code: sample = self.act(sample)
        sigmoid_21: "f32[4, 1280]" = torch.ops.aten.sigmoid.default(addmm_1)
        mul_103: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(addmm_1, sigmoid_21);  addmm_1 = sigmoid_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/embeddings.py:1309 in forward, code: sample = self.linear_2(sample)
        permute_24: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        addmm_2: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_176, mul_103, permute_24);  primals_176 = mul_103 = permute_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_82: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_98, primals_178, primals_179, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_179 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_83: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_98, primals_180, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_84: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_83, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_104: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_84, 1.0);  convolution_84 = None
        add_91: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_82, mul_104);  convolution_82 = mul_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_83: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_91, [4, 32, 10, 1024])
        var_mean_22 = torch.ops.aten.var_mean.correction(view_83, [2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[4, 32, 1, 1]" = var_mean_22[0]
        getitem_51: "f32[4, 32, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_22: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_22: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_83, getitem_51);  view_83 = None
        mul_105: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        view_84: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_105, [4, 320, 32, 32]);  mul_105 = None
        unsqueeze_132: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_183, 0)
        unsqueeze_133: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
        unsqueeze_135: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_182, 0)
        unsqueeze_136: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        mul_106: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_84, unsqueeze_137);  view_84 = unsqueeze_137 = None
        add_93: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_134);  mul_106 = unsqueeze_134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_22: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
        mul_107: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_22);  add_93 = sigmoid_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_85: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_107, primals_184, primals_185, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_185 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_86: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_107, primals_186, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_87: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_86, primals_187, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_108: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_87, 1.0);  convolution_87 = None
        add_94: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_85, mul_108);  convolution_85 = mul_108 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:345 in forward, code: temb = self.nonlinearity(temb)
        sigmoid_23: "f32[4, 1280]" = torch.ops.aten.sigmoid.default(addmm_2)
        mul_109: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(addmm_2, sigmoid_23);  addmm_2 = sigmoid_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_25: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        addmm_3: "f32[4, 320]" = torch.ops.aten.addmm.default(primals_189, mul_109, permute_25);  primals_189 = permute_25 = None
        unsqueeze_138: "f32[4, 320, 1]" = torch.ops.aten.unsqueeze.default(addmm_3, 2);  addmm_3 = None
        unsqueeze_139: "f32[4, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_95: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_94, unsqueeze_139);  add_94 = unsqueeze_139 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_85: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_95, [4, 32, 10, 1024])
        var_mean_23 = torch.ops.aten.var_mean.correction(view_85, [2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[4, 32, 1, 1]" = var_mean_23[0]
        getitem_53: "f32[4, 32, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_96: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_23: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_23: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_85, getitem_53);  view_85 = None
        mul_110: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        view_86: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_110, [4, 320, 32, 32]);  mul_110 = None
        unsqueeze_140: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_191, 0)
        unsqueeze_141: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        unsqueeze_142: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
        unsqueeze_143: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_190, 0)
        unsqueeze_144: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_111: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_86, unsqueeze_145);  view_86 = unsqueeze_145 = None
        add_97: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_142);  mul_111 = unsqueeze_142 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_24: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_97)
        mul_112: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_97, sigmoid_24);  add_97 = sigmoid_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_88: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_112, primals_192, primals_193, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_193 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_89: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_112, primals_194, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_90: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_89, primals_195, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_113: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_90, 1.0);  convolution_90 = None
        add_98: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_88, mul_113);  convolution_88 = mul_113 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_99: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_91, add_98);  add_98 = None
        div_12: "f32[4, 320, 32, 32]" = torch.ops.aten.div.Tensor(add_99, 1.0);  add_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_87: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(div_12, [4, 32, 10, 1024])
        var_mean_24 = torch.ops.aten.var_mean.correction(view_87, [2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[4, 32, 1, 1]" = var_mean_24[0]
        getitem_55: "f32[4, 32, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_24: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_24: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_87, getitem_55);  view_87 = None
        mul_114: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        view_88: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_114, [4, 320, 32, 32]);  mul_114 = None
        unsqueeze_146: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_197, 0);  primals_197 = None
        unsqueeze_147: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        unsqueeze_149: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_196, 0)
        unsqueeze_150: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_115: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_88, unsqueeze_151);  view_88 = unsqueeze_151 = None
        add_101: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_148);  mul_115 = unsqueeze_148 = None
        squeeze_48: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_55, [2, 3]);  getitem_55 = None
        squeeze_49: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_24, [2, 3]);  rsqrt_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_26: "f32[4, 32, 32, 320]" = torch.ops.aten.permute.default(add_101, [0, 2, 3, 1]);  add_101 = None
        view_89: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_26, [4, 1024, 320]);  permute_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_27: "f32[320, 320]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        expand_7: "f32[4, 1024, 320]" = torch.ops.aten.expand.default(view_89, [4, 1024, 320])
        expand_8: "f32[4, 320, 320]" = torch.ops.aten.expand.default(permute_27, [4, 320, 320]);  permute_27 = None
        bmm_3: "f32[4, 1024, 320]" = torch.ops.aten.bmm.default(expand_7, expand_8);  expand_7 = expand_8 = None
        add_102: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(bmm_3, primals_199);  bmm_3 = primals_199 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_28: "f32[320, 80]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        clone_17: "f32[4, 1024, 320]" = torch.ops.aten.clone.default(view_89, memory_format = torch.contiguous_format);  view_89 = None
        view_93: "f32[4096, 320]" = torch.ops.aten.view.default(clone_17, [4096, 320]);  clone_17 = None
        mm_8: "f32[4096, 80]" = torch.ops.aten.mm.default(view_93, permute_28)
        permute_29: "f32[80, 320]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        mm_9: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_8, permute_29)
        view_96: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_9, [4, 1024, 320]);  mm_9 = None
        mul_116: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_96, 1.0);  view_96 = None
        add_103: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_102, mul_116);  add_102 = mul_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_56: "f32[4, 1024, 1]" = var_mean_25[0]
        getitem_57: "f32[4, 1024, 1]" = var_mean_25[1];  var_mean_25 = None
        add_104: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_25: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_25: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_103, getitem_57);  getitem_57 = None
        mul_117: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        mul_118: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_117, primals_202)
        add_105: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_118, primals_203);  mul_118 = primals_203 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_30: "f32[320, 320]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        view_97: "f32[4096, 320]" = torch.ops.aten.view.default(add_105, [4096, 320]);  add_105 = None
        mm_10: "f32[4096, 320]" = torch.ops.aten.mm.default(view_97, permute_30)
        view_98: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_10, [4, 1024, 320]);  mm_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_31: "f32[320, 80]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        mm_11: "f32[4096, 80]" = torch.ops.aten.mm.default(view_97, permute_31)
        permute_32: "f32[80, 320]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        mm_12: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_11, permute_32)
        view_102: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_12, [4, 1024, 320]);  mm_12 = None
        mul_119: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_102, 1.0);  view_102 = None
        add_106: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_98, mul_119);  view_98 = mul_119 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_33: "f32[320, 320]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        mm_13: "f32[4096, 320]" = torch.ops.aten.mm.default(view_97, permute_33)
        view_106: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_13, [4, 1024, 320]);  mm_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_34: "f32[320, 80]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
        mm_14: "f32[4096, 80]" = torch.ops.aten.mm.default(view_97, permute_34)
        permute_35: "f32[80, 320]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
        mm_15: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_14, permute_35)
        view_110: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_15, [4, 1024, 320]);  mm_15 = None
        mul_120: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_110, 1.0);  view_110 = None
        add_107: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_106, mul_120);  view_106 = mul_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_36: "f32[320, 320]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
        mm_16: "f32[4096, 320]" = torch.ops.aten.mm.default(view_97, permute_36)
        view_114: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_16, [4, 1024, 320]);  mm_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_37: "f32[320, 80]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        mm_17: "f32[4096, 80]" = torch.ops.aten.mm.default(view_97, permute_37)
        permute_38: "f32[80, 320]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
        mm_18: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_17, permute_38)
        view_118: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_18, [4, 1024, 320]);  mm_18 = None
        mul_121: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_118, 1.0);  view_118 = None
        add_108: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_114, mul_121);  view_114 = mul_121 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_125: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_106, [4, -1, 5, 64]);  add_106 = None
        permute_42: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        view_127: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_107, [4, -1, 5, 64]);  add_107 = None
        permute_43: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        view_129: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_108, [4, -1, 5, 64]);  add_108 = None
        permute_44: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_43, permute_44, None, True)
        getitem_58: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_59: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_60: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_61: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_45: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3])
        view_130: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_45, [4, -1, 320]);  permute_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_131: "f32[4096, 320]" = torch.ops.aten.view.default(view_130, [4096, 320]);  view_130 = None
        permute_46: "f32[320, 320]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        addmm_4: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_214, view_131, permute_46);  primals_214 = None
        view_132: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_4, [4, 1024, 320]);  addmm_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_47: "f32[320, 80]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
        mm_19: "f32[4096, 80]" = torch.ops.aten.mm.default(view_131, permute_47);  view_131 = None
        permute_48: "f32[80, 320]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
        mm_20: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_19, permute_48)
        view_136: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_20, [4, 1024, 320]);  mm_20 = None
        mul_122: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_136, 1.0);  view_136 = None
        add_109: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_132, mul_122);  view_132 = mul_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_13: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_109, 1.0);  add_109 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_110: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_13, add_103);  div_13 = add_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 1024, 1]" = var_mean_26[0]
        getitem_63: "f32[4, 1024, 1]" = var_mean_26[1];  var_mean_26 = None
        add_111: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_26: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_26: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_110, getitem_63);  getitem_63 = None
        mul_123: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        mul_124: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_123, primals_217)
        add_112: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_124, primals_218);  mul_124 = primals_218 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_49: "f32[320, 320]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        view_140: "f32[4096, 320]" = torch.ops.aten.view.default(add_112, [4096, 320]);  add_112 = None
        mm_21: "f32[4096, 320]" = torch.ops.aten.mm.default(view_140, permute_49)
        view_141: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_21, [4, 1024, 320]);  mm_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_50: "f32[320, 80]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
        mm_22: "f32[4096, 80]" = torch.ops.aten.mm.default(view_140, permute_50)
        permute_51: "f32[80, 320]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        mm_23: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_22, permute_51)
        view_145: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_23, [4, 1024, 320]);  mm_23 = None
        mul_125: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_145, 1.0);  view_145 = None
        add_113: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_141, mul_125);  view_141 = mul_125 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_52: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
        view_148: "f32[308, 1024]" = torch.ops.aten.view.default(primals_177, [308, 1024]);  primals_177 = None
        mm_24: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_52);  permute_52 = None
        view_149: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_24, [4, 77, 320]);  mm_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_53: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
        mm_25: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_53);  permute_53 = None
        permute_54: "f32[80, 320]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
        mm_26: "f32[308, 320]" = torch.ops.aten.mm.default(mm_25, permute_54)
        view_153: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_26, [4, 77, 320]);  mm_26 = None
        mul_126: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_153, 1.0);  view_153 = None
        add_114: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_149, mul_126);  view_149 = mul_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_55: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
        mm_27: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_55);  permute_55 = None
        view_157: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_27, [4, 77, 320]);  mm_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_56: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
        mm_28: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_56);  permute_56 = None
        permute_57: "f32[80, 320]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        mm_29: "f32[308, 320]" = torch.ops.aten.mm.default(mm_28, permute_57)
        view_161: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_29, [4, 77, 320]);  mm_29 = None
        mul_127: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_161, 1.0);  view_161 = None
        add_115: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_157, mul_127);  view_157 = mul_127 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_168: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_113, [4, -1, 5, 64]);  add_113 = None
        permute_61: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        view_170: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_114, [4, -1, 5, 64]);  add_114 = None
        permute_62: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        view_172: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_115, [4, -1, 5, 64]);  add_115 = None
        permute_63: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_62, permute_63, None, True)
        getitem_64: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_65: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_66: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_67: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_64: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_64, [0, 2, 1, 3])
        view_173: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_64, [4, -1, 320]);  permute_64 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_174: "f32[4096, 320]" = torch.ops.aten.view.default(view_173, [4096, 320]);  view_173 = None
        permute_65: "f32[320, 320]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
        addmm_5: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_229, view_174, permute_65);  primals_229 = None
        view_175: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_5, [4, 1024, 320]);  addmm_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_66: "f32[320, 80]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
        mm_30: "f32[4096, 80]" = torch.ops.aten.mm.default(view_174, permute_66);  view_174 = None
        permute_67: "f32[80, 320]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
        mm_31: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_30, permute_67)
        view_179: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_31, [4, 1024, 320]);  mm_31 = None
        mul_128: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_179, 1.0);  view_179 = None
        add_116: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_175, mul_128);  view_175 = mul_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_14: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_116, 1.0);  add_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_117: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_14, add_110);  div_14 = add_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_68: "f32[4, 1024, 1]" = var_mean_27[0]
        getitem_69: "f32[4, 1024, 1]" = var_mean_27[1];  var_mean_27 = None
        add_118: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_27: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_27: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_117, getitem_69);  getitem_69 = None
        mul_129: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        mul_130: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_129, primals_232)
        add_119: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_130, primals_233);  mul_130 = primals_233 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_183: "f32[4096, 320]" = torch.ops.aten.view.default(add_119, [4096, 320]);  add_119 = None
        permute_68: "f32[320, 2560]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
        addmm_6: "f32[4096, 2560]" = torch.ops.aten.addmm.default(primals_235, view_183, permute_68);  primals_235 = None
        view_184: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(addmm_6, [4, 1024, 2560]);  addmm_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_69: "f32[320, 80]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
        mm_32: "f32[4096, 80]" = torch.ops.aten.mm.default(view_183, permute_69)
        permute_70: "f32[80, 2560]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        mm_33: "f32[4096, 2560]" = torch.ops.aten.mm.default(mm_32, permute_70)
        view_188: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(mm_33, [4, 1024, 2560]);  mm_33 = None
        mul_131: "f32[4, 1024, 2560]" = torch.ops.aten.mul.Tensor(view_188, 1.0);  view_188 = None
        add_120: "f32[4, 1024, 2560]" = torch.ops.aten.add.Tensor(view_184, mul_131);  view_184 = mul_131 = None
        view_189: "f32[4096, 2560]" = torch.ops.aten.view.default(add_120, [4096, 2560]);  add_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_192: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(view_189, [4, 1024, 2560]);  view_189 = None
        split_2 = torch.ops.aten.split.Tensor(view_192, 1280, -1);  view_192 = None
        getitem_73: "f32[4, 1024, 1280]" = split_2[1]
        mul_132: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_73, 0.5)
        mul_133: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_73, 0.7071067811865476)
        erf: "f32[4, 1024, 1280]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_121: "f32[4, 1024, 1280]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_134: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_132, add_121);  mul_132 = add_121 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_74: "f32[4, 1024, 1280]" = split_2[0];  split_2 = None
        mul_135: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_74, mul_134);  mul_134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_194: "f32[4096, 1280]" = torch.ops.aten.view.default(mul_135, [4096, 1280]);  mul_135 = None
        permute_71: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
        addmm_7: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_239, view_194, permute_71);  primals_239 = None
        view_195: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_7, [4, 1024, 320]);  addmm_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_72: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
        mm_34: "f32[4096, 80]" = torch.ops.aten.mm.default(view_194, permute_72)
        permute_73: "f32[80, 320]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
        mm_35: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_34, permute_73)
        view_199: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_35, [4, 1024, 320]);  mm_35 = None
        mul_136: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_199, 1.0);  view_199 = None
        add_122: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_195, mul_136);  view_195 = mul_136 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_123: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_122, add_117);  add_122 = add_117 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_203: "f32[4096, 320]" = torch.ops.aten.view.default(add_123, [4096, 320]);  add_123 = None
        permute_74: "f32[320, 320]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
        addmm_8: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_243, view_203, permute_74);  primals_243 = None
        view_204: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_8, [4, 1024, 320]);  addmm_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_75: "f32[320, 80]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
        mm_36: "f32[4096, 80]" = torch.ops.aten.mm.default(view_203, permute_75)
        permute_76: "f32[80, 320]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
        mm_37: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_36, permute_76)
        view_208: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_37, [4, 1024, 320]);  mm_37 = None
        mul_137: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_208, 1.0);  view_208 = None
        add_124: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_204, mul_137);  view_204 = mul_137 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_214: "f32[4, 32, 32, 320]" = torch.ops.aten.view.default(add_124, [4, 32, 32, 320]);  add_124 = None
        permute_78: "f32[4, 320, 32, 32]" = torch.ops.aten.permute.default(view_214, [0, 3, 1, 2]);  view_214 = None
        clone_21: "f32[4, 320, 32, 32]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_125: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(clone_21, div_12);  clone_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_215: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_125, [4, 32, 10, 1024])
        var_mean_28 = torch.ops.aten.var_mean.correction(view_215, [2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[4, 32, 1, 1]" = var_mean_28[0]
        getitem_77: "f32[4, 32, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_126: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_28: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_28: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_215, getitem_77);  view_215 = None
        mul_138: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        view_216: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_138, [4, 320, 32, 32]);  mul_138 = None
        unsqueeze_152: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_247, 0)
        unsqueeze_153: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
        unsqueeze_154: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
        unsqueeze_155: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_246, 0)
        unsqueeze_156: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_139: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_216, unsqueeze_157);  view_216 = unsqueeze_157 = None
        add_127: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_154);  mul_139 = unsqueeze_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_25: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_127)
        mul_140: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_25);  add_127 = sigmoid_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_91: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_140, primals_248, primals_249, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_249 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_92: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_140, primals_250, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_93: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_92, primals_251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_141: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_93, 1.0);  convolution_93 = None
        add_128: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_91, mul_141);  convolution_91 = mul_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_79: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
        addmm_9: "f32[4, 320]" = torch.ops.aten.addmm.default(primals_253, mul_109, permute_79);  primals_253 = permute_79 = None
        unsqueeze_158: "f32[4, 320, 1]" = torch.ops.aten.unsqueeze.default(addmm_9, 2);  addmm_9 = None
        unsqueeze_159: "f32[4, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_129: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_128, unsqueeze_159);  add_128 = unsqueeze_159 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_217: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_129, [4, 32, 10, 1024])
        var_mean_29 = torch.ops.aten.var_mean.correction(view_217, [2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[4, 32, 1, 1]" = var_mean_29[0]
        getitem_79: "f32[4, 32, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_130: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_29: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_29: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_217, getitem_79);  view_217 = None
        mul_143: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        view_218: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_143, [4, 320, 32, 32]);  mul_143 = None
        unsqueeze_160: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_255, 0)
        unsqueeze_161: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        unsqueeze_163: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_254, 0)
        unsqueeze_164: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_144: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_218, unsqueeze_165);  view_218 = unsqueeze_165 = None
        add_131: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_162);  mul_144 = unsqueeze_162 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_27: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_131)
        mul_145: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_27);  add_131 = sigmoid_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_94: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_145, primals_256, primals_257, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_257 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_95: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_145, primals_258, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_96: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_95, primals_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_146: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_96, 1.0);  convolution_96 = None
        add_132: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_94, mul_146);  convolution_94 = mul_146 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_133: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_125, add_132);  add_132 = None
        div_15: "f32[4, 320, 32, 32]" = torch.ops.aten.div.Tensor(add_133, 1.0);  add_133 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_219: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(div_15, [4, 32, 10, 1024])
        var_mean_30 = torch.ops.aten.var_mean.correction(view_219, [2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[4, 32, 1, 1]" = var_mean_30[0]
        getitem_81: "f32[4, 32, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_134: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_30: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_30: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_219, getitem_81);  view_219 = None
        mul_147: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        view_220: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_147, [4, 320, 32, 32]);  mul_147 = None
        unsqueeze_166: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
        unsqueeze_167: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        unsqueeze_169: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_260, 0)
        unsqueeze_170: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_148: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_220, unsqueeze_171);  view_220 = unsqueeze_171 = None
        add_135: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_148, unsqueeze_168);  mul_148 = unsqueeze_168 = None
        squeeze_54: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_81, [2, 3]);  getitem_81 = None
        squeeze_55: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_30, [2, 3]);  rsqrt_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_80: "f32[4, 32, 32, 320]" = torch.ops.aten.permute.default(add_135, [0, 2, 3, 1]);  add_135 = None
        view_221: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_80, [4, 1024, 320]);  permute_80 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_81: "f32[320, 320]" = torch.ops.aten.permute.default(primals_262, [1, 0])
        expand_9: "f32[4, 1024, 320]" = torch.ops.aten.expand.default(view_221, [4, 1024, 320])
        expand_10: "f32[4, 320, 320]" = torch.ops.aten.expand.default(permute_81, [4, 320, 320]);  permute_81 = None
        bmm_4: "f32[4, 1024, 320]" = torch.ops.aten.bmm.default(expand_9, expand_10);  expand_9 = expand_10 = None
        add_136: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(bmm_4, primals_263);  bmm_4 = primals_263 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_82: "f32[320, 80]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
        clone_23: "f32[4, 1024, 320]" = torch.ops.aten.clone.default(view_221, memory_format = torch.contiguous_format);  view_221 = None
        view_225: "f32[4096, 320]" = torch.ops.aten.view.default(clone_23, [4096, 320]);  clone_23 = None
        mm_38: "f32[4096, 80]" = torch.ops.aten.mm.default(view_225, permute_82)
        permute_83: "f32[80, 320]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        mm_39: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_38, permute_83)
        view_228: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_39, [4, 1024, 320]);  mm_39 = None
        mul_149: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_228, 1.0);  view_228 = None
        add_137: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_136, mul_149);  add_136 = mul_149 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_82: "f32[4, 1024, 1]" = var_mean_31[0]
        getitem_83: "f32[4, 1024, 1]" = var_mean_31[1];  var_mean_31 = None
        add_138: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_31: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_31: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_137, getitem_83);  getitem_83 = None
        mul_150: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        mul_151: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_150, primals_266)
        add_139: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_151, primals_267);  mul_151 = primals_267 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_84: "f32[320, 320]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
        view_229: "f32[4096, 320]" = torch.ops.aten.view.default(add_139, [4096, 320]);  add_139 = None
        mm_40: "f32[4096, 320]" = torch.ops.aten.mm.default(view_229, permute_84)
        view_230: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_40, [4, 1024, 320]);  mm_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_85: "f32[320, 80]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        mm_41: "f32[4096, 80]" = torch.ops.aten.mm.default(view_229, permute_85)
        permute_86: "f32[80, 320]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
        mm_42: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_41, permute_86)
        view_234: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_42, [4, 1024, 320]);  mm_42 = None
        mul_152: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_234, 1.0);  view_234 = None
        add_140: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_230, mul_152);  view_230 = mul_152 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_87: "f32[320, 320]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        mm_43: "f32[4096, 320]" = torch.ops.aten.mm.default(view_229, permute_87)
        view_238: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_43, [4, 1024, 320]);  mm_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_88: "f32[320, 80]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
        mm_44: "f32[4096, 80]" = torch.ops.aten.mm.default(view_229, permute_88)
        permute_89: "f32[80, 320]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        mm_45: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_44, permute_89)
        view_242: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_45, [4, 1024, 320]);  mm_45 = None
        mul_153: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_242, 1.0);  view_242 = None
        add_141: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_238, mul_153);  view_238 = mul_153 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_90: "f32[320, 320]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
        mm_46: "f32[4096, 320]" = torch.ops.aten.mm.default(view_229, permute_90)
        view_246: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_46, [4, 1024, 320]);  mm_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_91: "f32[320, 80]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        mm_47: "f32[4096, 80]" = torch.ops.aten.mm.default(view_229, permute_91)
        permute_92: "f32[80, 320]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
        mm_48: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_47, permute_92)
        view_250: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_48, [4, 1024, 320]);  mm_48 = None
        mul_154: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_250, 1.0);  view_250 = None
        add_142: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_246, mul_154);  view_246 = mul_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_257: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_140, [4, -1, 5, 64]);  add_140 = None
        permute_96: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        view_259: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_141, [4, -1, 5, 64]);  add_141 = None
        permute_97: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        view_261: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_142, [4, -1, 5, 64]);  add_142 = None
        permute_98: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_96, permute_97, permute_98, None, True)
        getitem_84: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_85: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_86: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_87: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_99: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3])
        view_262: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_99, [4, -1, 320]);  permute_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_263: "f32[4096, 320]" = torch.ops.aten.view.default(view_262, [4096, 320]);  view_262 = None
        permute_100: "f32[320, 320]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
        addmm_10: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_278, view_263, permute_100);  primals_278 = None
        view_264: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_10, [4, 1024, 320]);  addmm_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_101: "f32[320, 80]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
        mm_49: "f32[4096, 80]" = torch.ops.aten.mm.default(view_263, permute_101);  view_263 = None
        permute_102: "f32[80, 320]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
        mm_50: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_49, permute_102)
        view_268: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_50, [4, 1024, 320]);  mm_50 = None
        mul_155: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_268, 1.0);  view_268 = None
        add_143: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_264, mul_155);  view_264 = mul_155 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_16: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_143, 1.0);  add_143 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_144: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_16, add_137);  div_16 = add_137 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 1024, 1]" = var_mean_32[0]
        getitem_89: "f32[4, 1024, 1]" = var_mean_32[1];  var_mean_32 = None
        add_145: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_32: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_32: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_144, getitem_89);  getitem_89 = None
        mul_156: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        mul_157: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_156, primals_281)
        add_146: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_157, primals_282);  mul_157 = primals_282 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_103: "f32[320, 320]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        view_272: "f32[4096, 320]" = torch.ops.aten.view.default(add_146, [4096, 320]);  add_146 = None
        mm_51: "f32[4096, 320]" = torch.ops.aten.mm.default(view_272, permute_103)
        view_273: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_51, [4, 1024, 320]);  mm_51 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_104: "f32[320, 80]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
        mm_52: "f32[4096, 80]" = torch.ops.aten.mm.default(view_272, permute_104)
        permute_105: "f32[80, 320]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        mm_53: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_52, permute_105)
        view_277: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_53, [4, 1024, 320]);  mm_53 = None
        mul_158: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_277, 1.0);  view_277 = None
        add_147: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_273, mul_158);  view_273 = mul_158 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_106: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
        mm_54: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_106);  permute_106 = None
        view_281: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_54, [4, 77, 320]);  mm_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_107: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        mm_55: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_107);  permute_107 = None
        permute_108: "f32[80, 320]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
        mm_56: "f32[308, 320]" = torch.ops.aten.mm.default(mm_55, permute_108)
        view_285: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_56, [4, 77, 320]);  mm_56 = None
        mul_159: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_285, 1.0);  view_285 = None
        add_148: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_281, mul_159);  view_281 = mul_159 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_109: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        mm_57: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_109);  permute_109 = None
        view_289: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_57, [4, 77, 320]);  mm_57 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_110: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
        mm_58: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_110);  permute_110 = None
        permute_111: "f32[80, 320]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        mm_59: "f32[308, 320]" = torch.ops.aten.mm.default(mm_58, permute_111)
        view_293: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_59, [4, 77, 320]);  mm_59 = None
        mul_160: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_293, 1.0);  view_293 = None
        add_149: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_289, mul_160);  view_289 = mul_160 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_300: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_147, [4, -1, 5, 64]);  add_147 = None
        permute_115: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        view_302: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_148, [4, -1, 5, 64]);  add_148 = None
        permute_116: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        view_304: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_149, [4, -1, 5, 64]);  add_149 = None
        permute_117: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_115, permute_116, permute_117, None, True)
        getitem_90: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_91: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_92: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_93: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_118: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3])
        view_305: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_118, [4, -1, 320]);  permute_118 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_306: "f32[4096, 320]" = torch.ops.aten.view.default(view_305, [4096, 320]);  view_305 = None
        permute_119: "f32[320, 320]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
        addmm_11: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_293, view_306, permute_119);  primals_293 = None
        view_307: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_11, [4, 1024, 320]);  addmm_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_120: "f32[320, 80]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
        mm_60: "f32[4096, 80]" = torch.ops.aten.mm.default(view_306, permute_120);  view_306 = None
        permute_121: "f32[80, 320]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
        mm_61: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_60, permute_121)
        view_311: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_61, [4, 1024, 320]);  mm_61 = None
        mul_161: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_311, 1.0);  view_311 = None
        add_150: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_307, mul_161);  view_307 = mul_161 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_17: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_150, 1.0);  add_150 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_151: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_17, add_144);  div_17 = add_144 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 1024, 1]" = var_mean_33[0]
        getitem_95: "f32[4, 1024, 1]" = var_mean_33[1];  var_mean_33 = None
        add_152: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_33: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_33: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_151, getitem_95);  getitem_95 = None
        mul_162: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        mul_163: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_162, primals_296)
        add_153: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_163, primals_297);  mul_163 = primals_297 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_315: "f32[4096, 320]" = torch.ops.aten.view.default(add_153, [4096, 320]);  add_153 = None
        permute_122: "f32[320, 2560]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
        addmm_12: "f32[4096, 2560]" = torch.ops.aten.addmm.default(primals_299, view_315, permute_122);  primals_299 = None
        view_316: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(addmm_12, [4, 1024, 2560]);  addmm_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_123: "f32[320, 80]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
        mm_62: "f32[4096, 80]" = torch.ops.aten.mm.default(view_315, permute_123)
        permute_124: "f32[80, 2560]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        mm_63: "f32[4096, 2560]" = torch.ops.aten.mm.default(mm_62, permute_124)
        view_320: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(mm_63, [4, 1024, 2560]);  mm_63 = None
        mul_164: "f32[4, 1024, 2560]" = torch.ops.aten.mul.Tensor(view_320, 1.0);  view_320 = None
        add_154: "f32[4, 1024, 2560]" = torch.ops.aten.add.Tensor(view_316, mul_164);  view_316 = mul_164 = None
        view_321: "f32[4096, 2560]" = torch.ops.aten.view.default(add_154, [4096, 2560]);  add_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_324: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(view_321, [4, 1024, 2560]);  view_321 = None
        split_5 = torch.ops.aten.split.Tensor(view_324, 1280, -1);  view_324 = None
        getitem_99: "f32[4, 1024, 1280]" = split_5[1]
        mul_165: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_99, 0.5)
        mul_166: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_99, 0.7071067811865476)
        erf_1: "f32[4, 1024, 1280]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_155: "f32[4, 1024, 1280]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_167: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_165, add_155);  mul_165 = add_155 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_100: "f32[4, 1024, 1280]" = split_5[0];  split_5 = None
        mul_168: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_100, mul_167);  mul_167 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_326: "f32[4096, 1280]" = torch.ops.aten.view.default(mul_168, [4096, 1280]);  mul_168 = None
        permute_125: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
        addmm_13: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_303, view_326, permute_125);  primals_303 = None
        view_327: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_13, [4, 1024, 320]);  addmm_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_126: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
        mm_64: "f32[4096, 80]" = torch.ops.aten.mm.default(view_326, permute_126)
        permute_127: "f32[80, 320]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
        mm_65: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_64, permute_127)
        view_331: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_65, [4, 1024, 320]);  mm_65 = None
        mul_169: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_331, 1.0);  view_331 = None
        add_156: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_327, mul_169);  view_327 = mul_169 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_157: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_156, add_151);  add_156 = add_151 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_335: "f32[4096, 320]" = torch.ops.aten.view.default(add_157, [4096, 320]);  add_157 = None
        permute_128: "f32[320, 320]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
        addmm_14: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_307, view_335, permute_128);  primals_307 = None
        view_336: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_14, [4, 1024, 320]);  addmm_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_129: "f32[320, 80]" = torch.ops.aten.permute.default(primals_308, [1, 0]);  primals_308 = None
        mm_66: "f32[4096, 80]" = torch.ops.aten.mm.default(view_335, permute_129)
        permute_130: "f32[80, 320]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        mm_67: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_66, permute_130)
        view_340: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_67, [4, 1024, 320]);  mm_67 = None
        mul_170: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_340, 1.0);  view_340 = None
        add_158: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_336, mul_170);  view_336 = mul_170 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_346: "f32[4, 32, 32, 320]" = torch.ops.aten.view.default(add_158, [4, 32, 32, 320]);  add_158 = None
        permute_132: "f32[4, 320, 32, 32]" = torch.ops.aten.permute.default(view_346, [0, 3, 1, 2]);  view_346 = None
        clone_27: "f32[4, 320, 32, 32]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_159: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(clone_27, div_15);  clone_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_97: "f32[4, 320, 16, 16]" = torch.ops.aten.convolution.default(add_159, primals_310, primals_311, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_311 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_98: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(add_159, primals_312, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_99: "f32[4, 320, 16, 16]" = torch.ops.aten.convolution.default(convolution_98, primals_313, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_171: "f32[4, 320, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_99, 1.0);  convolution_99 = None
        add_160: "f32[4, 320, 16, 16]" = torch.ops.aten.add.Tensor(convolution_97, mul_171);  convolution_97 = mul_171 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_347: "f32[4, 32, 10, 256]" = torch.ops.aten.view.default(add_160, [4, 32, 10, 256])
        var_mean_34 = torch.ops.aten.var_mean.correction(view_347, [2, 3], correction = 0, keepdim = True)
        getitem_102: "f32[4, 32, 1, 1]" = var_mean_34[0]
        getitem_103: "f32[4, 32, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_161: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_34: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_34: "f32[4, 32, 10, 256]" = torch.ops.aten.sub.Tensor(view_347, getitem_103);  view_347 = None
        mul_172: "f32[4, 32, 10, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        view_348: "f32[4, 320, 16, 16]" = torch.ops.aten.view.default(mul_172, [4, 320, 16, 16]);  mul_172 = None
        unsqueeze_172: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_315, 0)
        unsqueeze_173: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        unsqueeze_175: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_314, 0)
        unsqueeze_176: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_173: "f32[4, 320, 16, 16]" = torch.ops.aten.mul.Tensor(view_348, unsqueeze_177);  view_348 = unsqueeze_177 = None
        add_162: "f32[4, 320, 16, 16]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_174);  mul_173 = unsqueeze_174 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_28: "f32[4, 320, 16, 16]" = torch.ops.aten.sigmoid.default(add_162)
        mul_174: "f32[4, 320, 16, 16]" = torch.ops.aten.mul.Tensor(add_162, sigmoid_28);  add_162 = sigmoid_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_100: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_174, primals_316, primals_317, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_317 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_101: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_174, primals_318, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_102: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_101, primals_319, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_175: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_102, 1.0);  convolution_102 = None
        add_163: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_100, mul_175);  convolution_100 = mul_175 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_133: "f32[1280, 640]" = torch.ops.aten.permute.default(primals_320, [1, 0]);  primals_320 = None
        addmm_15: "f32[4, 640]" = torch.ops.aten.addmm.default(primals_321, mul_109, permute_133);  primals_321 = permute_133 = None
        unsqueeze_178: "f32[4, 640, 1]" = torch.ops.aten.unsqueeze.default(addmm_15, 2);  addmm_15 = None
        unsqueeze_179: "f32[4, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_164: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_163, unsqueeze_179);  add_163 = unsqueeze_179 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_349: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_164, [4, 32, 20, 256])
        var_mean_35 = torch.ops.aten.var_mean.correction(view_349, [2, 3], correction = 0, keepdim = True)
        getitem_104: "f32[4, 32, 1, 1]" = var_mean_35[0]
        getitem_105: "f32[4, 32, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_165: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_35: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_35: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_349, getitem_105);  view_349 = None
        mul_177: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        view_350: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_177, [4, 640, 16, 16]);  mul_177 = None
        unsqueeze_180: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_323, 0)
        unsqueeze_181: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
        unsqueeze_182: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
        unsqueeze_183: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_322, 0)
        unsqueeze_184: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        unsqueeze_185: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
        mul_178: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_350, unsqueeze_185);  view_350 = unsqueeze_185 = None
        add_166: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_182);  mul_178 = unsqueeze_182 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_30: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
        mul_179: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_30);  add_166 = sigmoid_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_103: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_179, primals_324, primals_325, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_325 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_104: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_179, primals_326, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_105: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_104, primals_327, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_180: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_105, 1.0);  convolution_105 = None
        add_167: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_103, mul_180);  convolution_103 = mul_180 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_106: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(add_160, primals_328, primals_329, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_329 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_107: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(add_160, primals_330, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_108: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_107, primals_331, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_181: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_108, 1.0);  convolution_108 = None
        add_168: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_106, mul_181);  convolution_106 = mul_181 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_169: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_168, add_167);  add_168 = add_167 = None
        div_18: "f32[4, 640, 16, 16]" = torch.ops.aten.div.Tensor(add_169, 1.0);  add_169 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_351: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(div_18, [4, 32, 20, 256])
        var_mean_36 = torch.ops.aten.var_mean.correction(view_351, [2, 3], correction = 0, keepdim = True)
        getitem_106: "f32[4, 32, 1, 1]" = var_mean_36[0]
        getitem_107: "f32[4, 32, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_170: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
        rsqrt_36: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_36: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_351, getitem_107);  view_351 = None
        mul_182: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        view_352: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_182, [4, 640, 16, 16]);  mul_182 = None
        unsqueeze_186: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_333, 0);  primals_333 = None
        unsqueeze_187: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
        unsqueeze_188: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
        unsqueeze_189: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_332, 0)
        unsqueeze_190: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
        unsqueeze_191: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
        mul_183: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_352, unsqueeze_191);  view_352 = unsqueeze_191 = None
        add_171: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_188);  mul_183 = unsqueeze_188 = None
        squeeze_60: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_107, [2, 3]);  getitem_107 = None
        squeeze_61: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_36, [2, 3]);  rsqrt_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_134: "f32[4, 16, 16, 640]" = torch.ops.aten.permute.default(add_171, [0, 2, 3, 1]);  add_171 = None
        view_353: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_134, [4, 256, 640]);  permute_134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_135: "f32[640, 640]" = torch.ops.aten.permute.default(primals_334, [1, 0])
        expand_11: "f32[4, 256, 640]" = torch.ops.aten.expand.default(view_353, [4, 256, 640])
        expand_12: "f32[4, 640, 640]" = torch.ops.aten.expand.default(permute_135, [4, 640, 640]);  permute_135 = None
        bmm_5: "f32[4, 256, 640]" = torch.ops.aten.bmm.default(expand_11, expand_12);  expand_11 = expand_12 = None
        add_172: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(bmm_5, primals_335);  bmm_5 = primals_335 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_136: "f32[640, 80]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
        clone_29: "f32[4, 256, 640]" = torch.ops.aten.clone.default(view_353, memory_format = torch.contiguous_format);  view_353 = None
        view_357: "f32[1024, 640]" = torch.ops.aten.view.default(clone_29, [1024, 640]);  clone_29 = None
        mm_68: "f32[1024, 80]" = torch.ops.aten.mm.default(view_357, permute_136)
        permute_137: "f32[80, 640]" = torch.ops.aten.permute.default(primals_337, [1, 0]);  primals_337 = None
        mm_69: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_68, permute_137)
        view_360: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_69, [4, 256, 640]);  mm_69 = None
        mul_184: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_360, 1.0);  view_360 = None
        add_173: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_172, mul_184);  add_172 = mul_184 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_108: "f32[4, 256, 1]" = var_mean_37[0]
        getitem_109: "f32[4, 256, 1]" = var_mean_37[1];  var_mean_37 = None
        add_174: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_37: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_37: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_173, getitem_109);  getitem_109 = None
        mul_185: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        mul_186: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_185, primals_338)
        add_175: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_186, primals_339);  mul_186 = primals_339 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_138: "f32[640, 640]" = torch.ops.aten.permute.default(primals_340, [1, 0]);  primals_340 = None
        view_361: "f32[1024, 640]" = torch.ops.aten.view.default(add_175, [1024, 640]);  add_175 = None
        mm_70: "f32[1024, 640]" = torch.ops.aten.mm.default(view_361, permute_138)
        view_362: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_70, [4, 256, 640]);  mm_70 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_139: "f32[640, 80]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
        mm_71: "f32[1024, 80]" = torch.ops.aten.mm.default(view_361, permute_139)
        permute_140: "f32[80, 640]" = torch.ops.aten.permute.default(primals_342, [1, 0]);  primals_342 = None
        mm_72: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_71, permute_140)
        view_366: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_72, [4, 256, 640]);  mm_72 = None
        mul_187: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_366, 1.0);  view_366 = None
        add_176: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_362, mul_187);  view_362 = mul_187 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_141: "f32[640, 640]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
        mm_73: "f32[1024, 640]" = torch.ops.aten.mm.default(view_361, permute_141)
        view_370: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_73, [4, 256, 640]);  mm_73 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_142: "f32[640, 80]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
        mm_74: "f32[1024, 80]" = torch.ops.aten.mm.default(view_361, permute_142)
        permute_143: "f32[80, 640]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
        mm_75: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_74, permute_143)
        view_374: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_75, [4, 256, 640]);  mm_75 = None
        mul_188: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_374, 1.0);  view_374 = None
        add_177: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_370, mul_188);  view_370 = mul_188 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_144: "f32[640, 640]" = torch.ops.aten.permute.default(primals_346, [1, 0]);  primals_346 = None
        mm_76: "f32[1024, 640]" = torch.ops.aten.mm.default(view_361, permute_144)
        view_378: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_76, [4, 256, 640]);  mm_76 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_145: "f32[640, 80]" = torch.ops.aten.permute.default(primals_347, [1, 0]);  primals_347 = None
        mm_77: "f32[1024, 80]" = torch.ops.aten.mm.default(view_361, permute_145)
        permute_146: "f32[80, 640]" = torch.ops.aten.permute.default(primals_348, [1, 0]);  primals_348 = None
        mm_78: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_77, permute_146)
        view_382: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_78, [4, 256, 640]);  mm_78 = None
        mul_189: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_382, 1.0);  view_382 = None
        add_178: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_378, mul_189);  view_378 = mul_189 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_389: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_176, [4, -1, 10, 64]);  add_176 = None
        permute_150: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
        view_391: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_177, [4, -1, 10, 64]);  add_177 = None
        permute_151: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
        view_393: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_178, [4, -1, 10, 64]);  add_178 = None
        permute_152: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_150, permute_151, permute_152, None, True)
        getitem_110: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_111: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_112: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_113: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_153: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3])
        view_394: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_153, [4, -1, 640]);  permute_153 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_395: "f32[1024, 640]" = torch.ops.aten.view.default(view_394, [1024, 640]);  view_394 = None
        permute_154: "f32[640, 640]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
        addmm_16: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_350, view_395, permute_154);  primals_350 = None
        view_396: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_16, [4, 256, 640]);  addmm_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_155: "f32[640, 80]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
        mm_79: "f32[1024, 80]" = torch.ops.aten.mm.default(view_395, permute_155);  view_395 = None
        permute_156: "f32[80, 640]" = torch.ops.aten.permute.default(primals_352, [1, 0]);  primals_352 = None
        mm_80: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_79, permute_156)
        view_400: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_80, [4, 256, 640]);  mm_80 = None
        mul_190: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_400, 1.0);  view_400 = None
        add_179: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_396, mul_190);  view_396 = mul_190 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_19: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_179, 1.0);  add_179 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_180: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_19, add_173);  div_19 = add_173 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_114: "f32[4, 256, 1]" = var_mean_38[0]
        getitem_115: "f32[4, 256, 1]" = var_mean_38[1];  var_mean_38 = None
        add_181: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_38: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_38: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_180, getitem_115);  getitem_115 = None
        mul_191: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        mul_192: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_191, primals_353)
        add_182: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_192, primals_354);  mul_192 = primals_354 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_157: "f32[640, 640]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
        view_404: "f32[1024, 640]" = torch.ops.aten.view.default(add_182, [1024, 640]);  add_182 = None
        mm_81: "f32[1024, 640]" = torch.ops.aten.mm.default(view_404, permute_157)
        view_405: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_81, [4, 256, 640]);  mm_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_158: "f32[640, 80]" = torch.ops.aten.permute.default(primals_356, [1, 0]);  primals_356 = None
        mm_82: "f32[1024, 80]" = torch.ops.aten.mm.default(view_404, permute_158)
        permute_159: "f32[80, 640]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
        mm_83: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_82, permute_159)
        view_409: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_83, [4, 256, 640]);  mm_83 = None
        mul_193: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_409, 1.0);  view_409 = None
        add_183: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_405, mul_193);  view_405 = mul_193 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_160: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
        mm_84: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_160);  permute_160 = None
        view_413: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_84, [4, 77, 640]);  mm_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_161: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
        mm_85: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_161);  permute_161 = None
        permute_162: "f32[80, 640]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None
        mm_86: "f32[308, 640]" = torch.ops.aten.mm.default(mm_85, permute_162)
        view_417: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_86, [4, 77, 640]);  mm_86 = None
        mul_194: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_417, 1.0);  view_417 = None
        add_184: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_413, mul_194);  view_413 = mul_194 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_163: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
        mm_87: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_163);  permute_163 = None
        view_421: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_87, [4, 77, 640]);  mm_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_164: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_362, [1, 0]);  primals_362 = None
        mm_88: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_164);  permute_164 = None
        permute_165: "f32[80, 640]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
        mm_89: "f32[308, 640]" = torch.ops.aten.mm.default(mm_88, permute_165)
        view_425: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_89, [4, 77, 640]);  mm_89 = None
        mul_195: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_425, 1.0);  view_425 = None
        add_185: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_421, mul_195);  view_421 = mul_195 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_432: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_183, [4, -1, 10, 64]);  add_183 = None
        permute_169: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        view_434: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_184, [4, -1, 10, 64]);  add_184 = None
        permute_170: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        view_436: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_185, [4, -1, 10, 64]);  add_185 = None
        permute_171: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_169, permute_170, permute_171, None, True)
        getitem_116: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_6[0]
        getitem_117: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_6[1]
        getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
        getitem_119: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_172: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_116, [0, 2, 1, 3])
        view_437: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_172, [4, -1, 640]);  permute_172 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_438: "f32[1024, 640]" = torch.ops.aten.view.default(view_437, [1024, 640]);  view_437 = None
        permute_173: "f32[640, 640]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None
        addmm_17: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_365, view_438, permute_173);  primals_365 = None
        view_439: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_17, [4, 256, 640]);  addmm_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_174: "f32[640, 80]" = torch.ops.aten.permute.default(primals_366, [1, 0]);  primals_366 = None
        mm_90: "f32[1024, 80]" = torch.ops.aten.mm.default(view_438, permute_174);  view_438 = None
        permute_175: "f32[80, 640]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
        mm_91: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_90, permute_175)
        view_443: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_91, [4, 256, 640]);  mm_91 = None
        mul_196: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_443, 1.0);  view_443 = None
        add_186: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_439, mul_196);  view_439 = mul_196 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_20: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_186, 1.0);  add_186 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_187: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_20, add_180);  div_20 = add_180 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
        getitem_120: "f32[4, 256, 1]" = var_mean_39[0]
        getitem_121: "f32[4, 256, 1]" = var_mean_39[1];  var_mean_39 = None
        add_188: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_39: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_39: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_187, getitem_121);  getitem_121 = None
        mul_197: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        mul_198: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_197, primals_368)
        add_189: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_198, primals_369);  mul_198 = primals_369 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_447: "f32[1024, 640]" = torch.ops.aten.view.default(add_189, [1024, 640]);  add_189 = None
        permute_176: "f32[640, 5120]" = torch.ops.aten.permute.default(primals_370, [1, 0]);  primals_370 = None
        addmm_18: "f32[1024, 5120]" = torch.ops.aten.addmm.default(primals_371, view_447, permute_176);  primals_371 = None
        view_448: "f32[4, 256, 5120]" = torch.ops.aten.view.default(addmm_18, [4, 256, 5120]);  addmm_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_177: "f32[640, 80]" = torch.ops.aten.permute.default(primals_372, [1, 0]);  primals_372 = None
        mm_92: "f32[1024, 80]" = torch.ops.aten.mm.default(view_447, permute_177)
        permute_178: "f32[80, 5120]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
        mm_93: "f32[1024, 5120]" = torch.ops.aten.mm.default(mm_92, permute_178)
        view_452: "f32[4, 256, 5120]" = torch.ops.aten.view.default(mm_93, [4, 256, 5120]);  mm_93 = None
        mul_199: "f32[4, 256, 5120]" = torch.ops.aten.mul.Tensor(view_452, 1.0);  view_452 = None
        add_190: "f32[4, 256, 5120]" = torch.ops.aten.add.Tensor(view_448, mul_199);  view_448 = mul_199 = None
        view_453: "f32[1024, 5120]" = torch.ops.aten.view.default(add_190, [1024, 5120]);  add_190 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_456: "f32[4, 256, 5120]" = torch.ops.aten.view.default(view_453, [4, 256, 5120]);  view_453 = None
        split_8 = torch.ops.aten.split.Tensor(view_456, 2560, -1);  view_456 = None
        getitem_125: "f32[4, 256, 2560]" = split_8[1]
        mul_200: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_125, 0.5)
        mul_201: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_125, 0.7071067811865476)
        erf_2: "f32[4, 256, 2560]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_191: "f32[4, 256, 2560]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_202: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_200, add_191);  mul_200 = add_191 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_126: "f32[4, 256, 2560]" = split_8[0];  split_8 = None
        mul_203: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_126, mul_202);  mul_202 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_458: "f32[1024, 2560]" = torch.ops.aten.view.default(mul_203, [1024, 2560]);  mul_203 = None
        permute_179: "f32[2560, 640]" = torch.ops.aten.permute.default(primals_374, [1, 0]);  primals_374 = None
        addmm_19: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_375, view_458, permute_179);  primals_375 = None
        view_459: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_19, [4, 256, 640]);  addmm_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_180: "f32[2560, 80]" = torch.ops.aten.permute.default(primals_376, [1, 0]);  primals_376 = None
        mm_94: "f32[1024, 80]" = torch.ops.aten.mm.default(view_458, permute_180)
        permute_181: "f32[80, 640]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
        mm_95: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_94, permute_181)
        view_463: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_95, [4, 256, 640]);  mm_95 = None
        mul_204: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_463, 1.0);  view_463 = None
        add_192: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_459, mul_204);  view_459 = mul_204 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_193: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_192, add_187);  add_192 = add_187 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_467: "f32[1024, 640]" = torch.ops.aten.view.default(add_193, [1024, 640]);  add_193 = None
        permute_182: "f32[640, 640]" = torch.ops.aten.permute.default(primals_378, [1, 0]);  primals_378 = None
        addmm_20: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_379, view_467, permute_182);  primals_379 = None
        view_468: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_20, [4, 256, 640]);  addmm_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_183: "f32[640, 80]" = torch.ops.aten.permute.default(primals_380, [1, 0]);  primals_380 = None
        mm_96: "f32[1024, 80]" = torch.ops.aten.mm.default(view_467, permute_183)
        permute_184: "f32[80, 640]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
        mm_97: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_96, permute_184)
        view_472: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_97, [4, 256, 640]);  mm_97 = None
        mul_205: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_472, 1.0);  view_472 = None
        add_194: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_468, mul_205);  view_468 = mul_205 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_478: "f32[4, 16, 16, 640]" = torch.ops.aten.view.default(add_194, [4, 16, 16, 640]);  add_194 = None
        permute_186: "f32[4, 640, 16, 16]" = torch.ops.aten.permute.default(view_478, [0, 3, 1, 2]);  view_478 = None
        clone_33: "f32[4, 640, 16, 16]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_195: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(clone_33, div_18);  clone_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_479: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_195, [4, 32, 20, 256])
        var_mean_40 = torch.ops.aten.var_mean.correction(view_479, [2, 3], correction = 0, keepdim = True)
        getitem_128: "f32[4, 32, 1, 1]" = var_mean_40[0]
        getitem_129: "f32[4, 32, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_196: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_40: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_40: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_479, getitem_129);  view_479 = None
        mul_206: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        view_480: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_206, [4, 640, 16, 16]);  mul_206 = None
        unsqueeze_192: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_383, 0)
        unsqueeze_193: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
        unsqueeze_194: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
        unsqueeze_195: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_382, 0)
        unsqueeze_196: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        unsqueeze_197: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
        mul_207: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_480, unsqueeze_197);  view_480 = unsqueeze_197 = None
        add_197: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_194);  mul_207 = unsqueeze_194 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_31: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_197)
        mul_208: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_197, sigmoid_31);  add_197 = sigmoid_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_109: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_208, primals_384, primals_385, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_385 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_110: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_208, primals_386, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_111: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_110, primals_387, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_209: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_111, 1.0);  convolution_111 = None
        add_198: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_109, mul_209);  convolution_109 = mul_209 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_187: "f32[1280, 640]" = torch.ops.aten.permute.default(primals_388, [1, 0]);  primals_388 = None
        addmm_21: "f32[4, 640]" = torch.ops.aten.addmm.default(primals_389, mul_109, permute_187);  primals_389 = permute_187 = None
        unsqueeze_198: "f32[4, 640, 1]" = torch.ops.aten.unsqueeze.default(addmm_21, 2);  addmm_21 = None
        unsqueeze_199: "f32[4, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_199: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_198, unsqueeze_199);  add_198 = unsqueeze_199 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_481: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_199, [4, 32, 20, 256])
        var_mean_41 = torch.ops.aten.var_mean.correction(view_481, [2, 3], correction = 0, keepdim = True)
        getitem_130: "f32[4, 32, 1, 1]" = var_mean_41[0]
        getitem_131: "f32[4, 32, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_200: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_41: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_41: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_481, getitem_131);  view_481 = None
        mul_211: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        view_482: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_211, [4, 640, 16, 16]);  mul_211 = None
        unsqueeze_200: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_391, 0)
        unsqueeze_201: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
        unsqueeze_202: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
        unsqueeze_203: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_390, 0)
        unsqueeze_204: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_212: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_482, unsqueeze_205);  view_482 = unsqueeze_205 = None
        add_201: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_202);  mul_212 = unsqueeze_202 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_33: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_201)
        mul_213: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_201, sigmoid_33);  add_201 = sigmoid_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_112: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_213, primals_392, primals_393, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_393 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_113: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_213, primals_394, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_114: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_113, primals_395, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_214: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_114, 1.0);  convolution_114 = None
        add_202: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_112, mul_214);  convolution_112 = mul_214 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_203: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_195, add_202);  add_202 = None
        div_21: "f32[4, 640, 16, 16]" = torch.ops.aten.div.Tensor(add_203, 1.0);  add_203 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_483: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(div_21, [4, 32, 20, 256])
        var_mean_42 = torch.ops.aten.var_mean.correction(view_483, [2, 3], correction = 0, keepdim = True)
        getitem_132: "f32[4, 32, 1, 1]" = var_mean_42[0]
        getitem_133: "f32[4, 32, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_204: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_42: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_42: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_483, getitem_133);  view_483 = None
        mul_215: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        view_484: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_215, [4, 640, 16, 16]);  mul_215 = None
        unsqueeze_206: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_397, 0);  primals_397 = None
        unsqueeze_207: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        unsqueeze_209: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_396, 0)
        unsqueeze_210: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_216: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_484, unsqueeze_211);  view_484 = unsqueeze_211 = None
        add_205: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_208);  mul_216 = unsqueeze_208 = None
        squeeze_66: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_133, [2, 3]);  getitem_133 = None
        squeeze_67: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_42, [2, 3]);  rsqrt_42 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_188: "f32[4, 16, 16, 640]" = torch.ops.aten.permute.default(add_205, [0, 2, 3, 1]);  add_205 = None
        view_485: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_188, [4, 256, 640]);  permute_188 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_189: "f32[640, 640]" = torch.ops.aten.permute.default(primals_398, [1, 0])
        expand_13: "f32[4, 256, 640]" = torch.ops.aten.expand.default(view_485, [4, 256, 640])
        expand_14: "f32[4, 640, 640]" = torch.ops.aten.expand.default(permute_189, [4, 640, 640]);  permute_189 = None
        bmm_6: "f32[4, 256, 640]" = torch.ops.aten.bmm.default(expand_13, expand_14);  expand_13 = expand_14 = None
        add_206: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(bmm_6, primals_399);  bmm_6 = primals_399 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_190: "f32[640, 80]" = torch.ops.aten.permute.default(primals_400, [1, 0]);  primals_400 = None
        clone_35: "f32[4, 256, 640]" = torch.ops.aten.clone.default(view_485, memory_format = torch.contiguous_format);  view_485 = None
        view_489: "f32[1024, 640]" = torch.ops.aten.view.default(clone_35, [1024, 640]);  clone_35 = None
        mm_98: "f32[1024, 80]" = torch.ops.aten.mm.default(view_489, permute_190)
        permute_191: "f32[80, 640]" = torch.ops.aten.permute.default(primals_401, [1, 0]);  primals_401 = None
        mm_99: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_98, permute_191)
        view_492: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_99, [4, 256, 640]);  mm_99 = None
        mul_217: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_492, 1.0);  view_492 = None
        add_207: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_206, mul_217);  add_206 = mul_217 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
        getitem_134: "f32[4, 256, 1]" = var_mean_43[0]
        getitem_135: "f32[4, 256, 1]" = var_mean_43[1];  var_mean_43 = None
        add_208: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_43: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_43: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_207, getitem_135);  getitem_135 = None
        mul_218: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        mul_219: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_218, primals_402)
        add_209: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_219, primals_403);  mul_219 = primals_403 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_192: "f32[640, 640]" = torch.ops.aten.permute.default(primals_404, [1, 0]);  primals_404 = None
        view_493: "f32[1024, 640]" = torch.ops.aten.view.default(add_209, [1024, 640]);  add_209 = None
        mm_100: "f32[1024, 640]" = torch.ops.aten.mm.default(view_493, permute_192)
        view_494: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_100, [4, 256, 640]);  mm_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_193: "f32[640, 80]" = torch.ops.aten.permute.default(primals_405, [1, 0]);  primals_405 = None
        mm_101: "f32[1024, 80]" = torch.ops.aten.mm.default(view_493, permute_193)
        permute_194: "f32[80, 640]" = torch.ops.aten.permute.default(primals_406, [1, 0]);  primals_406 = None
        mm_102: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_101, permute_194)
        view_498: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_102, [4, 256, 640]);  mm_102 = None
        mul_220: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_498, 1.0);  view_498 = None
        add_210: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_494, mul_220);  view_494 = mul_220 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_195: "f32[640, 640]" = torch.ops.aten.permute.default(primals_407, [1, 0]);  primals_407 = None
        mm_103: "f32[1024, 640]" = torch.ops.aten.mm.default(view_493, permute_195)
        view_502: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_103, [4, 256, 640]);  mm_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_196: "f32[640, 80]" = torch.ops.aten.permute.default(primals_408, [1, 0]);  primals_408 = None
        mm_104: "f32[1024, 80]" = torch.ops.aten.mm.default(view_493, permute_196)
        permute_197: "f32[80, 640]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
        mm_105: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_104, permute_197)
        view_506: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_105, [4, 256, 640]);  mm_105 = None
        mul_221: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_506, 1.0);  view_506 = None
        add_211: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_502, mul_221);  view_502 = mul_221 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_198: "f32[640, 640]" = torch.ops.aten.permute.default(primals_410, [1, 0]);  primals_410 = None
        mm_106: "f32[1024, 640]" = torch.ops.aten.mm.default(view_493, permute_198)
        view_510: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_106, [4, 256, 640]);  mm_106 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_199: "f32[640, 80]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
        mm_107: "f32[1024, 80]" = torch.ops.aten.mm.default(view_493, permute_199)
        permute_200: "f32[80, 640]" = torch.ops.aten.permute.default(primals_412, [1, 0]);  primals_412 = None
        mm_108: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_107, permute_200)
        view_514: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_108, [4, 256, 640]);  mm_108 = None
        mul_222: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_514, 1.0);  view_514 = None
        add_212: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_510, mul_222);  view_510 = mul_222 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_521: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_210, [4, -1, 10, 64]);  add_210 = None
        permute_204: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
        view_523: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_211, [4, -1, 10, 64]);  add_211 = None
        permute_205: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
        view_525: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_212, [4, -1, 10, 64]);  add_212 = None
        permute_206: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_204, permute_205, permute_206, None, True)
        getitem_136: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_7[0]
        getitem_137: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_7[1]
        getitem_138: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
        getitem_139: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_207: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3])
        view_526: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_207, [4, -1, 640]);  permute_207 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_527: "f32[1024, 640]" = torch.ops.aten.view.default(view_526, [1024, 640]);  view_526 = None
        permute_208: "f32[640, 640]" = torch.ops.aten.permute.default(primals_413, [1, 0]);  primals_413 = None
        addmm_22: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_414, view_527, permute_208);  primals_414 = None
        view_528: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_22, [4, 256, 640]);  addmm_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_209: "f32[640, 80]" = torch.ops.aten.permute.default(primals_415, [1, 0]);  primals_415 = None
        mm_109: "f32[1024, 80]" = torch.ops.aten.mm.default(view_527, permute_209);  view_527 = None
        permute_210: "f32[80, 640]" = torch.ops.aten.permute.default(primals_416, [1, 0]);  primals_416 = None
        mm_110: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_109, permute_210)
        view_532: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_110, [4, 256, 640]);  mm_110 = None
        mul_223: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_532, 1.0);  view_532 = None
        add_213: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_528, mul_223);  view_528 = mul_223 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_22: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_213, 1.0);  add_213 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_214: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_22, add_207);  div_22 = add_207 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_214, [2], correction = 0, keepdim = True)
        getitem_140: "f32[4, 256, 1]" = var_mean_44[0]
        getitem_141: "f32[4, 256, 1]" = var_mean_44[1];  var_mean_44 = None
        add_215: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_44: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_44: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_214, getitem_141);  getitem_141 = None
        mul_224: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        mul_225: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_224, primals_417)
        add_216: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_225, primals_418);  mul_225 = primals_418 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_211: "f32[640, 640]" = torch.ops.aten.permute.default(primals_419, [1, 0]);  primals_419 = None
        view_536: "f32[1024, 640]" = torch.ops.aten.view.default(add_216, [1024, 640]);  add_216 = None
        mm_111: "f32[1024, 640]" = torch.ops.aten.mm.default(view_536, permute_211)
        view_537: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_111, [4, 256, 640]);  mm_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_212: "f32[640, 80]" = torch.ops.aten.permute.default(primals_420, [1, 0]);  primals_420 = None
        mm_112: "f32[1024, 80]" = torch.ops.aten.mm.default(view_536, permute_212)
        permute_213: "f32[80, 640]" = torch.ops.aten.permute.default(primals_421, [1, 0]);  primals_421 = None
        mm_113: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_112, permute_213)
        view_541: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_113, [4, 256, 640]);  mm_113 = None
        mul_226: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_541, 1.0);  view_541 = None
        add_217: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_537, mul_226);  view_537 = mul_226 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_214: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_422, [1, 0]);  primals_422 = None
        mm_114: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_214);  permute_214 = None
        view_545: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_114, [4, 77, 640]);  mm_114 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_215: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
        mm_115: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_215);  permute_215 = None
        permute_216: "f32[80, 640]" = torch.ops.aten.permute.default(primals_424, [1, 0]);  primals_424 = None
        mm_116: "f32[308, 640]" = torch.ops.aten.mm.default(mm_115, permute_216)
        view_549: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_116, [4, 77, 640]);  mm_116 = None
        mul_227: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_549, 1.0);  view_549 = None
        add_218: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_545, mul_227);  view_545 = mul_227 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_217: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_425, [1, 0]);  primals_425 = None
        mm_117: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_217);  permute_217 = None
        view_553: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_117, [4, 77, 640]);  mm_117 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_218: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_426, [1, 0]);  primals_426 = None
        mm_118: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_218);  permute_218 = None
        permute_219: "f32[80, 640]" = torch.ops.aten.permute.default(primals_427, [1, 0]);  primals_427 = None
        mm_119: "f32[308, 640]" = torch.ops.aten.mm.default(mm_118, permute_219)
        view_557: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_119, [4, 77, 640]);  mm_119 = None
        mul_228: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_557, 1.0);  view_557 = None
        add_219: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_553, mul_228);  view_553 = mul_228 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_564: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_217, [4, -1, 10, 64]);  add_217 = None
        permute_223: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None
        view_566: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_218, [4, -1, 10, 64]);  add_218 = None
        permute_224: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
        view_568: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_219, [4, -1, 10, 64]);  add_219 = None
        permute_225: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_223, permute_224, permute_225, None, True)
        getitem_142: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_8[0]
        getitem_143: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_8[1]
        getitem_144: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
        getitem_145: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_226: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3])
        view_569: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_226, [4, -1, 640]);  permute_226 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_570: "f32[1024, 640]" = torch.ops.aten.view.default(view_569, [1024, 640]);  view_569 = None
        permute_227: "f32[640, 640]" = torch.ops.aten.permute.default(primals_428, [1, 0]);  primals_428 = None
        addmm_23: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_429, view_570, permute_227);  primals_429 = None
        view_571: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_23, [4, 256, 640]);  addmm_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_228: "f32[640, 80]" = torch.ops.aten.permute.default(primals_430, [1, 0]);  primals_430 = None
        mm_120: "f32[1024, 80]" = torch.ops.aten.mm.default(view_570, permute_228);  view_570 = None
        permute_229: "f32[80, 640]" = torch.ops.aten.permute.default(primals_431, [1, 0]);  primals_431 = None
        mm_121: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_120, permute_229)
        view_575: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_121, [4, 256, 640]);  mm_121 = None
        mul_229: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_575, 1.0);  view_575 = None
        add_220: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_571, mul_229);  view_571 = mul_229 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_23: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_220, 1.0);  add_220 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_221: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_23, add_214);  div_23 = add_214 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_221, [2], correction = 0, keepdim = True)
        getitem_146: "f32[4, 256, 1]" = var_mean_45[0]
        getitem_147: "f32[4, 256, 1]" = var_mean_45[1];  var_mean_45 = None
        add_222: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_45: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_45: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_221, getitem_147);  getitem_147 = None
        mul_230: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        mul_231: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_230, primals_432)
        add_223: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_231, primals_433);  mul_231 = primals_433 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_579: "f32[1024, 640]" = torch.ops.aten.view.default(add_223, [1024, 640]);  add_223 = None
        permute_230: "f32[640, 5120]" = torch.ops.aten.permute.default(primals_434, [1, 0]);  primals_434 = None
        addmm_24: "f32[1024, 5120]" = torch.ops.aten.addmm.default(primals_435, view_579, permute_230);  primals_435 = None
        view_580: "f32[4, 256, 5120]" = torch.ops.aten.view.default(addmm_24, [4, 256, 5120]);  addmm_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_231: "f32[640, 80]" = torch.ops.aten.permute.default(primals_436, [1, 0]);  primals_436 = None
        mm_122: "f32[1024, 80]" = torch.ops.aten.mm.default(view_579, permute_231)
        permute_232: "f32[80, 5120]" = torch.ops.aten.permute.default(primals_437, [1, 0]);  primals_437 = None
        mm_123: "f32[1024, 5120]" = torch.ops.aten.mm.default(mm_122, permute_232)
        view_584: "f32[4, 256, 5120]" = torch.ops.aten.view.default(mm_123, [4, 256, 5120]);  mm_123 = None
        mul_232: "f32[4, 256, 5120]" = torch.ops.aten.mul.Tensor(view_584, 1.0);  view_584 = None
        add_224: "f32[4, 256, 5120]" = torch.ops.aten.add.Tensor(view_580, mul_232);  view_580 = mul_232 = None
        view_585: "f32[1024, 5120]" = torch.ops.aten.view.default(add_224, [1024, 5120]);  add_224 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_588: "f32[4, 256, 5120]" = torch.ops.aten.view.default(view_585, [4, 256, 5120]);  view_585 = None
        split_11 = torch.ops.aten.split.Tensor(view_588, 2560, -1);  view_588 = None
        getitem_151: "f32[4, 256, 2560]" = split_11[1]
        mul_233: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_151, 0.5)
        mul_234: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_151, 0.7071067811865476)
        erf_3: "f32[4, 256, 2560]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
        add_225: "f32[4, 256, 2560]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_235: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_233, add_225);  mul_233 = add_225 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_152: "f32[4, 256, 2560]" = split_11[0];  split_11 = None
        mul_236: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_152, mul_235);  mul_235 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_590: "f32[1024, 2560]" = torch.ops.aten.view.default(mul_236, [1024, 2560]);  mul_236 = None
        permute_233: "f32[2560, 640]" = torch.ops.aten.permute.default(primals_438, [1, 0]);  primals_438 = None
        addmm_25: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_439, view_590, permute_233);  primals_439 = None
        view_591: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_25, [4, 256, 640]);  addmm_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_234: "f32[2560, 80]" = torch.ops.aten.permute.default(primals_440, [1, 0]);  primals_440 = None
        mm_124: "f32[1024, 80]" = torch.ops.aten.mm.default(view_590, permute_234)
        permute_235: "f32[80, 640]" = torch.ops.aten.permute.default(primals_441, [1, 0]);  primals_441 = None
        mm_125: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_124, permute_235)
        view_595: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_125, [4, 256, 640]);  mm_125 = None
        mul_237: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_595, 1.0);  view_595 = None
        add_226: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_591, mul_237);  view_591 = mul_237 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_227: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_226, add_221);  add_226 = add_221 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_599: "f32[1024, 640]" = torch.ops.aten.view.default(add_227, [1024, 640]);  add_227 = None
        permute_236: "f32[640, 640]" = torch.ops.aten.permute.default(primals_442, [1, 0]);  primals_442 = None
        addmm_26: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_443, view_599, permute_236);  primals_443 = None
        view_600: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_26, [4, 256, 640]);  addmm_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_237: "f32[640, 80]" = torch.ops.aten.permute.default(primals_444, [1, 0]);  primals_444 = None
        mm_126: "f32[1024, 80]" = torch.ops.aten.mm.default(view_599, permute_237)
        permute_238: "f32[80, 640]" = torch.ops.aten.permute.default(primals_445, [1, 0]);  primals_445 = None
        mm_127: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_126, permute_238)
        view_604: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_127, [4, 256, 640]);  mm_127 = None
        mul_238: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_604, 1.0);  view_604 = None
        add_228: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_600, mul_238);  view_600 = mul_238 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_610: "f32[4, 16, 16, 640]" = torch.ops.aten.view.default(add_228, [4, 16, 16, 640]);  add_228 = None
        permute_240: "f32[4, 640, 16, 16]" = torch.ops.aten.permute.default(view_610, [0, 3, 1, 2]);  view_610 = None
        clone_39: "f32[4, 640, 16, 16]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_229: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(clone_39, div_21);  clone_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_115: "f32[4, 640, 8, 8]" = torch.ops.aten.convolution.default(add_229, primals_446, primals_447, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_447 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_116: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(add_229, primals_448, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_117: "f32[4, 640, 8, 8]" = torch.ops.aten.convolution.default(convolution_116, primals_449, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_239: "f32[4, 640, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_117, 1.0);  convolution_117 = None
        add_230: "f32[4, 640, 8, 8]" = torch.ops.aten.add.Tensor(convolution_115, mul_239);  convolution_115 = mul_239 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_611: "f32[4, 32, 20, 64]" = torch.ops.aten.view.default(add_230, [4, 32, 20, 64])
        var_mean_46 = torch.ops.aten.var_mean.correction(view_611, [2, 3], correction = 0, keepdim = True)
        getitem_154: "f32[4, 32, 1, 1]" = var_mean_46[0]
        getitem_155: "f32[4, 32, 1, 1]" = var_mean_46[1];  var_mean_46 = None
        add_231: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_46: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_46: "f32[4, 32, 20, 64]" = torch.ops.aten.sub.Tensor(view_611, getitem_155);  view_611 = None
        mul_240: "f32[4, 32, 20, 64]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        view_612: "f32[4, 640, 8, 8]" = torch.ops.aten.view.default(mul_240, [4, 640, 8, 8]);  mul_240 = None
        unsqueeze_212: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_451, 0)
        unsqueeze_213: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_450, 0)
        unsqueeze_216: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_241: "f32[4, 640, 8, 8]" = torch.ops.aten.mul.Tensor(view_612, unsqueeze_217);  view_612 = unsqueeze_217 = None
        add_232: "f32[4, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_214);  mul_241 = unsqueeze_214 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_34: "f32[4, 640, 8, 8]" = torch.ops.aten.sigmoid.default(add_232)
        mul_242: "f32[4, 640, 8, 8]" = torch.ops.aten.mul.Tensor(add_232, sigmoid_34);  add_232 = sigmoid_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_118: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_242, primals_452, primals_453, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_453 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_119: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_242, primals_454, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_120: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_119, primals_455, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_243: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_120, 1.0);  convolution_120 = None
        add_233: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_118, mul_243);  convolution_118 = mul_243 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_241: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_456, [1, 0]);  primals_456 = None
        addmm_27: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_457, mul_109, permute_241);  primals_457 = permute_241 = None
        unsqueeze_218: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_27, 2);  addmm_27 = None
        unsqueeze_219: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_234: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_233, unsqueeze_219);  add_233 = unsqueeze_219 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_613: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_234, [4, 32, 40, 64])
        var_mean_47 = torch.ops.aten.var_mean.correction(view_613, [2, 3], correction = 0, keepdim = True)
        getitem_156: "f32[4, 32, 1, 1]" = var_mean_47[0]
        getitem_157: "f32[4, 32, 1, 1]" = var_mean_47[1];  var_mean_47 = None
        add_235: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_47: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_47: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_613, getitem_157);  view_613 = None
        mul_245: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        view_614: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_245, [4, 1280, 8, 8]);  mul_245 = None
        unsqueeze_220: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_459, 0)
        unsqueeze_221: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        unsqueeze_223: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_458, 0)
        unsqueeze_224: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_246: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_614, unsqueeze_225);  view_614 = unsqueeze_225 = None
        add_236: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_222);  mul_246 = unsqueeze_222 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_36: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_236)
        mul_247: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_236, sigmoid_36);  add_236 = sigmoid_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_121: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_247, primals_460, primals_461, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_461 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_122: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_247, primals_462, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_123: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_122, primals_463, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_248: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_123, 1.0);  convolution_123 = None
        add_237: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_121, mul_248);  convolution_121 = mul_248 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_124: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(add_230, primals_464, primals_465, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_465 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_125: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(add_230, primals_466, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_126: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_125, primals_467, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_249: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_126, 1.0);  convolution_126 = None
        add_238: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_124, mul_249);  convolution_124 = mul_249 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_239: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_238, add_237);  add_238 = add_237 = None
        div_24: "f32[4, 1280, 8, 8]" = torch.ops.aten.div.Tensor(add_239, 1.0);  add_239 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_615: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(div_24, [4, 32, 40, 64])
        var_mean_48 = torch.ops.aten.var_mean.correction(view_615, [2, 3], correction = 0, keepdim = True)
        getitem_158: "f32[4, 32, 1, 1]" = var_mean_48[0]
        getitem_159: "f32[4, 32, 1, 1]" = var_mean_48[1];  var_mean_48 = None
        add_240: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_48: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_48: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_615, getitem_159);  view_615 = None
        mul_250: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        view_616: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_250, [4, 1280, 8, 8]);  mul_250 = None
        unsqueeze_226: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_469, 0);  primals_469 = None
        unsqueeze_227: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        unsqueeze_229: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_468, 0)
        unsqueeze_230: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_251: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_616, unsqueeze_231);  view_616 = unsqueeze_231 = None
        add_241: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_228);  mul_251 = unsqueeze_228 = None
        squeeze_72: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_159, [2, 3]);  getitem_159 = None
        squeeze_73: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_48, [2, 3]);  rsqrt_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_242: "f32[4, 8, 8, 1280]" = torch.ops.aten.permute.default(add_241, [0, 2, 3, 1]);  add_241 = None
        view_617: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_242, [4, 64, 1280]);  permute_242 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_243: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_470, [1, 0])
        expand_15: "f32[4, 64, 1280]" = torch.ops.aten.expand.default(view_617, [4, 64, 1280])
        expand_16: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_243, [4, 1280, 1280]);  permute_243 = None
        bmm_7: "f32[4, 64, 1280]" = torch.ops.aten.bmm.default(expand_15, expand_16);  expand_15 = expand_16 = None
        add_242: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(bmm_7, primals_471);  bmm_7 = primals_471 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_244: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_472, [1, 0]);  primals_472 = None
        clone_41: "f32[4, 64, 1280]" = torch.ops.aten.clone.default(view_617, memory_format = torch.contiguous_format);  view_617 = None
        view_621: "f32[256, 1280]" = torch.ops.aten.view.default(clone_41, [256, 1280]);  clone_41 = None
        mm_128: "f32[256, 80]" = torch.ops.aten.mm.default(view_621, permute_244)
        permute_245: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_473, [1, 0]);  primals_473 = None
        mm_129: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_128, permute_245)
        view_624: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_129, [4, 64, 1280]);  mm_129 = None
        mul_252: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_624, 1.0);  view_624 = None
        add_243: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_242, mul_252);  add_242 = mul_252 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_243, [2], correction = 0, keepdim = True)
        getitem_160: "f32[4, 64, 1]" = var_mean_49[0]
        getitem_161: "f32[4, 64, 1]" = var_mean_49[1];  var_mean_49 = None
        add_244: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_49: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_49: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_243, getitem_161);  getitem_161 = None
        mul_253: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        mul_254: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_253, primals_474)
        add_245: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_254, primals_475);  mul_254 = primals_475 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_246: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_476, [1, 0]);  primals_476 = None
        view_625: "f32[256, 1280]" = torch.ops.aten.view.default(add_245, [256, 1280]);  add_245 = None
        mm_130: "f32[256, 1280]" = torch.ops.aten.mm.default(view_625, permute_246)
        view_626: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_130, [4, 64, 1280]);  mm_130 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_247: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_477, [1, 0]);  primals_477 = None
        mm_131: "f32[256, 80]" = torch.ops.aten.mm.default(view_625, permute_247)
        permute_248: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_478, [1, 0]);  primals_478 = None
        mm_132: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_131, permute_248)
        view_630: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_132, [4, 64, 1280]);  mm_132 = None
        mul_255: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_630, 1.0);  view_630 = None
        add_246: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_626, mul_255);  view_626 = mul_255 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_249: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_479, [1, 0]);  primals_479 = None
        mm_133: "f32[256, 1280]" = torch.ops.aten.mm.default(view_625, permute_249)
        view_634: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_133, [4, 64, 1280]);  mm_133 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_250: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_480, [1, 0]);  primals_480 = None
        mm_134: "f32[256, 80]" = torch.ops.aten.mm.default(view_625, permute_250)
        permute_251: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_481, [1, 0]);  primals_481 = None
        mm_135: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_134, permute_251)
        view_638: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_135, [4, 64, 1280]);  mm_135 = None
        mul_256: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_638, 1.0);  view_638 = None
        add_247: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_634, mul_256);  view_634 = mul_256 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_252: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_482, [1, 0]);  primals_482 = None
        mm_136: "f32[256, 1280]" = torch.ops.aten.mm.default(view_625, permute_252)
        view_642: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_136, [4, 64, 1280]);  mm_136 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_253: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_483, [1, 0]);  primals_483 = None
        mm_137: "f32[256, 80]" = torch.ops.aten.mm.default(view_625, permute_253)
        permute_254: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_484, [1, 0]);  primals_484 = None
        mm_138: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_137, permute_254)
        view_646: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_138, [4, 64, 1280]);  mm_138 = None
        mul_257: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_646, 1.0);  view_646 = None
        add_248: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_642, mul_257);  view_642 = mul_257 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_653: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_246, [4, -1, 20, 64]);  add_246 = None
        permute_258: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
        view_655: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_247, [4, -1, 20, 64]);  add_247 = None
        permute_259: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_655, [0, 2, 1, 3]);  view_655 = None
        view_657: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_248, [4, -1, 20, 64]);  add_248 = None
        permute_260: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_258, permute_259, permute_260, None, True)
        getitem_162: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_9[0]
        getitem_163: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_9[1]
        getitem_164: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
        getitem_165: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_261: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3])
        view_658: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_261, [4, -1, 1280]);  permute_261 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_659: "f32[256, 1280]" = torch.ops.aten.view.default(view_658, [256, 1280]);  view_658 = None
        permute_262: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_485, [1, 0]);  primals_485 = None
        addmm_28: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_486, view_659, permute_262);  primals_486 = None
        view_660: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_28, [4, 64, 1280]);  addmm_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_263: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_487, [1, 0]);  primals_487 = None
        mm_139: "f32[256, 80]" = torch.ops.aten.mm.default(view_659, permute_263);  view_659 = None
        permute_264: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_488, [1, 0]);  primals_488 = None
        mm_140: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_139, permute_264)
        view_664: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_140, [4, 64, 1280]);  mm_140 = None
        mul_258: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_664, 1.0);  view_664 = None
        add_249: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_660, mul_258);  view_660 = mul_258 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_25: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_249, 1.0);  add_249 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_250: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_25, add_243);  div_25 = add_243 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_50 = torch.ops.aten.var_mean.correction(add_250, [2], correction = 0, keepdim = True)
        getitem_166: "f32[4, 64, 1]" = var_mean_50[0]
        getitem_167: "f32[4, 64, 1]" = var_mean_50[1];  var_mean_50 = None
        add_251: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_50: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_50: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_250, getitem_167);  getitem_167 = None
        mul_259: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        mul_260: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_259, primals_489)
        add_252: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_260, primals_490);  mul_260 = primals_490 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_265: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_491, [1, 0]);  primals_491 = None
        view_668: "f32[256, 1280]" = torch.ops.aten.view.default(add_252, [256, 1280]);  add_252 = None
        mm_141: "f32[256, 1280]" = torch.ops.aten.mm.default(view_668, permute_265)
        view_669: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_141, [4, 64, 1280]);  mm_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_266: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_492, [1, 0]);  primals_492 = None
        mm_142: "f32[256, 80]" = torch.ops.aten.mm.default(view_668, permute_266)
        permute_267: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_493, [1, 0]);  primals_493 = None
        mm_143: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_142, permute_267)
        view_673: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_143, [4, 64, 1280]);  mm_143 = None
        mul_261: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_673, 1.0);  view_673 = None
        add_253: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_669, mul_261);  view_669 = mul_261 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_268: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_494, [1, 0]);  primals_494 = None
        mm_144: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_268);  permute_268 = None
        view_677: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_144, [4, 77, 1280]);  mm_144 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_269: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
        mm_145: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_269);  permute_269 = None
        permute_270: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_496, [1, 0]);  primals_496 = None
        mm_146: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_145, permute_270)
        view_681: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_146, [4, 77, 1280]);  mm_146 = None
        mul_262: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_681, 1.0);  view_681 = None
        add_254: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_677, mul_262);  view_677 = mul_262 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_271: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_497, [1, 0]);  primals_497 = None
        mm_147: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_271);  permute_271 = None
        view_685: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_147, [4, 77, 1280]);  mm_147 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_272: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_498, [1, 0]);  primals_498 = None
        mm_148: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_272);  permute_272 = None
        permute_273: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_499, [1, 0]);  primals_499 = None
        mm_149: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_148, permute_273)
        view_689: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_149, [4, 77, 1280]);  mm_149 = None
        mul_263: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_689, 1.0);  view_689 = None
        add_255: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_685, mul_263);  view_685 = mul_263 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_696: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_253, [4, -1, 20, 64]);  add_253 = None
        permute_277: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None
        view_698: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_254, [4, -1, 20, 64]);  add_254 = None
        permute_278: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        view_700: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_255, [4, -1, 20, 64]);  add_255 = None
        permute_279: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_277, permute_278, permute_279, None, True)
        getitem_168: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_10[0]
        getitem_169: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_10[1]
        getitem_170: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
        getitem_171: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_280: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3])
        view_701: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_280, [4, -1, 1280]);  permute_280 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_702: "f32[256, 1280]" = torch.ops.aten.view.default(view_701, [256, 1280]);  view_701 = None
        permute_281: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_500, [1, 0]);  primals_500 = None
        addmm_29: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_501, view_702, permute_281);  primals_501 = None
        view_703: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_29, [4, 64, 1280]);  addmm_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_282: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_502, [1, 0]);  primals_502 = None
        mm_150: "f32[256, 80]" = torch.ops.aten.mm.default(view_702, permute_282);  view_702 = None
        permute_283: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_503, [1, 0]);  primals_503 = None
        mm_151: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_150, permute_283)
        view_707: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_151, [4, 64, 1280]);  mm_151 = None
        mul_264: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_707, 1.0);  view_707 = None
        add_256: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_703, mul_264);  view_703 = mul_264 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_26: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_256, 1.0);  add_256 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_257: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_26, add_250);  div_26 = add_250 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_51 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
        getitem_172: "f32[4, 64, 1]" = var_mean_51[0]
        getitem_173: "f32[4, 64, 1]" = var_mean_51[1];  var_mean_51 = None
        add_258: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_51: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_51: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_257, getitem_173);  getitem_173 = None
        mul_265: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        mul_266: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_265, primals_504)
        add_259: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_266, primals_505);  mul_266 = primals_505 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_711: "f32[256, 1280]" = torch.ops.aten.view.default(add_259, [256, 1280]);  add_259 = None
        permute_284: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_506, [1, 0]);  primals_506 = None
        addmm_30: "f32[256, 10240]" = torch.ops.aten.addmm.default(primals_507, view_711, permute_284);  primals_507 = None
        view_712: "f32[4, 64, 10240]" = torch.ops.aten.view.default(addmm_30, [4, 64, 10240]);  addmm_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_285: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_508, [1, 0]);  primals_508 = None
        mm_152: "f32[256, 80]" = torch.ops.aten.mm.default(view_711, permute_285)
        permute_286: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_509, [1, 0]);  primals_509 = None
        mm_153: "f32[256, 10240]" = torch.ops.aten.mm.default(mm_152, permute_286)
        view_716: "f32[4, 64, 10240]" = torch.ops.aten.view.default(mm_153, [4, 64, 10240]);  mm_153 = None
        mul_267: "f32[4, 64, 10240]" = torch.ops.aten.mul.Tensor(view_716, 1.0);  view_716 = None
        add_260: "f32[4, 64, 10240]" = torch.ops.aten.add.Tensor(view_712, mul_267);  view_712 = mul_267 = None
        view_717: "f32[256, 10240]" = torch.ops.aten.view.default(add_260, [256, 10240]);  add_260 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_720: "f32[4, 64, 10240]" = torch.ops.aten.view.default(view_717, [4, 64, 10240]);  view_717 = None
        split_14 = torch.ops.aten.split.Tensor(view_720, 5120, -1);  view_720 = None
        getitem_177: "f32[4, 64, 5120]" = split_14[1]
        mul_268: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_177, 0.5)
        mul_269: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_177, 0.7071067811865476)
        erf_4: "f32[4, 64, 5120]" = torch.ops.aten.erf.default(mul_269);  mul_269 = None
        add_261: "f32[4, 64, 5120]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_270: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_268, add_261);  mul_268 = add_261 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_178: "f32[4, 64, 5120]" = split_14[0];  split_14 = None
        mul_271: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_178, mul_270);  mul_270 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_722: "f32[256, 5120]" = torch.ops.aten.view.default(mul_271, [256, 5120]);  mul_271 = None
        permute_287: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_510, [1, 0]);  primals_510 = None
        addmm_31: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_511, view_722, permute_287);  primals_511 = None
        view_723: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_31, [4, 64, 1280]);  addmm_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_288: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_512, [1, 0]);  primals_512 = None
        mm_154: "f32[256, 80]" = torch.ops.aten.mm.default(view_722, permute_288)
        permute_289: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_513, [1, 0]);  primals_513 = None
        mm_155: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_154, permute_289)
        view_727: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_155, [4, 64, 1280]);  mm_155 = None
        mul_272: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_727, 1.0);  view_727 = None
        add_262: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_723, mul_272);  view_723 = mul_272 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_263: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_262, add_257);  add_262 = add_257 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_731: "f32[256, 1280]" = torch.ops.aten.view.default(add_263, [256, 1280]);  add_263 = None
        permute_290: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_514, [1, 0]);  primals_514 = None
        addmm_32: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_515, view_731, permute_290);  primals_515 = None
        view_732: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_32, [4, 64, 1280]);  addmm_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_291: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_516, [1, 0]);  primals_516 = None
        mm_156: "f32[256, 80]" = torch.ops.aten.mm.default(view_731, permute_291)
        permute_292: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_517, [1, 0]);  primals_517 = None
        mm_157: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_156, permute_292)
        view_736: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_157, [4, 64, 1280]);  mm_157 = None
        mul_273: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_736, 1.0);  view_736 = None
        add_264: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_732, mul_273);  view_732 = mul_273 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_742: "f32[4, 8, 8, 1280]" = torch.ops.aten.view.default(add_264, [4, 8, 8, 1280]);  add_264 = None
        permute_294: "f32[4, 1280, 8, 8]" = torch.ops.aten.permute.default(view_742, [0, 3, 1, 2]);  view_742 = None
        clone_45: "f32[4, 1280, 8, 8]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_265: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(clone_45, div_24);  clone_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_743: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_265, [4, 32, 40, 64])
        var_mean_52 = torch.ops.aten.var_mean.correction(view_743, [2, 3], correction = 0, keepdim = True)
        getitem_180: "f32[4, 32, 1, 1]" = var_mean_52[0]
        getitem_181: "f32[4, 32, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_266: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_52: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_52: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_743, getitem_181);  view_743 = None
        mul_274: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        view_744: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_274, [4, 1280, 8, 8]);  mul_274 = None
        unsqueeze_232: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_519, 0)
        unsqueeze_233: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        unsqueeze_235: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_518, 0)
        unsqueeze_236: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_275: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_744, unsqueeze_237);  view_744 = unsqueeze_237 = None
        add_267: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_234);  mul_275 = unsqueeze_234 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_37: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_267)
        mul_276: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_267, sigmoid_37);  add_267 = sigmoid_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_127: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_276, primals_520, primals_521, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_521 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_128: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_276, primals_522, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_129: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_128, primals_523, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_277: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_129, 1.0);  convolution_129 = None
        add_268: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_127, mul_277);  convolution_127 = mul_277 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_295: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_524, [1, 0]);  primals_524 = None
        addmm_33: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_525, mul_109, permute_295);  primals_525 = permute_295 = None
        unsqueeze_238: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_33, 2);  addmm_33 = None
        unsqueeze_239: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_269: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_268, unsqueeze_239);  add_268 = unsqueeze_239 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_745: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_269, [4, 32, 40, 64])
        var_mean_53 = torch.ops.aten.var_mean.correction(view_745, [2, 3], correction = 0, keepdim = True)
        getitem_182: "f32[4, 32, 1, 1]" = var_mean_53[0]
        getitem_183: "f32[4, 32, 1, 1]" = var_mean_53[1];  var_mean_53 = None
        add_270: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_53: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
        sub_53: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_745, getitem_183);  view_745 = None
        mul_279: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        view_746: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_279, [4, 1280, 8, 8]);  mul_279 = None
        unsqueeze_240: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_527, 0)
        unsqueeze_241: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_526, 0)
        unsqueeze_244: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_280: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_746, unsqueeze_245);  view_746 = unsqueeze_245 = None
        add_271: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_242);  mul_280 = unsqueeze_242 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_39: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_271)
        mul_281: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_271, sigmoid_39);  add_271 = sigmoid_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_130: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_281, primals_528, primals_529, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_529 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_131: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_281, primals_530, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_132: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_131, primals_531, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_282: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_132, 1.0);  convolution_132 = None
        add_272: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_130, mul_282);  convolution_130 = mul_282 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_273: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_265, add_272);  add_272 = None
        div_27: "f32[4, 1280, 8, 8]" = torch.ops.aten.div.Tensor(add_273, 1.0);  add_273 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_747: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(div_27, [4, 32, 40, 64])
        var_mean_54 = torch.ops.aten.var_mean.correction(view_747, [2, 3], correction = 0, keepdim = True)
        getitem_184: "f32[4, 32, 1, 1]" = var_mean_54[0]
        getitem_185: "f32[4, 32, 1, 1]" = var_mean_54[1];  var_mean_54 = None
        add_274: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
        rsqrt_54: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        sub_54: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_747, getitem_185);  view_747 = None
        mul_283: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        view_748: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_283, [4, 1280, 8, 8]);  mul_283 = None
        unsqueeze_246: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_533, 0);  primals_533 = None
        unsqueeze_247: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
        unsqueeze_248: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_532, 0)
        unsqueeze_250: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_284: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_748, unsqueeze_251);  view_748 = unsqueeze_251 = None
        add_275: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_248);  mul_284 = unsqueeze_248 = None
        squeeze_78: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_185, [2, 3]);  getitem_185 = None
        squeeze_79: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_54, [2, 3]);  rsqrt_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_296: "f32[4, 8, 8, 1280]" = torch.ops.aten.permute.default(add_275, [0, 2, 3, 1]);  add_275 = None
        view_749: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_296, [4, 64, 1280]);  permute_296 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_297: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_534, [1, 0])
        expand_17: "f32[4, 64, 1280]" = torch.ops.aten.expand.default(view_749, [4, 64, 1280])
        expand_18: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_297, [4, 1280, 1280]);  permute_297 = None
        bmm_8: "f32[4, 64, 1280]" = torch.ops.aten.bmm.default(expand_17, expand_18);  expand_17 = expand_18 = None
        add_276: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(bmm_8, primals_535);  bmm_8 = primals_535 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_298: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_536, [1, 0]);  primals_536 = None
        clone_47: "f32[4, 64, 1280]" = torch.ops.aten.clone.default(view_749, memory_format = torch.contiguous_format);  view_749 = None
        view_753: "f32[256, 1280]" = torch.ops.aten.view.default(clone_47, [256, 1280]);  clone_47 = None
        mm_158: "f32[256, 80]" = torch.ops.aten.mm.default(view_753, permute_298)
        permute_299: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_537, [1, 0]);  primals_537 = None
        mm_159: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_158, permute_299)
        view_756: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_159, [4, 64, 1280]);  mm_159 = None
        mul_285: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_756, 1.0);  view_756 = None
        add_277: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_276, mul_285);  add_276 = mul_285 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_55 = torch.ops.aten.var_mean.correction(add_277, [2], correction = 0, keepdim = True)
        getitem_186: "f32[4, 64, 1]" = var_mean_55[0]
        getitem_187: "f32[4, 64, 1]" = var_mean_55[1];  var_mean_55 = None
        add_278: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_55: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_55: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_277, getitem_187);  getitem_187 = None
        mul_286: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        mul_287: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_286, primals_538)
        add_279: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_287, primals_539);  mul_287 = primals_539 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_300: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_540, [1, 0]);  primals_540 = None
        view_757: "f32[256, 1280]" = torch.ops.aten.view.default(add_279, [256, 1280]);  add_279 = None
        mm_160: "f32[256, 1280]" = torch.ops.aten.mm.default(view_757, permute_300)
        view_758: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_160, [4, 64, 1280]);  mm_160 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_301: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_541, [1, 0]);  primals_541 = None
        mm_161: "f32[256, 80]" = torch.ops.aten.mm.default(view_757, permute_301)
        permute_302: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_542, [1, 0]);  primals_542 = None
        mm_162: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_161, permute_302)
        view_762: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_162, [4, 64, 1280]);  mm_162 = None
        mul_288: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_762, 1.0);  view_762 = None
        add_280: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_758, mul_288);  view_758 = mul_288 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_303: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_543, [1, 0]);  primals_543 = None
        mm_163: "f32[256, 1280]" = torch.ops.aten.mm.default(view_757, permute_303)
        view_766: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_163, [4, 64, 1280]);  mm_163 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_304: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_544, [1, 0]);  primals_544 = None
        mm_164: "f32[256, 80]" = torch.ops.aten.mm.default(view_757, permute_304)
        permute_305: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_545, [1, 0]);  primals_545 = None
        mm_165: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_164, permute_305)
        view_770: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_165, [4, 64, 1280]);  mm_165 = None
        mul_289: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_770, 1.0);  view_770 = None
        add_281: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_766, mul_289);  view_766 = mul_289 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_306: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_546, [1, 0]);  primals_546 = None
        mm_166: "f32[256, 1280]" = torch.ops.aten.mm.default(view_757, permute_306)
        view_774: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_166, [4, 64, 1280]);  mm_166 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_307: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_547, [1, 0]);  primals_547 = None
        mm_167: "f32[256, 80]" = torch.ops.aten.mm.default(view_757, permute_307)
        permute_308: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_548, [1, 0]);  primals_548 = None
        mm_168: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_167, permute_308)
        view_778: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_168, [4, 64, 1280]);  mm_168 = None
        mul_290: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_778, 1.0);  view_778 = None
        add_282: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_774, mul_290);  view_774 = mul_290 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_785: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_280, [4, -1, 20, 64]);  add_280 = None
        permute_312: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_785, [0, 2, 1, 3]);  view_785 = None
        view_787: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_281, [4, -1, 20, 64]);  add_281 = None
        permute_313: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
        view_789: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_282, [4, -1, 20, 64]);  add_282 = None
        permute_314: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_312, permute_313, permute_314, None, True)
        getitem_188: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_11[0]
        getitem_189: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_11[1]
        getitem_190: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
        getitem_191: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_315: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3])
        view_790: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_315, [4, -1, 1280]);  permute_315 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_791: "f32[256, 1280]" = torch.ops.aten.view.default(view_790, [256, 1280]);  view_790 = None
        permute_316: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_549, [1, 0]);  primals_549 = None
        addmm_34: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_550, view_791, permute_316);  primals_550 = None
        view_792: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_34, [4, 64, 1280]);  addmm_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_317: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_551, [1, 0]);  primals_551 = None
        mm_169: "f32[256, 80]" = torch.ops.aten.mm.default(view_791, permute_317);  view_791 = None
        permute_318: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_552, [1, 0]);  primals_552 = None
        mm_170: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_169, permute_318)
        view_796: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_170, [4, 64, 1280]);  mm_170 = None
        mul_291: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_796, 1.0);  view_796 = None
        add_283: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_792, mul_291);  view_792 = mul_291 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_28: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_283, 1.0);  add_283 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_284: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_28, add_277);  div_28 = add_277 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_56 = torch.ops.aten.var_mean.correction(add_284, [2], correction = 0, keepdim = True)
        getitem_192: "f32[4, 64, 1]" = var_mean_56[0]
        getitem_193: "f32[4, 64, 1]" = var_mean_56[1];  var_mean_56 = None
        add_285: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_56: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_56: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_284, getitem_193);  getitem_193 = None
        mul_292: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        mul_293: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_292, primals_553)
        add_286: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_293, primals_554);  mul_293 = primals_554 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_319: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_555, [1, 0]);  primals_555 = None
        view_800: "f32[256, 1280]" = torch.ops.aten.view.default(add_286, [256, 1280]);  add_286 = None
        mm_171: "f32[256, 1280]" = torch.ops.aten.mm.default(view_800, permute_319)
        view_801: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_171, [4, 64, 1280]);  mm_171 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_320: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_556, [1, 0]);  primals_556 = None
        mm_172: "f32[256, 80]" = torch.ops.aten.mm.default(view_800, permute_320)
        permute_321: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_557, [1, 0]);  primals_557 = None
        mm_173: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_172, permute_321)
        view_805: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_173, [4, 64, 1280]);  mm_173 = None
        mul_294: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_805, 1.0);  view_805 = None
        add_287: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_801, mul_294);  view_801 = mul_294 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_322: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_558, [1, 0]);  primals_558 = None
        mm_174: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_322);  permute_322 = None
        view_809: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_174, [4, 77, 1280]);  mm_174 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_323: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_559, [1, 0]);  primals_559 = None
        mm_175: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_323);  permute_323 = None
        permute_324: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_560, [1, 0]);  primals_560 = None
        mm_176: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_175, permute_324)
        view_813: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_176, [4, 77, 1280]);  mm_176 = None
        mul_295: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_813, 1.0);  view_813 = None
        add_288: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_809, mul_295);  view_809 = mul_295 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_325: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_561, [1, 0]);  primals_561 = None
        mm_177: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_325);  permute_325 = None
        view_817: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_177, [4, 77, 1280]);  mm_177 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_326: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_562, [1, 0]);  primals_562 = None
        mm_178: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_326);  permute_326 = None
        permute_327: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_563, [1, 0]);  primals_563 = None
        mm_179: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_178, permute_327)
        view_821: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_179, [4, 77, 1280]);  mm_179 = None
        mul_296: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_821, 1.0);  view_821 = None
        add_289: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_817, mul_296);  view_817 = mul_296 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_828: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_287, [4, -1, 20, 64]);  add_287 = None
        permute_331: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None
        view_830: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_288, [4, -1, 20, 64]);  add_288 = None
        permute_332: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None
        view_832: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_289, [4, -1, 20, 64]);  add_289 = None
        permute_333: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_331, permute_332, permute_333, None, True)
        getitem_194: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_12[0]
        getitem_195: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_12[1]
        getitem_196: "i64[]" = _scaled_dot_product_efficient_attention_12[2]
        getitem_197: "i64[]" = _scaled_dot_product_efficient_attention_12[3];  _scaled_dot_product_efficient_attention_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_334: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3])
        view_833: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_334, [4, -1, 1280]);  permute_334 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_834: "f32[256, 1280]" = torch.ops.aten.view.default(view_833, [256, 1280]);  view_833 = None
        permute_335: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_564, [1, 0]);  primals_564 = None
        addmm_35: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_565, view_834, permute_335);  primals_565 = None
        view_835: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_35, [4, 64, 1280]);  addmm_35 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_336: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_566, [1, 0]);  primals_566 = None
        mm_180: "f32[256, 80]" = torch.ops.aten.mm.default(view_834, permute_336);  view_834 = None
        permute_337: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_567, [1, 0]);  primals_567 = None
        mm_181: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_180, permute_337)
        view_839: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_181, [4, 64, 1280]);  mm_181 = None
        mul_297: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_839, 1.0);  view_839 = None
        add_290: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_835, mul_297);  view_835 = mul_297 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_29: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_290, 1.0);  add_290 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_291: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_29, add_284);  div_29 = add_284 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_57 = torch.ops.aten.var_mean.correction(add_291, [2], correction = 0, keepdim = True)
        getitem_198: "f32[4, 64, 1]" = var_mean_57[0]
        getitem_199: "f32[4, 64, 1]" = var_mean_57[1];  var_mean_57 = None
        add_292: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_57: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_57: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_291, getitem_199);  getitem_199 = None
        mul_298: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        mul_299: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_298, primals_568)
        add_293: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_299, primals_569);  mul_299 = primals_569 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_843: "f32[256, 1280]" = torch.ops.aten.view.default(add_293, [256, 1280]);  add_293 = None
        permute_338: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_570, [1, 0]);  primals_570 = None
        addmm_36: "f32[256, 10240]" = torch.ops.aten.addmm.default(primals_571, view_843, permute_338);  primals_571 = None
        view_844: "f32[4, 64, 10240]" = torch.ops.aten.view.default(addmm_36, [4, 64, 10240]);  addmm_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_339: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_572, [1, 0]);  primals_572 = None
        mm_182: "f32[256, 80]" = torch.ops.aten.mm.default(view_843, permute_339)
        permute_340: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_573, [1, 0]);  primals_573 = None
        mm_183: "f32[256, 10240]" = torch.ops.aten.mm.default(mm_182, permute_340)
        view_848: "f32[4, 64, 10240]" = torch.ops.aten.view.default(mm_183, [4, 64, 10240]);  mm_183 = None
        mul_300: "f32[4, 64, 10240]" = torch.ops.aten.mul.Tensor(view_848, 1.0);  view_848 = None
        add_294: "f32[4, 64, 10240]" = torch.ops.aten.add.Tensor(view_844, mul_300);  view_844 = mul_300 = None
        view_849: "f32[256, 10240]" = torch.ops.aten.view.default(add_294, [256, 10240]);  add_294 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_852: "f32[4, 64, 10240]" = torch.ops.aten.view.default(view_849, [4, 64, 10240]);  view_849 = None
        split_17 = torch.ops.aten.split.Tensor(view_852, 5120, -1);  view_852 = None
        getitem_203: "f32[4, 64, 5120]" = split_17[1]
        mul_301: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_203, 0.5)
        mul_302: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_203, 0.7071067811865476)
        erf_5: "f32[4, 64, 5120]" = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_295: "f32[4, 64, 5120]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_303: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_301, add_295);  mul_301 = add_295 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_204: "f32[4, 64, 5120]" = split_17[0];  split_17 = None
        mul_304: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_204, mul_303);  mul_303 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_854: "f32[256, 5120]" = torch.ops.aten.view.default(mul_304, [256, 5120]);  mul_304 = None
        permute_341: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_574, [1, 0]);  primals_574 = None
        addmm_37: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_575, view_854, permute_341);  primals_575 = None
        view_855: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_37, [4, 64, 1280]);  addmm_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_342: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_576, [1, 0]);  primals_576 = None
        mm_184: "f32[256, 80]" = torch.ops.aten.mm.default(view_854, permute_342)
        permute_343: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_577, [1, 0]);  primals_577 = None
        mm_185: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_184, permute_343)
        view_859: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_185, [4, 64, 1280]);  mm_185 = None
        mul_305: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_859, 1.0);  view_859 = None
        add_296: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_855, mul_305);  view_855 = mul_305 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_297: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_296, add_291);  add_296 = add_291 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_863: "f32[256, 1280]" = torch.ops.aten.view.default(add_297, [256, 1280]);  add_297 = None
        permute_344: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_578, [1, 0]);  primals_578 = None
        addmm_38: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_579, view_863, permute_344);  primals_579 = None
        view_864: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_38, [4, 64, 1280]);  addmm_38 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_345: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_580, [1, 0]);  primals_580 = None
        mm_186: "f32[256, 80]" = torch.ops.aten.mm.default(view_863, permute_345)
        permute_346: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_581, [1, 0]);  primals_581 = None
        mm_187: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_186, permute_346)
        view_868: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_187, [4, 64, 1280]);  mm_187 = None
        mul_306: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_868, 1.0);  view_868 = None
        add_298: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_864, mul_306);  view_864 = mul_306 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_874: "f32[4, 8, 8, 1280]" = torch.ops.aten.view.default(add_298, [4, 8, 8, 1280]);  add_298 = None
        permute_348: "f32[4, 1280, 8, 8]" = torch.ops.aten.permute.default(view_874, [0, 3, 1, 2]);  view_874 = None
        clone_51: "f32[4, 1280, 8, 8]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_299: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(clone_51, div_27);  clone_51 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_133: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(add_299, primals_582, primals_583, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_583 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_134: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(add_299, primals_584, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_135: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_134, primals_585, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_307: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_135, 1.0);  convolution_135 = None
        add_300: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_133, mul_307);  convolution_133 = mul_307 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_875: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_300, [4, 32, 40, 16])
        var_mean_58 = torch.ops.aten.var_mean.correction(view_875, [2, 3], correction = 0, keepdim = True)
        getitem_206: "f32[4, 32, 1, 1]" = var_mean_58[0]
        getitem_207: "f32[4, 32, 1, 1]" = var_mean_58[1];  var_mean_58 = None
        add_301: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_58: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
        sub_58: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_875, getitem_207);  view_875 = None
        mul_308: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        view_876: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_308, [4, 1280, 4, 4]);  mul_308 = None
        unsqueeze_252: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_587, 0)
        unsqueeze_253: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_586, 0)
        unsqueeze_256: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_309: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_876, unsqueeze_257);  view_876 = unsqueeze_257 = None
        add_302: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_254);  mul_309 = unsqueeze_254 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_40: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_302)
        mul_310: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_302, sigmoid_40);  add_302 = sigmoid_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_136: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_310, primals_588, primals_589, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_589 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_137: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_310, primals_590, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_138: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_137, primals_591, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_311: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_138, 1.0);  convolution_138 = None
        add_303: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_136, mul_311);  convolution_136 = mul_311 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_349: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_592, [1, 0]);  primals_592 = None
        addmm_39: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_593, mul_109, permute_349);  primals_593 = permute_349 = None
        unsqueeze_258: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_39, 2);  addmm_39 = None
        unsqueeze_259: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_304: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_303, unsqueeze_259);  add_303 = unsqueeze_259 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_877: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_304, [4, 32, 40, 16])
        var_mean_59 = torch.ops.aten.var_mean.correction(view_877, [2, 3], correction = 0, keepdim = True)
        getitem_208: "f32[4, 32, 1, 1]" = var_mean_59[0]
        getitem_209: "f32[4, 32, 1, 1]" = var_mean_59[1];  var_mean_59 = None
        add_305: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_59: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
        sub_59: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_877, getitem_209);  view_877 = None
        mul_313: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        view_878: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_313, [4, 1280, 4, 4]);  mul_313 = None
        unsqueeze_260: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_595, 0)
        unsqueeze_261: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        unsqueeze_263: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_594, 0)
        unsqueeze_264: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_314: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_878, unsqueeze_265);  view_878 = unsqueeze_265 = None
        add_306: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_262);  mul_314 = unsqueeze_262 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_42: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_306)
        mul_315: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_306, sigmoid_42);  add_306 = sigmoid_42 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_139: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_315, primals_596, primals_597, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_597 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_140: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_315, primals_598, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_141: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_140, primals_599, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_316: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_141, 1.0);  convolution_141 = None
        add_307: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_139, mul_316);  convolution_139 = mul_316 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_308: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_300, add_307);  add_307 = None
        div_30: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_308, 1.0);  add_308 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_879: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(div_30, [4, 32, 40, 16])
        var_mean_60 = torch.ops.aten.var_mean.correction(view_879, [2, 3], correction = 0, keepdim = True)
        getitem_210: "f32[4, 32, 1, 1]" = var_mean_60[0]
        getitem_211: "f32[4, 32, 1, 1]" = var_mean_60[1];  var_mean_60 = None
        add_309: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_60: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_60: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_879, getitem_211);  view_879 = None
        mul_317: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        view_880: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_317, [4, 1280, 4, 4]);  mul_317 = None
        unsqueeze_266: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_601, 0)
        unsqueeze_267: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        unsqueeze_269: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_600, 0)
        unsqueeze_270: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_318: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_880, unsqueeze_271);  view_880 = unsqueeze_271 = None
        add_310: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_268);  mul_318 = unsqueeze_268 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_43: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_310)
        mul_319: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_310, sigmoid_43);  add_310 = sigmoid_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_142: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_319, primals_602, primals_603, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_603 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_143: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_319, primals_604, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_144: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_143, primals_605, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_320: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_144, 1.0);  convolution_144 = None
        add_311: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_142, mul_320);  convolution_142 = mul_320 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_350: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_606, [1, 0]);  primals_606 = None
        addmm_40: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_607, mul_109, permute_350);  primals_607 = permute_350 = None
        unsqueeze_272: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_40, 2);  addmm_40 = None
        unsqueeze_273: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_312: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_311, unsqueeze_273);  add_311 = unsqueeze_273 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_881: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_312, [4, 32, 40, 16])
        var_mean_61 = torch.ops.aten.var_mean.correction(view_881, [2, 3], correction = 0, keepdim = True)
        getitem_212: "f32[4, 32, 1, 1]" = var_mean_61[0]
        getitem_213: "f32[4, 32, 1, 1]" = var_mean_61[1];  var_mean_61 = None
        add_313: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_61: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_61: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_881, getitem_213);  view_881 = None
        mul_322: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        view_882: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_322, [4, 1280, 4, 4]);  mul_322 = None
        unsqueeze_274: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_609, 0)
        unsqueeze_275: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_608, 0)
        unsqueeze_278: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_323: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_882, unsqueeze_279);  view_882 = unsqueeze_279 = None
        add_314: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_276);  mul_323 = unsqueeze_276 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_45: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_314)
        mul_324: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_314, sigmoid_45);  add_314 = sigmoid_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_145: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_324, primals_610, primals_611, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_611 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_146: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_324, primals_612, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_147: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_146, primals_613, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_325: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_147, 1.0);  convolution_147 = None
        add_315: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_145, mul_325);  convolution_145 = mul_325 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_316: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(div_30, add_315);  add_315 = None
        div_31: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_316, 1.0);  add_316 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_883: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(div_31, [4, 32, 40, 16])
        var_mean_62 = torch.ops.aten.var_mean.correction(view_883, [2, 3], correction = 0, keepdim = True)
        getitem_214: "f32[4, 32, 1, 1]" = var_mean_62[0]
        getitem_215: "f32[4, 32, 1, 1]" = var_mean_62[1];  var_mean_62 = None
        add_317: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_62: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_62: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_883, getitem_215);  view_883 = None
        mul_326: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
        view_884: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_326, [4, 1280, 4, 4]);  mul_326 = None
        unsqueeze_280: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_615, 0)
        unsqueeze_281: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_614, 0)
        unsqueeze_284: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_327: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_884, unsqueeze_285);  view_884 = unsqueeze_285 = None
        add_318: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_282);  mul_327 = unsqueeze_282 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_46: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_318)
        mul_328: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_318, sigmoid_46);  add_318 = sigmoid_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_148: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_328, primals_616, primals_617, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_617 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_149: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_328, primals_618, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_150: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_149, primals_619, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_329: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_150, 1.0);  convolution_150 = None
        add_319: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_148, mul_329);  convolution_148 = mul_329 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_351: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_620, [1, 0]);  primals_620 = None
        addmm_41: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_621, mul_109, permute_351);  primals_621 = permute_351 = None
        unsqueeze_286: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_41, 2);  addmm_41 = None
        unsqueeze_287: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_320: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_319, unsqueeze_287);  add_319 = unsqueeze_287 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_885: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_320, [4, 32, 40, 16])
        var_mean_63 = torch.ops.aten.var_mean.correction(view_885, [2, 3], correction = 0, keepdim = True)
        getitem_216: "f32[4, 32, 1, 1]" = var_mean_63[0]
        getitem_217: "f32[4, 32, 1, 1]" = var_mean_63[1];  var_mean_63 = None
        add_321: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_63: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
        sub_63: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_885, getitem_217);  view_885 = None
        mul_331: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        view_886: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_331, [4, 1280, 4, 4]);  mul_331 = None
        unsqueeze_288: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_623, 0)
        unsqueeze_289: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_622, 0)
        unsqueeze_292: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_332: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_886, unsqueeze_293);  view_886 = unsqueeze_293 = None
        add_322: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_290);  mul_332 = unsqueeze_290 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_48: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_322)
        mul_333: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_322, sigmoid_48);  add_322 = sigmoid_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_151: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_333, primals_624, primals_625, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_625 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_152: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_333, primals_626, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_153: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_152, primals_627, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_334: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_153, 1.0);  convolution_153 = None
        add_323: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_151, mul_334);  convolution_151 = mul_334 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_324: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(div_31, add_323);  add_323 = None
        div_32: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_324, 1);  add_324 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_887: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(div_32, [4, 32, 40, 16])
        var_mean_64 = torch.ops.aten.var_mean.correction(view_887, [2, 3], correction = 0, keepdim = True)
        getitem_218: "f32[4, 32, 1, 1]" = var_mean_64[0]
        getitem_219: "f32[4, 32, 1, 1]" = var_mean_64[1];  var_mean_64 = None
        add_325: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_64: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
        sub_64: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_887, getitem_219);  view_887 = None
        mul_335: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        view_888: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_335, [4, 1280, 4, 4]);  mul_335 = None
        unsqueeze_294: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_629, 0);  primals_629 = None
        unsqueeze_295: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
        unsqueeze_296: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_628, 0)
        unsqueeze_298: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_336: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_888, unsqueeze_299);  view_888 = unsqueeze_299 = None
        add_326: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_296);  mul_336 = unsqueeze_296 = None
        squeeze_92: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_219, [2, 3]);  getitem_219 = None
        squeeze_93: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_64, [2, 3]);  rsqrt_64 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_352: "f32[4, 4, 4, 1280]" = torch.ops.aten.permute.default(add_326, [0, 2, 3, 1]);  add_326 = None
        view_889: "f32[4, 16, 1280]" = torch.ops.aten.view.default(permute_352, [4, 16, 1280]);  permute_352 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_353: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_630, [1, 0])
        expand_19: "f32[4, 16, 1280]" = torch.ops.aten.expand.default(view_889, [4, 16, 1280])
        expand_20: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_353, [4, 1280, 1280]);  permute_353 = None
        bmm_9: "f32[4, 16, 1280]" = torch.ops.aten.bmm.default(expand_19, expand_20);  expand_19 = expand_20 = None
        add_327: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(bmm_9, primals_631);  bmm_9 = primals_631 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_354: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_632, [1, 0]);  primals_632 = None
        clone_55: "f32[4, 16, 1280]" = torch.ops.aten.clone.default(view_889, memory_format = torch.contiguous_format);  view_889 = None
        view_893: "f32[64, 1280]" = torch.ops.aten.view.default(clone_55, [64, 1280]);  clone_55 = None
        mm_188: "f32[64, 80]" = torch.ops.aten.mm.default(view_893, permute_354)
        permute_355: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_633, [1, 0]);  primals_633 = None
        mm_189: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_188, permute_355)
        view_896: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_189, [4, 16, 1280]);  mm_189 = None
        mul_337: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_896, 1.0);  view_896 = None
        add_328: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(add_327, mul_337);  add_327 = mul_337 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_65 = torch.ops.aten.var_mean.correction(add_328, [2], correction = 0, keepdim = True)
        getitem_220: "f32[4, 16, 1]" = var_mean_65[0]
        getitem_221: "f32[4, 16, 1]" = var_mean_65[1];  var_mean_65 = None
        add_329: "f32[4, 16, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_65: "f32[4, 16, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
        sub_65: "f32[4, 16, 1280]" = torch.ops.aten.sub.Tensor(add_328, getitem_221);  getitem_221 = None
        mul_338: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
        mul_339: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(mul_338, primals_634)
        add_330: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(mul_339, primals_635);  mul_339 = primals_635 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_356: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_636, [1, 0]);  primals_636 = None
        view_897: "f32[64, 1280]" = torch.ops.aten.view.default(add_330, [64, 1280]);  add_330 = None
        mm_190: "f32[64, 1280]" = torch.ops.aten.mm.default(view_897, permute_356)
        view_898: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_190, [4, 16, 1280]);  mm_190 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_357: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_637, [1, 0]);  primals_637 = None
        mm_191: "f32[64, 80]" = torch.ops.aten.mm.default(view_897, permute_357)
        permute_358: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_638, [1, 0]);  primals_638 = None
        mm_192: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_191, permute_358)
        view_902: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_192, [4, 16, 1280]);  mm_192 = None
        mul_340: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_902, 1.0);  view_902 = None
        add_331: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_898, mul_340);  view_898 = mul_340 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_359: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_639, [1, 0]);  primals_639 = None
        mm_193: "f32[64, 1280]" = torch.ops.aten.mm.default(view_897, permute_359)
        view_906: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_193, [4, 16, 1280]);  mm_193 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_360: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_640, [1, 0]);  primals_640 = None
        mm_194: "f32[64, 80]" = torch.ops.aten.mm.default(view_897, permute_360)
        permute_361: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_641, [1, 0]);  primals_641 = None
        mm_195: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_194, permute_361)
        view_910: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_195, [4, 16, 1280]);  mm_195 = None
        mul_341: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_910, 1.0);  view_910 = None
        add_332: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_906, mul_341);  view_906 = mul_341 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_362: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_642, [1, 0]);  primals_642 = None
        mm_196: "f32[64, 1280]" = torch.ops.aten.mm.default(view_897, permute_362)
        view_914: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_196, [4, 16, 1280]);  mm_196 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_363: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_643, [1, 0]);  primals_643 = None
        mm_197: "f32[64, 80]" = torch.ops.aten.mm.default(view_897, permute_363)
        permute_364: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_644, [1, 0]);  primals_644 = None
        mm_198: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_197, permute_364)
        view_918: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_198, [4, 16, 1280]);  mm_198 = None
        mul_342: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_918, 1.0);  view_918 = None
        add_333: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_914, mul_342);  view_914 = mul_342 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_925: "f32[4, 16, 20, 64]" = torch.ops.aten.view.default(add_331, [4, -1, 20, 64]);  add_331 = None
        permute_368: "f32[4, 20, 16, 64]" = torch.ops.aten.permute.default(view_925, [0, 2, 1, 3]);  view_925 = None
        view_927: "f32[4, 16, 20, 64]" = torch.ops.aten.view.default(add_332, [4, -1, 20, 64]);  add_332 = None
        permute_369: "f32[4, 20, 16, 64]" = torch.ops.aten.permute.default(view_927, [0, 2, 1, 3]);  view_927 = None
        view_929: "f32[4, 16, 20, 64]" = torch.ops.aten.view.default(add_333, [4, -1, 20, 64]);  add_333 = None
        permute_370: "f32[4, 20, 16, 64]" = torch.ops.aten.permute.default(view_929, [0, 2, 1, 3]);  view_929 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_368, permute_369, permute_370, None, True)
        getitem_222: "f32[4, 20, 16, 64]" = _scaled_dot_product_efficient_attention_13[0]
        getitem_223: "f32[4, 20, 32]" = _scaled_dot_product_efficient_attention_13[1]
        getitem_224: "i64[]" = _scaled_dot_product_efficient_attention_13[2]
        getitem_225: "i64[]" = _scaled_dot_product_efficient_attention_13[3];  _scaled_dot_product_efficient_attention_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_371: "f32[4, 16, 20, 64]" = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3])
        view_930: "f32[4, 16, 1280]" = torch.ops.aten.view.default(permute_371, [4, -1, 1280]);  permute_371 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_931: "f32[64, 1280]" = torch.ops.aten.view.default(view_930, [64, 1280]);  view_930 = None
        permute_372: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_645, [1, 0]);  primals_645 = None
        addmm_42: "f32[64, 1280]" = torch.ops.aten.addmm.default(primals_646, view_931, permute_372);  primals_646 = None
        view_932: "f32[4, 16, 1280]" = torch.ops.aten.view.default(addmm_42, [4, 16, 1280]);  addmm_42 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_373: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_647, [1, 0]);  primals_647 = None
        mm_199: "f32[64, 80]" = torch.ops.aten.mm.default(view_931, permute_373);  view_931 = None
        permute_374: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_648, [1, 0]);  primals_648 = None
        mm_200: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_199, permute_374)
        view_936: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_200, [4, 16, 1280]);  mm_200 = None
        mul_343: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_936, 1.0);  view_936 = None
        add_334: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_932, mul_343);  view_932 = mul_343 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_33: "f32[4, 16, 1280]" = torch.ops.aten.div.Tensor(add_334, 1.0);  add_334 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_335: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(div_33, add_328);  div_33 = add_328 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_66 = torch.ops.aten.var_mean.correction(add_335, [2], correction = 0, keepdim = True)
        getitem_226: "f32[4, 16, 1]" = var_mean_66[0]
        getitem_227: "f32[4, 16, 1]" = var_mean_66[1];  var_mean_66 = None
        add_336: "f32[4, 16, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_66: "f32[4, 16, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
        sub_66: "f32[4, 16, 1280]" = torch.ops.aten.sub.Tensor(add_335, getitem_227);  getitem_227 = None
        mul_344: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
        mul_345: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(mul_344, primals_649)
        add_337: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(mul_345, primals_650);  mul_345 = primals_650 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_375: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_651, [1, 0]);  primals_651 = None
        view_940: "f32[64, 1280]" = torch.ops.aten.view.default(add_337, [64, 1280]);  add_337 = None
        mm_201: "f32[64, 1280]" = torch.ops.aten.mm.default(view_940, permute_375)
        view_941: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_201, [4, 16, 1280]);  mm_201 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_376: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_652, [1, 0]);  primals_652 = None
        mm_202: "f32[64, 80]" = torch.ops.aten.mm.default(view_940, permute_376)
        permute_377: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_653, [1, 0]);  primals_653 = None
        mm_203: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_202, permute_377)
        view_945: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_203, [4, 16, 1280]);  mm_203 = None
        mul_346: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_945, 1.0);  view_945 = None
        add_338: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_941, mul_346);  view_941 = mul_346 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_378: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_654, [1, 0]);  primals_654 = None
        mm_204: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_378);  permute_378 = None
        view_949: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_204, [4, 77, 1280]);  mm_204 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_379: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_655, [1, 0]);  primals_655 = None
        mm_205: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_379);  permute_379 = None
        permute_380: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_656, [1, 0]);  primals_656 = None
        mm_206: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_205, permute_380)
        view_953: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_206, [4, 77, 1280]);  mm_206 = None
        mul_347: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_953, 1.0);  view_953 = None
        add_339: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_949, mul_347);  view_949 = mul_347 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_381: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_657, [1, 0]);  primals_657 = None
        mm_207: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_381);  permute_381 = None
        view_957: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_207, [4, 77, 1280]);  mm_207 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_382: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_658, [1, 0]);  primals_658 = None
        mm_208: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_382);  permute_382 = None
        permute_383: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_659, [1, 0]);  primals_659 = None
        mm_209: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_208, permute_383)
        view_961: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_209, [4, 77, 1280]);  mm_209 = None
        mul_348: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_961, 1.0);  view_961 = None
        add_340: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_957, mul_348);  view_957 = mul_348 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_968: "f32[4, 16, 20, 64]" = torch.ops.aten.view.default(add_338, [4, -1, 20, 64]);  add_338 = None
        permute_387: "f32[4, 20, 16, 64]" = torch.ops.aten.permute.default(view_968, [0, 2, 1, 3]);  view_968 = None
        view_970: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_339, [4, -1, 20, 64]);  add_339 = None
        permute_388: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
        view_972: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_340, [4, -1, 20, 64]);  add_340 = None
        permute_389: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_387, permute_388, permute_389, None, True)
        getitem_228: "f32[4, 20, 16, 64]" = _scaled_dot_product_efficient_attention_14[0]
        getitem_229: "f32[4, 20, 32]" = _scaled_dot_product_efficient_attention_14[1]
        getitem_230: "i64[]" = _scaled_dot_product_efficient_attention_14[2]
        getitem_231: "i64[]" = _scaled_dot_product_efficient_attention_14[3];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_390: "f32[4, 16, 20, 64]" = torch.ops.aten.permute.default(getitem_228, [0, 2, 1, 3])
        view_973: "f32[4, 16, 1280]" = torch.ops.aten.view.default(permute_390, [4, -1, 1280]);  permute_390 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_974: "f32[64, 1280]" = torch.ops.aten.view.default(view_973, [64, 1280]);  view_973 = None
        permute_391: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_660, [1, 0]);  primals_660 = None
        addmm_43: "f32[64, 1280]" = torch.ops.aten.addmm.default(primals_661, view_974, permute_391);  primals_661 = None
        view_975: "f32[4, 16, 1280]" = torch.ops.aten.view.default(addmm_43, [4, 16, 1280]);  addmm_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_392: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_662, [1, 0]);  primals_662 = None
        mm_210: "f32[64, 80]" = torch.ops.aten.mm.default(view_974, permute_392);  view_974 = None
        permute_393: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_663, [1, 0]);  primals_663 = None
        mm_211: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_210, permute_393)
        view_979: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_211, [4, 16, 1280]);  mm_211 = None
        mul_349: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_979, 1.0);  view_979 = None
        add_341: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_975, mul_349);  view_975 = mul_349 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_34: "f32[4, 16, 1280]" = torch.ops.aten.div.Tensor(add_341, 1.0);  add_341 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_342: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(div_34, add_335);  div_34 = add_335 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_67 = torch.ops.aten.var_mean.correction(add_342, [2], correction = 0, keepdim = True)
        getitem_232: "f32[4, 16, 1]" = var_mean_67[0]
        getitem_233: "f32[4, 16, 1]" = var_mean_67[1];  var_mean_67 = None
        add_343: "f32[4, 16, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_67: "f32[4, 16, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
        sub_67: "f32[4, 16, 1280]" = torch.ops.aten.sub.Tensor(add_342, getitem_233);  getitem_233 = None
        mul_350: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
        mul_351: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(mul_350, primals_664)
        add_344: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(mul_351, primals_665);  mul_351 = primals_665 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_983: "f32[64, 1280]" = torch.ops.aten.view.default(add_344, [64, 1280]);  add_344 = None
        permute_394: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_666, [1, 0]);  primals_666 = None
        addmm_44: "f32[64, 10240]" = torch.ops.aten.addmm.default(primals_667, view_983, permute_394);  primals_667 = None
        view_984: "f32[4, 16, 10240]" = torch.ops.aten.view.default(addmm_44, [4, 16, 10240]);  addmm_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_395: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_668, [1, 0]);  primals_668 = None
        mm_212: "f32[64, 80]" = torch.ops.aten.mm.default(view_983, permute_395)
        permute_396: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_669, [1, 0]);  primals_669 = None
        mm_213: "f32[64, 10240]" = torch.ops.aten.mm.default(mm_212, permute_396)
        view_988: "f32[4, 16, 10240]" = torch.ops.aten.view.default(mm_213, [4, 16, 10240]);  mm_213 = None
        mul_352: "f32[4, 16, 10240]" = torch.ops.aten.mul.Tensor(view_988, 1.0);  view_988 = None
        add_345: "f32[4, 16, 10240]" = torch.ops.aten.add.Tensor(view_984, mul_352);  view_984 = mul_352 = None
        view_989: "f32[64, 10240]" = torch.ops.aten.view.default(add_345, [64, 10240]);  add_345 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_992: "f32[4, 16, 10240]" = torch.ops.aten.view.default(view_989, [4, 16, 10240]);  view_989 = None
        split_20 = torch.ops.aten.split.Tensor(view_992, 5120, -1);  view_992 = None
        getitem_237: "f32[4, 16, 5120]" = split_20[1]
        mul_353: "f32[4, 16, 5120]" = torch.ops.aten.mul.Tensor(getitem_237, 0.5)
        mul_354: "f32[4, 16, 5120]" = torch.ops.aten.mul.Tensor(getitem_237, 0.7071067811865476)
        erf_6: "f32[4, 16, 5120]" = torch.ops.aten.erf.default(mul_354);  mul_354 = None
        add_346: "f32[4, 16, 5120]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_355: "f32[4, 16, 5120]" = torch.ops.aten.mul.Tensor(mul_353, add_346);  mul_353 = add_346 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_238: "f32[4, 16, 5120]" = split_20[0];  split_20 = None
        mul_356: "f32[4, 16, 5120]" = torch.ops.aten.mul.Tensor(getitem_238, mul_355);  mul_355 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_994: "f32[64, 5120]" = torch.ops.aten.view.default(mul_356, [64, 5120]);  mul_356 = None
        permute_397: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_670, [1, 0]);  primals_670 = None
        addmm_45: "f32[64, 1280]" = torch.ops.aten.addmm.default(primals_671, view_994, permute_397);  primals_671 = None
        view_995: "f32[4, 16, 1280]" = torch.ops.aten.view.default(addmm_45, [4, 16, 1280]);  addmm_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_398: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_672, [1, 0]);  primals_672 = None
        mm_214: "f32[64, 80]" = torch.ops.aten.mm.default(view_994, permute_398)
        permute_399: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_673, [1, 0]);  primals_673 = None
        mm_215: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_214, permute_399)
        view_999: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_215, [4, 16, 1280]);  mm_215 = None
        mul_357: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_999, 1.0);  view_999 = None
        add_347: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_995, mul_357);  view_995 = mul_357 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_348: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(add_347, add_342);  add_347 = add_342 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1003: "f32[64, 1280]" = torch.ops.aten.view.default(add_348, [64, 1280]);  add_348 = None
        permute_400: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_674, [1, 0]);  primals_674 = None
        addmm_46: "f32[64, 1280]" = torch.ops.aten.addmm.default(primals_675, view_1003, permute_400);  primals_675 = None
        view_1004: "f32[4, 16, 1280]" = torch.ops.aten.view.default(addmm_46, [4, 16, 1280]);  addmm_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_401: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_676, [1, 0]);  primals_676 = None
        mm_216: "f32[64, 80]" = torch.ops.aten.mm.default(view_1003, permute_401)
        permute_402: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_677, [1, 0]);  primals_677 = None
        mm_217: "f32[64, 1280]" = torch.ops.aten.mm.default(mm_216, permute_402)
        view_1008: "f32[4, 16, 1280]" = torch.ops.aten.view.default(mm_217, [4, 16, 1280]);  mm_217 = None
        mul_358: "f32[4, 16, 1280]" = torch.ops.aten.mul.Tensor(view_1008, 1.0);  view_1008 = None
        add_349: "f32[4, 16, 1280]" = torch.ops.aten.add.Tensor(view_1004, mul_358);  view_1004 = mul_358 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1014: "f32[4, 4, 4, 1280]" = torch.ops.aten.view.default(add_349, [4, 4, 4, 1280]);  add_349 = None
        permute_404: "f32[4, 1280, 4, 4]" = torch.ops.aten.permute.default(view_1014, [0, 3, 1, 2]);  view_1014 = None
        clone_59: "f32[4, 1280, 4, 4]" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_350: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(clone_59, div_32);  clone_59 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1015: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_350, [4, 32, 40, 16])
        var_mean_68 = torch.ops.aten.var_mean.correction(view_1015, [2, 3], correction = 0, keepdim = True)
        getitem_240: "f32[4, 32, 1, 1]" = var_mean_68[0]
        getitem_241: "f32[4, 32, 1, 1]" = var_mean_68[1];  var_mean_68 = None
        add_351: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_68: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_68: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_1015, getitem_241);  view_1015 = None
        mul_359: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        view_1016: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_359, [4, 1280, 4, 4]);  mul_359 = None
        unsqueeze_300: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_679, 0)
        unsqueeze_301: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_678, 0)
        unsqueeze_304: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_360: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_1016, unsqueeze_305);  view_1016 = unsqueeze_305 = None
        add_352: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_360, unsqueeze_302);  mul_360 = unsqueeze_302 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_49: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_352)
        mul_361: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_352, sigmoid_49);  add_352 = sigmoid_49 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_154: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_361, primals_680, primals_681, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_681 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_155: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_361, primals_682, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_156: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_155, primals_683, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_362: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_156, 1.0);  convolution_156 = None
        add_353: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_154, mul_362);  convolution_154 = mul_362 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_405: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_684, [1, 0]);  primals_684 = None
        addmm_47: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_685, mul_109, permute_405);  primals_685 = permute_405 = None
        unsqueeze_306: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_47, 2);  addmm_47 = None
        unsqueeze_307: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_354: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_353, unsqueeze_307);  add_353 = unsqueeze_307 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1017: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_354, [4, 32, 40, 16])
        var_mean_69 = torch.ops.aten.var_mean.correction(view_1017, [2, 3], correction = 0, keepdim = True)
        getitem_242: "f32[4, 32, 1, 1]" = var_mean_69[0]
        getitem_243: "f32[4, 32, 1, 1]" = var_mean_69[1];  var_mean_69 = None
        add_355: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_69: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_69: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_1017, getitem_243);  view_1017 = None
        mul_364: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        view_1018: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_364, [4, 1280, 4, 4]);  mul_364 = None
        unsqueeze_308: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_687, 0)
        unsqueeze_309: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        unsqueeze_311: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_686, 0)
        unsqueeze_312: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_365: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_1018, unsqueeze_313);  view_1018 = unsqueeze_313 = None
        add_356: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_310);  mul_365 = unsqueeze_310 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_51: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_356)
        mul_366: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_356, sigmoid_51);  add_356 = sigmoid_51 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_157: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_366, primals_688, primals_689, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_689 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_158: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_366, primals_690, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_159: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_158, primals_691, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_367: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_159, 1.0);  convolution_159 = None
        add_357: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_157, mul_367);  convolution_157 = mul_367 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_358: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_350, add_357);  add_357 = None
        div_35: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_358, 1);  add_358 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2721 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_2: "f32[4, 2560, 4, 4]" = torch.ops.aten.cat.default([div_35, div_31], 1);  div_35 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1019: "f32[4, 32, 80, 16]" = torch.ops.aten.view.default(cat_2, [4, 32, 80, 16])
        var_mean_70 = torch.ops.aten.var_mean.correction(view_1019, [2, 3], correction = 0, keepdim = True)
        getitem_244: "f32[4, 32, 1, 1]" = var_mean_70[0]
        getitem_245: "f32[4, 32, 1, 1]" = var_mean_70[1];  var_mean_70 = None
        add_359: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-05);  getitem_244 = None
        rsqrt_70: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        sub_70: "f32[4, 32, 80, 16]" = torch.ops.aten.sub.Tensor(view_1019, getitem_245);  view_1019 = None
        mul_368: "f32[4, 32, 80, 16]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
        view_1020: "f32[4, 2560, 4, 4]" = torch.ops.aten.view.default(mul_368, [4, 2560, 4, 4]);  mul_368 = None
        unsqueeze_314: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_693, 0)
        unsqueeze_315: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        unsqueeze_317: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_692, 0)
        unsqueeze_318: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_369: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(view_1020, unsqueeze_319);  view_1020 = unsqueeze_319 = None
        add_360: "f32[4, 2560, 4, 4]" = torch.ops.aten.add.Tensor(mul_369, unsqueeze_316);  mul_369 = unsqueeze_316 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_52: "f32[4, 2560, 4, 4]" = torch.ops.aten.sigmoid.default(add_360)
        mul_370: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(add_360, sigmoid_52);  add_360 = sigmoid_52 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_160: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_370, primals_694, primals_695, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_695 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_161: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_370, primals_696, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_162: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_161, primals_697, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_371: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_162, 1.0);  convolution_162 = None
        add_361: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_160, mul_371);  convolution_160 = mul_371 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_406: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_698, [1, 0]);  primals_698 = None
        addmm_48: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_699, mul_109, permute_406);  primals_699 = permute_406 = None
        unsqueeze_320: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_48, 2);  addmm_48 = None
        unsqueeze_321: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_362: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_361, unsqueeze_321);  add_361 = unsqueeze_321 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1021: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_362, [4, 32, 40, 16])
        var_mean_71 = torch.ops.aten.var_mean.correction(view_1021, [2, 3], correction = 0, keepdim = True)
        getitem_246: "f32[4, 32, 1, 1]" = var_mean_71[0]
        getitem_247: "f32[4, 32, 1, 1]" = var_mean_71[1];  var_mean_71 = None
        add_363: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_71: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
        sub_71: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_1021, getitem_247);  view_1021 = None
        mul_373: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        view_1022: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_373, [4, 1280, 4, 4]);  mul_373 = None
        unsqueeze_322: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_701, 0)
        unsqueeze_323: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_700, 0)
        unsqueeze_326: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_374: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_1022, unsqueeze_327);  view_1022 = unsqueeze_327 = None
        add_364: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_324);  mul_374 = unsqueeze_324 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_54: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_364)
        mul_375: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_364, sigmoid_54);  add_364 = sigmoid_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_163: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_375, primals_702, primals_703, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_703 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_164: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_375, primals_704, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_165: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_164, primals_705, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_376: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_165, 1.0);  convolution_165 = None
        add_365: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_163, mul_376);  convolution_163 = mul_376 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_166: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(cat_2, primals_706, primals_707, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_707 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_167: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(cat_2, primals_708, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_168: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_167, primals_709, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_377: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_168, 1.0);  convolution_168 = None
        add_366: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_166, mul_377);  convolution_166 = mul_377 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_367: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_366, add_365);  add_366 = add_365 = None
        div_36: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_367, 1.0);  add_367 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2721 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_3: "f32[4, 2560, 4, 4]" = torch.ops.aten.cat.default([div_36, div_30], 1);  div_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1023: "f32[4, 32, 80, 16]" = torch.ops.aten.view.default(cat_3, [4, 32, 80, 16])
        var_mean_72 = torch.ops.aten.var_mean.correction(view_1023, [2, 3], correction = 0, keepdim = True)
        getitem_248: "f32[4, 32, 1, 1]" = var_mean_72[0]
        getitem_249: "f32[4, 32, 1, 1]" = var_mean_72[1];  var_mean_72 = None
        add_368: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_72: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        sub_72: "f32[4, 32, 80, 16]" = torch.ops.aten.sub.Tensor(view_1023, getitem_249);  view_1023 = None
        mul_378: "f32[4, 32, 80, 16]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        view_1024: "f32[4, 2560, 4, 4]" = torch.ops.aten.view.default(mul_378, [4, 2560, 4, 4]);  mul_378 = None
        unsqueeze_328: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_711, 0)
        unsqueeze_329: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_710, 0)
        unsqueeze_332: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_379: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(view_1024, unsqueeze_333);  view_1024 = unsqueeze_333 = None
        add_369: "f32[4, 2560, 4, 4]" = torch.ops.aten.add.Tensor(mul_379, unsqueeze_330);  mul_379 = unsqueeze_330 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_55: "f32[4, 2560, 4, 4]" = torch.ops.aten.sigmoid.default(add_369)
        mul_380: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(add_369, sigmoid_55);  add_369 = sigmoid_55 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_169: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_380, primals_712, primals_713, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_713 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_170: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_380, primals_714, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_171: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_170, primals_715, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_381: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_171, 1.0);  convolution_171 = None
        add_370: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_169, mul_381);  convolution_169 = mul_381 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_407: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_716, [1, 0]);  primals_716 = None
        addmm_49: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_717, mul_109, permute_407);  primals_717 = permute_407 = None
        unsqueeze_334: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_49, 2);  addmm_49 = None
        unsqueeze_335: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_371: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_370, unsqueeze_335);  add_370 = unsqueeze_335 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1025: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_371, [4, 32, 40, 16])
        var_mean_73 = torch.ops.aten.var_mean.correction(view_1025, [2, 3], correction = 0, keepdim = True)
        getitem_250: "f32[4, 32, 1, 1]" = var_mean_73[0]
        getitem_251: "f32[4, 32, 1, 1]" = var_mean_73[1];  var_mean_73 = None
        add_372: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
        rsqrt_73: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_372);  add_372 = None
        sub_73: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_1025, getitem_251);  view_1025 = None
        mul_383: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
        view_1026: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_383, [4, 1280, 4, 4]);  mul_383 = None
        unsqueeze_336: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_719, 0)
        unsqueeze_337: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
        unsqueeze_338: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_718, 0)
        unsqueeze_340: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        unsqueeze_341: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
        mul_384: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_1026, unsqueeze_341);  view_1026 = unsqueeze_341 = None
        add_373: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_338);  mul_384 = unsqueeze_338 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_57: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_373)
        mul_385: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_373, sigmoid_57);  add_373 = sigmoid_57 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_172: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_385, primals_720, primals_721, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_721 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_173: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_385, primals_722, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_174: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_173, primals_723, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_386: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_174, 1.0);  convolution_174 = None
        add_374: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_172, mul_386);  convolution_172 = mul_386 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_175: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(cat_3, primals_724, primals_725, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_725 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_176: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(cat_3, primals_726, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_177: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_176, primals_727, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_387: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_177, 1.0);  convolution_177 = None
        add_375: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_175, mul_387);  convolution_175 = mul_387 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_376: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_375, add_374);  add_375 = add_374 = None
        div_37: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_376, 1.0);  add_376 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2721 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_4: "f32[4, 2560, 4, 4]" = torch.ops.aten.cat.default([div_37, add_300], 1);  div_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1027: "f32[4, 32, 80, 16]" = torch.ops.aten.view.default(cat_4, [4, 32, 80, 16])
        var_mean_74 = torch.ops.aten.var_mean.correction(view_1027, [2, 3], correction = 0, keepdim = True)
        getitem_252: "f32[4, 32, 1, 1]" = var_mean_74[0]
        getitem_253: "f32[4, 32, 1, 1]" = var_mean_74[1];  var_mean_74 = None
        add_377: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_74: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_377);  add_377 = None
        sub_74: "f32[4, 32, 80, 16]" = torch.ops.aten.sub.Tensor(view_1027, getitem_253);  view_1027 = None
        mul_388: "f32[4, 32, 80, 16]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
        view_1028: "f32[4, 2560, 4, 4]" = torch.ops.aten.view.default(mul_388, [4, 2560, 4, 4]);  mul_388 = None
        unsqueeze_342: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_729, 0)
        unsqueeze_343: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
        unsqueeze_344: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_728, 0)
        unsqueeze_346: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
        unsqueeze_347: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
        mul_389: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(view_1028, unsqueeze_347);  view_1028 = unsqueeze_347 = None
        add_378: "f32[4, 2560, 4, 4]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_344);  mul_389 = unsqueeze_344 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_58: "f32[4, 2560, 4, 4]" = torch.ops.aten.sigmoid.default(add_378)
        mul_390: "f32[4, 2560, 4, 4]" = torch.ops.aten.mul.Tensor(add_378, sigmoid_58);  add_378 = sigmoid_58 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_178: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_390, primals_730, primals_731, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_731 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_179: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_390, primals_732, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_180: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_179, primals_733, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_391: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_180, 1.0);  convolution_180 = None
        add_379: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_178, mul_391);  convolution_178 = mul_391 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_408: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_734, [1, 0]);  primals_734 = None
        addmm_50: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_735, mul_109, permute_408);  primals_735 = permute_408 = None
        unsqueeze_348: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_50, 2);  addmm_50 = None
        unsqueeze_349: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_380: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_379, unsqueeze_349);  add_379 = unsqueeze_349 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1029: "f32[4, 32, 40, 16]" = torch.ops.aten.view.default(add_380, [4, 32, 40, 16])
        var_mean_75 = torch.ops.aten.var_mean.correction(view_1029, [2, 3], correction = 0, keepdim = True)
        getitem_254: "f32[4, 32, 1, 1]" = var_mean_75[0]
        getitem_255: "f32[4, 32, 1, 1]" = var_mean_75[1];  var_mean_75 = None
        add_381: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_75: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_381);  add_381 = None
        sub_75: "f32[4, 32, 40, 16]" = torch.ops.aten.sub.Tensor(view_1029, getitem_255);  view_1029 = None
        mul_393: "f32[4, 32, 40, 16]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        view_1030: "f32[4, 1280, 4, 4]" = torch.ops.aten.view.default(mul_393, [4, 1280, 4, 4]);  mul_393 = None
        unsqueeze_350: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_737, 0)
        unsqueeze_351: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        unsqueeze_353: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_736, 0)
        unsqueeze_354: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_394: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(view_1030, unsqueeze_355);  view_1030 = unsqueeze_355 = None
        add_382: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_352);  mul_394 = unsqueeze_352 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_60: "f32[4, 1280, 4, 4]" = torch.ops.aten.sigmoid.default(add_382)
        mul_395: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(add_382, sigmoid_60);  add_382 = sigmoid_60 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_181: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(mul_395, primals_738, primals_739, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_739 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_182: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(mul_395, primals_740, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_183: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_182, primals_741, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_396: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_183, 1.0);  convolution_183 = None
        add_383: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_181, mul_396);  convolution_181 = mul_396 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_184: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(cat_4, primals_742, primals_743, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_743 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_185: "f32[4, 80, 4, 4]" = torch.ops.aten.convolution.default(cat_4, primals_744, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_186: "f32[4, 1280, 4, 4]" = torch.ops.aten.convolution.default(convolution_185, primals_745, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_397: "f32[4, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(convolution_186, 1.0);  convolution_186 = None
        add_384: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(convolution_184, mul_397);  convolution_184 = mul_397 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_385: "f32[4, 1280, 4, 4]" = torch.ops.aten.add.Tensor(add_384, add_383);  add_384 = add_383 = None
        div_38: "f32[4, 1280, 4, 4]" = torch.ops.aten.div.Tensor(add_385, 1.0);  add_385 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota_1: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_398: "i64[8]" = torch.ops.aten.mul.Tensor(iota_1, 1);  iota_1 = None
        add_386: "i64[8]" = torch.ops.aten.add.Tensor(mul_398, 0);  mul_398 = None
        convert_element_type_2: "f32[8]" = torch.ops.prims.convert_element_type.default(add_386, torch.float32);  add_386 = None
        add_387: "f32[8]" = torch.ops.aten.add.Tensor(convert_element_type_2, 0.0);  convert_element_type_2 = None
        mul_399: "f32[8]" = torch.ops.aten.mul.Tensor(add_387, 0.5);  add_387 = None
        convert_element_type_3: "i64[8]" = torch.ops.prims.convert_element_type.default(mul_399, torch.int64);  mul_399 = None
        unsqueeze_356: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1)
        _unsafe_index: "f32[4, 1280, 8, 8]" = torch.ops.aten._unsafe_index.Tensor(div_38, [None, None, unsqueeze_356, convert_element_type_3]);  div_38 = unsqueeze_356 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_187: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(_unsafe_index, primals_746, primals_747, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_747 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_188: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(_unsafe_index, primals_748, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_189: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_188, primals_749, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_402: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_189, 1.0);  convolution_189 = None
        add_390: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_187, mul_402);  convolution_187 = mul_402 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_5: "f32[4, 2560, 8, 8]" = torch.ops.aten.cat.default([add_390, add_299], 1);  add_390 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1031: "f32[4, 32, 80, 64]" = torch.ops.aten.view.default(cat_5, [4, 32, 80, 64])
        var_mean_76 = torch.ops.aten.var_mean.correction(view_1031, [2, 3], correction = 0, keepdim = True)
        getitem_256: "f32[4, 32, 1, 1]" = var_mean_76[0]
        getitem_257: "f32[4, 32, 1, 1]" = var_mean_76[1];  var_mean_76 = None
        add_391: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-05);  getitem_256 = None
        rsqrt_76: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
        sub_76: "f32[4, 32, 80, 64]" = torch.ops.aten.sub.Tensor(view_1031, getitem_257);  view_1031 = None
        mul_403: "f32[4, 32, 80, 64]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        view_1032: "f32[4, 2560, 8, 8]" = torch.ops.aten.view.default(mul_403, [4, 2560, 8, 8]);  mul_403 = None
        unsqueeze_357: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_751, 0)
        unsqueeze_358: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
        unsqueeze_359: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_750, 0)
        unsqueeze_361: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
        unsqueeze_362: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
        mul_404: "f32[4, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(view_1032, unsqueeze_362);  view_1032 = unsqueeze_362 = None
        add_392: "f32[4, 2560, 8, 8]" = torch.ops.aten.add.Tensor(mul_404, unsqueeze_359);  mul_404 = unsqueeze_359 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_61: "f32[4, 2560, 8, 8]" = torch.ops.aten.sigmoid.default(add_392)
        mul_405: "f32[4, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(add_392, sigmoid_61);  add_392 = sigmoid_61 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_190: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_405, primals_752, primals_753, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_753 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_191: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_405, primals_754, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_192: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_191, primals_755, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_406: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_192, 1.0);  convolution_192 = None
        add_393: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_190, mul_406);  convolution_190 = mul_406 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_409: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_756, [1, 0]);  primals_756 = None
        addmm_51: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_757, mul_109, permute_409);  primals_757 = permute_409 = None
        unsqueeze_363: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_51, 2);  addmm_51 = None
        unsqueeze_364: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_394: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_393, unsqueeze_364);  add_393 = unsqueeze_364 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1033: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_394, [4, 32, 40, 64])
        var_mean_77 = torch.ops.aten.var_mean.correction(view_1033, [2, 3], correction = 0, keepdim = True)
        getitem_258: "f32[4, 32, 1, 1]" = var_mean_77[0]
        getitem_259: "f32[4, 32, 1, 1]" = var_mean_77[1];  var_mean_77 = None
        add_395: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_77: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
        sub_77: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1033, getitem_259);  view_1033 = None
        mul_408: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
        view_1034: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_408, [4, 1280, 8, 8]);  mul_408 = None
        unsqueeze_365: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_759, 0)
        unsqueeze_366: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        unsqueeze_368: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_758, 0)
        unsqueeze_369: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        mul_409: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1034, unsqueeze_370);  view_1034 = unsqueeze_370 = None
        add_396: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_367);  mul_409 = unsqueeze_367 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_63: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_396)
        mul_410: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_396, sigmoid_63);  add_396 = sigmoid_63 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_193: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_410, primals_760, primals_761, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_761 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_194: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_410, primals_762, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_195: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_194, primals_763, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_411: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_195, 1.0);  convolution_195 = None
        add_397: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_193, mul_411);  convolution_193 = mul_411 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_196: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(cat_5, primals_764, primals_765, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_765 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_197: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(cat_5, primals_766, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_198: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_197, primals_767, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_412: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_198, 1.0);  convolution_198 = None
        add_398: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_196, mul_412);  convolution_196 = mul_412 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_399: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_398, add_397);  add_398 = add_397 = None
        div_39: "f32[4, 1280, 8, 8]" = torch.ops.aten.div.Tensor(add_399, 1.0);  add_399 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1035: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(div_39, [4, 32, 40, 64])
        var_mean_78 = torch.ops.aten.var_mean.correction(view_1035, [2, 3], correction = 0, keepdim = True)
        getitem_260: "f32[4, 32, 1, 1]" = var_mean_78[0]
        getitem_261: "f32[4, 32, 1, 1]" = var_mean_78[1];  var_mean_78 = None
        add_400: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-06);  getitem_260 = None
        rsqrt_78: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_400);  add_400 = None
        sub_78: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1035, getitem_261);  view_1035 = None
        mul_413: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        view_1036: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_413, [4, 1280, 8, 8]);  mul_413 = None
        unsqueeze_371: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_769, 0);  primals_769 = None
        unsqueeze_372: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        unsqueeze_374: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_768, 0)
        unsqueeze_375: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_414: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1036, unsqueeze_376);  view_1036 = unsqueeze_376 = None
        add_401: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_373);  mul_414 = unsqueeze_373 = None
        squeeze_114: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_261, [2, 3]);  getitem_261 = None
        squeeze_115: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_78, [2, 3]);  rsqrt_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_410: "f32[4, 8, 8, 1280]" = torch.ops.aten.permute.default(add_401, [0, 2, 3, 1]);  add_401 = None
        view_1037: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_410, [4, 64, 1280]);  permute_410 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_411: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_770, [1, 0])
        expand_21: "f32[4, 64, 1280]" = torch.ops.aten.expand.default(view_1037, [4, 64, 1280])
        expand_22: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_411, [4, 1280, 1280]);  permute_411 = None
        bmm_10: "f32[4, 64, 1280]" = torch.ops.aten.bmm.default(expand_21, expand_22);  expand_21 = expand_22 = None
        add_402: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(bmm_10, primals_771);  bmm_10 = primals_771 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_412: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_772, [1, 0]);  primals_772 = None
        clone_65: "f32[4, 64, 1280]" = torch.ops.aten.clone.default(view_1037, memory_format = torch.contiguous_format);  view_1037 = None
        view_1041: "f32[256, 1280]" = torch.ops.aten.view.default(clone_65, [256, 1280]);  clone_65 = None
        mm_218: "f32[256, 80]" = torch.ops.aten.mm.default(view_1041, permute_412)
        permute_413: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_773, [1, 0]);  primals_773 = None
        mm_219: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_218, permute_413)
        view_1044: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_219, [4, 64, 1280]);  mm_219 = None
        mul_415: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1044, 1.0);  view_1044 = None
        add_403: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_402, mul_415);  add_402 = mul_415 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_79 = torch.ops.aten.var_mean.correction(add_403, [2], correction = 0, keepdim = True)
        getitem_262: "f32[4, 64, 1]" = var_mean_79[0]
        getitem_263: "f32[4, 64, 1]" = var_mean_79[1];  var_mean_79 = None
        add_404: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_79: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        sub_79: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_403, getitem_263);  getitem_263 = None
        mul_416: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        mul_417: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_416, primals_774)
        add_405: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_417, primals_775);  mul_417 = primals_775 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_414: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_776, [1, 0]);  primals_776 = None
        view_1045: "f32[256, 1280]" = torch.ops.aten.view.default(add_405, [256, 1280]);  add_405 = None
        mm_220: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1045, permute_414)
        view_1046: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_220, [4, 64, 1280]);  mm_220 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_415: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_777, [1, 0]);  primals_777 = None
        mm_221: "f32[256, 80]" = torch.ops.aten.mm.default(view_1045, permute_415)
        permute_416: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_778, [1, 0]);  primals_778 = None
        mm_222: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_221, permute_416)
        view_1050: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_222, [4, 64, 1280]);  mm_222 = None
        mul_418: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1050, 1.0);  view_1050 = None
        add_406: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1046, mul_418);  view_1046 = mul_418 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_417: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_779, [1, 0]);  primals_779 = None
        mm_223: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1045, permute_417)
        view_1054: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_223, [4, 64, 1280]);  mm_223 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_418: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_780, [1, 0]);  primals_780 = None
        mm_224: "f32[256, 80]" = torch.ops.aten.mm.default(view_1045, permute_418)
        permute_419: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_781, [1, 0]);  primals_781 = None
        mm_225: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_224, permute_419)
        view_1058: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_225, [4, 64, 1280]);  mm_225 = None
        mul_419: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1058, 1.0);  view_1058 = None
        add_407: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1054, mul_419);  view_1054 = mul_419 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_420: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_782, [1, 0]);  primals_782 = None
        mm_226: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1045, permute_420)
        view_1062: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_226, [4, 64, 1280]);  mm_226 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_421: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_783, [1, 0]);  primals_783 = None
        mm_227: "f32[256, 80]" = torch.ops.aten.mm.default(view_1045, permute_421)
        permute_422: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_784, [1, 0]);  primals_784 = None
        mm_228: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_227, permute_422)
        view_1066: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_228, [4, 64, 1280]);  mm_228 = None
        mul_420: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1066, 1.0);  view_1066 = None
        add_408: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1062, mul_420);  view_1062 = mul_420 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1073: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_406, [4, -1, 20, 64]);  add_406 = None
        permute_426: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None
        view_1075: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_407, [4, -1, 20, 64]);  add_407 = None
        permute_427: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1075, [0, 2, 1, 3]);  view_1075 = None
        view_1077: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_408, [4, -1, 20, 64]);  add_408 = None
        permute_428: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1077, [0, 2, 1, 3]);  view_1077 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_426, permute_427, permute_428, None, True)
        getitem_264: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_15[0]
        getitem_265: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_15[1]
        getitem_266: "i64[]" = _scaled_dot_product_efficient_attention_15[2]
        getitem_267: "i64[]" = _scaled_dot_product_efficient_attention_15[3];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_429: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_264, [0, 2, 1, 3])
        view_1078: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_429, [4, -1, 1280]);  permute_429 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1079: "f32[256, 1280]" = torch.ops.aten.view.default(view_1078, [256, 1280]);  view_1078 = None
        permute_430: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_785, [1, 0]);  primals_785 = None
        addmm_52: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_786, view_1079, permute_430);  primals_786 = None
        view_1080: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_52, [4, 64, 1280]);  addmm_52 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_431: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_787, [1, 0]);  primals_787 = None
        mm_229: "f32[256, 80]" = torch.ops.aten.mm.default(view_1079, permute_431);  view_1079 = None
        permute_432: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_788, [1, 0]);  primals_788 = None
        mm_230: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_229, permute_432)
        view_1084: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_230, [4, 64, 1280]);  mm_230 = None
        mul_421: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1084, 1.0);  view_1084 = None
        add_409: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1080, mul_421);  view_1080 = mul_421 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_40: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_409, 1.0);  add_409 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_410: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_40, add_403);  div_40 = add_403 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_80 = torch.ops.aten.var_mean.correction(add_410, [2], correction = 0, keepdim = True)
        getitem_268: "f32[4, 64, 1]" = var_mean_80[0]
        getitem_269: "f32[4, 64, 1]" = var_mean_80[1];  var_mean_80 = None
        add_411: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_80: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        sub_80: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_410, getitem_269);  getitem_269 = None
        mul_422: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
        mul_423: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_422, primals_789)
        add_412: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_423, primals_790);  mul_423 = primals_790 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_433: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_791, [1, 0]);  primals_791 = None
        view_1088: "f32[256, 1280]" = torch.ops.aten.view.default(add_412, [256, 1280]);  add_412 = None
        mm_231: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1088, permute_433)
        view_1089: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_231, [4, 64, 1280]);  mm_231 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_434: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_792, [1, 0]);  primals_792 = None
        mm_232: "f32[256, 80]" = torch.ops.aten.mm.default(view_1088, permute_434)
        permute_435: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_793, [1, 0]);  primals_793 = None
        mm_233: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_232, permute_435)
        view_1093: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_233, [4, 64, 1280]);  mm_233 = None
        mul_424: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1093, 1.0);  view_1093 = None
        add_413: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1089, mul_424);  view_1089 = mul_424 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_436: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_794, [1, 0]);  primals_794 = None
        mm_234: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_436);  permute_436 = None
        view_1097: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_234, [4, 77, 1280]);  mm_234 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_437: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_795, [1, 0]);  primals_795 = None
        mm_235: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_437);  permute_437 = None
        permute_438: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_796, [1, 0]);  primals_796 = None
        mm_236: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_235, permute_438)
        view_1101: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_236, [4, 77, 1280]);  mm_236 = None
        mul_425: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1101, 1.0);  view_1101 = None
        add_414: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1097, mul_425);  view_1097 = mul_425 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_439: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_797, [1, 0]);  primals_797 = None
        mm_237: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_439);  permute_439 = None
        view_1105: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_237, [4, 77, 1280]);  mm_237 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_440: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_798, [1, 0]);  primals_798 = None
        mm_238: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_440);  permute_440 = None
        permute_441: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_799, [1, 0]);  primals_799 = None
        mm_239: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_238, permute_441)
        view_1109: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_239, [4, 77, 1280]);  mm_239 = None
        mul_426: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1109, 1.0);  view_1109 = None
        add_415: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1105, mul_426);  view_1105 = mul_426 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1116: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_413, [4, -1, 20, 64]);  add_413 = None
        permute_445: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1116, [0, 2, 1, 3]);  view_1116 = None
        view_1118: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_414, [4, -1, 20, 64]);  add_414 = None
        permute_446: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1118, [0, 2, 1, 3]);  view_1118 = None
        view_1120: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_415, [4, -1, 20, 64]);  add_415 = None
        permute_447: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1120, [0, 2, 1, 3]);  view_1120 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_445, permute_446, permute_447, None, True)
        getitem_270: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_16[0]
        getitem_271: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_16[1]
        getitem_272: "i64[]" = _scaled_dot_product_efficient_attention_16[2]
        getitem_273: "i64[]" = _scaled_dot_product_efficient_attention_16[3];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_448: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_270, [0, 2, 1, 3])
        view_1121: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_448, [4, -1, 1280]);  permute_448 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1122: "f32[256, 1280]" = torch.ops.aten.view.default(view_1121, [256, 1280]);  view_1121 = None
        permute_449: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_800, [1, 0]);  primals_800 = None
        addmm_53: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_801, view_1122, permute_449);  primals_801 = None
        view_1123: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_53, [4, 64, 1280]);  addmm_53 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_450: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_802, [1, 0]);  primals_802 = None
        mm_240: "f32[256, 80]" = torch.ops.aten.mm.default(view_1122, permute_450);  view_1122 = None
        permute_451: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_803, [1, 0]);  primals_803 = None
        mm_241: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_240, permute_451)
        view_1127: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_241, [4, 64, 1280]);  mm_241 = None
        mul_427: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1127, 1.0);  view_1127 = None
        add_416: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1123, mul_427);  view_1123 = mul_427 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_41: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_416, 1.0);  add_416 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_417: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_41, add_410);  div_41 = add_410 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_81 = torch.ops.aten.var_mean.correction(add_417, [2], correction = 0, keepdim = True)
        getitem_274: "f32[4, 64, 1]" = var_mean_81[0]
        getitem_275: "f32[4, 64, 1]" = var_mean_81[1];  var_mean_81 = None
        add_418: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-05);  getitem_274 = None
        rsqrt_81: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_81: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_417, getitem_275);  getitem_275 = None
        mul_428: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
        mul_429: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_428, primals_804)
        add_419: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_429, primals_805);  mul_429 = primals_805 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1131: "f32[256, 1280]" = torch.ops.aten.view.default(add_419, [256, 1280]);  add_419 = None
        permute_452: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_806, [1, 0]);  primals_806 = None
        addmm_54: "f32[256, 10240]" = torch.ops.aten.addmm.default(primals_807, view_1131, permute_452);  primals_807 = None
        view_1132: "f32[4, 64, 10240]" = torch.ops.aten.view.default(addmm_54, [4, 64, 10240]);  addmm_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_453: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_808, [1, 0]);  primals_808 = None
        mm_242: "f32[256, 80]" = torch.ops.aten.mm.default(view_1131, permute_453)
        permute_454: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_809, [1, 0]);  primals_809 = None
        mm_243: "f32[256, 10240]" = torch.ops.aten.mm.default(mm_242, permute_454)
        view_1136: "f32[4, 64, 10240]" = torch.ops.aten.view.default(mm_243, [4, 64, 10240]);  mm_243 = None
        mul_430: "f32[4, 64, 10240]" = torch.ops.aten.mul.Tensor(view_1136, 1.0);  view_1136 = None
        add_420: "f32[4, 64, 10240]" = torch.ops.aten.add.Tensor(view_1132, mul_430);  view_1132 = mul_430 = None
        view_1137: "f32[256, 10240]" = torch.ops.aten.view.default(add_420, [256, 10240]);  add_420 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1140: "f32[4, 64, 10240]" = torch.ops.aten.view.default(view_1137, [4, 64, 10240]);  view_1137 = None
        split_23 = torch.ops.aten.split.Tensor(view_1140, 5120, -1);  view_1140 = None
        getitem_279: "f32[4, 64, 5120]" = split_23[1]
        mul_431: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_279, 0.5)
        mul_432: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_279, 0.7071067811865476)
        erf_7: "f32[4, 64, 5120]" = torch.ops.aten.erf.default(mul_432);  mul_432 = None
        add_421: "f32[4, 64, 5120]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_433: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_431, add_421);  mul_431 = add_421 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_280: "f32[4, 64, 5120]" = split_23[0];  split_23 = None
        mul_434: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_280, mul_433);  mul_433 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1142: "f32[256, 5120]" = torch.ops.aten.view.default(mul_434, [256, 5120]);  mul_434 = None
        permute_455: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_810, [1, 0]);  primals_810 = None
        addmm_55: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_811, view_1142, permute_455);  primals_811 = None
        view_1143: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_55, [4, 64, 1280]);  addmm_55 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_456: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_812, [1, 0]);  primals_812 = None
        mm_244: "f32[256, 80]" = torch.ops.aten.mm.default(view_1142, permute_456)
        permute_457: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_813, [1, 0]);  primals_813 = None
        mm_245: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_244, permute_457)
        view_1147: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_245, [4, 64, 1280]);  mm_245 = None
        mul_435: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1147, 1.0);  view_1147 = None
        add_422: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1143, mul_435);  view_1143 = mul_435 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_423: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_422, add_417);  add_422 = add_417 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1151: "f32[256, 1280]" = torch.ops.aten.view.default(add_423, [256, 1280]);  add_423 = None
        permute_458: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_814, [1, 0]);  primals_814 = None
        addmm_56: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_815, view_1151, permute_458);  primals_815 = None
        view_1152: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_56, [4, 64, 1280]);  addmm_56 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_459: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_816, [1, 0]);  primals_816 = None
        mm_246: "f32[256, 80]" = torch.ops.aten.mm.default(view_1151, permute_459)
        permute_460: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_817, [1, 0]);  primals_817 = None
        mm_247: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_246, permute_460)
        view_1156: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_247, [4, 64, 1280]);  mm_247 = None
        mul_436: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1156, 1.0);  view_1156 = None
        add_424: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1152, mul_436);  view_1152 = mul_436 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1162: "f32[4, 8, 8, 1280]" = torch.ops.aten.view.default(add_424, [4, 8, 8, 1280]);  add_424 = None
        permute_462: "f32[4, 1280, 8, 8]" = torch.ops.aten.permute.default(view_1162, [0, 3, 1, 2]);  view_1162 = None
        clone_69: "f32[4, 1280, 8, 8]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_425: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(clone_69, div_39);  clone_69 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_6: "f32[4, 2560, 8, 8]" = torch.ops.aten.cat.default([add_425, add_265], 1);  add_425 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1163: "f32[4, 32, 80, 64]" = torch.ops.aten.view.default(cat_6, [4, 32, 80, 64])
        var_mean_82 = torch.ops.aten.var_mean.correction(view_1163, [2, 3], correction = 0, keepdim = True)
        getitem_282: "f32[4, 32, 1, 1]" = var_mean_82[0]
        getitem_283: "f32[4, 32, 1, 1]" = var_mean_82[1];  var_mean_82 = None
        add_426: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_82: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        sub_82: "f32[4, 32, 80, 64]" = torch.ops.aten.sub.Tensor(view_1163, getitem_283);  view_1163 = None
        mul_437: "f32[4, 32, 80, 64]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        view_1164: "f32[4, 2560, 8, 8]" = torch.ops.aten.view.default(mul_437, [4, 2560, 8, 8]);  mul_437 = None
        unsqueeze_377: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_819, 0)
        unsqueeze_378: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        unsqueeze_380: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(primals_818, 0)
        unsqueeze_381: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        mul_438: "f32[4, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(view_1164, unsqueeze_382);  view_1164 = unsqueeze_382 = None
        add_427: "f32[4, 2560, 8, 8]" = torch.ops.aten.add.Tensor(mul_438, unsqueeze_379);  mul_438 = unsqueeze_379 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_64: "f32[4, 2560, 8, 8]" = torch.ops.aten.sigmoid.default(add_427)
        mul_439: "f32[4, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(add_427, sigmoid_64);  add_427 = sigmoid_64 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_199: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_439, primals_820, primals_821, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_821 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_200: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_439, primals_822, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_201: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_200, primals_823, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_440: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_201, 1.0);  convolution_201 = None
        add_428: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_199, mul_440);  convolution_199 = mul_440 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_463: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_824, [1, 0]);  primals_824 = None
        addmm_57: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_825, mul_109, permute_463);  primals_825 = permute_463 = None
        unsqueeze_383: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_57, 2);  addmm_57 = None
        unsqueeze_384: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_429: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_428, unsqueeze_384);  add_428 = unsqueeze_384 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1165: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_429, [4, 32, 40, 64])
        var_mean_83 = torch.ops.aten.var_mean.correction(view_1165, [2, 3], correction = 0, keepdim = True)
        getitem_284: "f32[4, 32, 1, 1]" = var_mean_83[0]
        getitem_285: "f32[4, 32, 1, 1]" = var_mean_83[1];  var_mean_83 = None
        add_430: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_83: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_430);  add_430 = None
        sub_83: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1165, getitem_285);  view_1165 = None
        mul_442: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        view_1166: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_442, [4, 1280, 8, 8]);  mul_442 = None
        unsqueeze_385: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_827, 0)
        unsqueeze_386: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_826, 0)
        unsqueeze_389: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        mul_443: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1166, unsqueeze_390);  view_1166 = unsqueeze_390 = None
        add_431: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_387);  mul_443 = unsqueeze_387 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_66: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_431)
        mul_444: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_431, sigmoid_66);  add_431 = sigmoid_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_202: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_444, primals_828, primals_829, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_829 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_203: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_444, primals_830, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_204: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_203, primals_831, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_445: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_204, 1.0);  convolution_204 = None
        add_432: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_202, mul_445);  convolution_202 = mul_445 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_205: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(cat_6, primals_832, primals_833, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_833 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_206: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(cat_6, primals_834, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_207: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_206, primals_835, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_446: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_207, 1.0);  convolution_207 = None
        add_433: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_205, mul_446);  convolution_205 = mul_446 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_434: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_433, add_432);  add_433 = add_432 = None
        div_42: "f32[4, 1280, 8, 8]" = torch.ops.aten.div.Tensor(add_434, 1.0);  add_434 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1167: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(div_42, [4, 32, 40, 64])
        var_mean_84 = torch.ops.aten.var_mean.correction(view_1167, [2, 3], correction = 0, keepdim = True)
        getitem_286: "f32[4, 32, 1, 1]" = var_mean_84[0]
        getitem_287: "f32[4, 32, 1, 1]" = var_mean_84[1];  var_mean_84 = None
        add_435: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_84: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_435);  add_435 = None
        sub_84: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1167, getitem_287);  view_1167 = None
        mul_447: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        view_1168: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_447, [4, 1280, 8, 8]);  mul_447 = None
        unsqueeze_391: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_837, 0);  primals_837 = None
        unsqueeze_392: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_836, 0)
        unsqueeze_395: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_448: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1168, unsqueeze_396);  view_1168 = unsqueeze_396 = None
        add_436: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_448, unsqueeze_393);  mul_448 = unsqueeze_393 = None
        squeeze_120: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_287, [2, 3]);  getitem_287 = None
        squeeze_121: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_84, [2, 3]);  rsqrt_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_464: "f32[4, 8, 8, 1280]" = torch.ops.aten.permute.default(add_436, [0, 2, 3, 1]);  add_436 = None
        view_1169: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_464, [4, 64, 1280]);  permute_464 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_465: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_838, [1, 0])
        expand_23: "f32[4, 64, 1280]" = torch.ops.aten.expand.default(view_1169, [4, 64, 1280])
        expand_24: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_465, [4, 1280, 1280]);  permute_465 = None
        bmm_11: "f32[4, 64, 1280]" = torch.ops.aten.bmm.default(expand_23, expand_24);  expand_23 = expand_24 = None
        add_437: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(bmm_11, primals_839);  bmm_11 = primals_839 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_466: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_840, [1, 0]);  primals_840 = None
        clone_71: "f32[4, 64, 1280]" = torch.ops.aten.clone.default(view_1169, memory_format = torch.contiguous_format);  view_1169 = None
        view_1173: "f32[256, 1280]" = torch.ops.aten.view.default(clone_71, [256, 1280]);  clone_71 = None
        mm_248: "f32[256, 80]" = torch.ops.aten.mm.default(view_1173, permute_466)
        permute_467: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_841, [1, 0]);  primals_841 = None
        mm_249: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_248, permute_467)
        view_1176: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_249, [4, 64, 1280]);  mm_249 = None
        mul_449: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1176, 1.0);  view_1176 = None
        add_438: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_437, mul_449);  add_437 = mul_449 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_85 = torch.ops.aten.var_mean.correction(add_438, [2], correction = 0, keepdim = True)
        getitem_288: "f32[4, 64, 1]" = var_mean_85[0]
        getitem_289: "f32[4, 64, 1]" = var_mean_85[1];  var_mean_85 = None
        add_439: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_85: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
        sub_85: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_438, getitem_289);  getitem_289 = None
        mul_450: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        mul_451: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_450, primals_842)
        add_440: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_451, primals_843);  mul_451 = primals_843 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_468: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_844, [1, 0]);  primals_844 = None
        view_1177: "f32[256, 1280]" = torch.ops.aten.view.default(add_440, [256, 1280]);  add_440 = None
        mm_250: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1177, permute_468)
        view_1178: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_250, [4, 64, 1280]);  mm_250 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_469: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_845, [1, 0]);  primals_845 = None
        mm_251: "f32[256, 80]" = torch.ops.aten.mm.default(view_1177, permute_469)
        permute_470: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_846, [1, 0]);  primals_846 = None
        mm_252: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_251, permute_470)
        view_1182: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_252, [4, 64, 1280]);  mm_252 = None
        mul_452: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1182, 1.0);  view_1182 = None
        add_441: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1178, mul_452);  view_1178 = mul_452 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_471: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_847, [1, 0]);  primals_847 = None
        mm_253: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1177, permute_471)
        view_1186: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_253, [4, 64, 1280]);  mm_253 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_472: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_848, [1, 0]);  primals_848 = None
        mm_254: "f32[256, 80]" = torch.ops.aten.mm.default(view_1177, permute_472)
        permute_473: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_849, [1, 0]);  primals_849 = None
        mm_255: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_254, permute_473)
        view_1190: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_255, [4, 64, 1280]);  mm_255 = None
        mul_453: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1190, 1.0);  view_1190 = None
        add_442: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1186, mul_453);  view_1186 = mul_453 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_474: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_850, [1, 0]);  primals_850 = None
        mm_256: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1177, permute_474)
        view_1194: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_256, [4, 64, 1280]);  mm_256 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_475: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_851, [1, 0]);  primals_851 = None
        mm_257: "f32[256, 80]" = torch.ops.aten.mm.default(view_1177, permute_475)
        permute_476: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_852, [1, 0]);  primals_852 = None
        mm_258: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_257, permute_476)
        view_1198: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_258, [4, 64, 1280]);  mm_258 = None
        mul_454: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1198, 1.0);  view_1198 = None
        add_443: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1194, mul_454);  view_1194 = mul_454 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1205: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_441, [4, -1, 20, 64]);  add_441 = None
        permute_480: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1205, [0, 2, 1, 3]);  view_1205 = None
        view_1207: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_442, [4, -1, 20, 64]);  add_442 = None
        permute_481: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1207, [0, 2, 1, 3]);  view_1207 = None
        view_1209: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_443, [4, -1, 20, 64]);  add_443 = None
        permute_482: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1209, [0, 2, 1, 3]);  view_1209 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_480, permute_481, permute_482, None, True)
        getitem_290: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_17[0]
        getitem_291: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_17[1]
        getitem_292: "i64[]" = _scaled_dot_product_efficient_attention_17[2]
        getitem_293: "i64[]" = _scaled_dot_product_efficient_attention_17[3];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_483: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_290, [0, 2, 1, 3])
        view_1210: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_483, [4, -1, 1280]);  permute_483 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1211: "f32[256, 1280]" = torch.ops.aten.view.default(view_1210, [256, 1280]);  view_1210 = None
        permute_484: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_853, [1, 0]);  primals_853 = None
        addmm_58: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_854, view_1211, permute_484);  primals_854 = None
        view_1212: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_58, [4, 64, 1280]);  addmm_58 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_485: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_855, [1, 0]);  primals_855 = None
        mm_259: "f32[256, 80]" = torch.ops.aten.mm.default(view_1211, permute_485);  view_1211 = None
        permute_486: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_856, [1, 0]);  primals_856 = None
        mm_260: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_259, permute_486)
        view_1216: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_260, [4, 64, 1280]);  mm_260 = None
        mul_455: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1216, 1.0);  view_1216 = None
        add_444: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1212, mul_455);  view_1212 = mul_455 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_43: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_444, 1.0);  add_444 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_445: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_43, add_438);  div_43 = add_438 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_86 = torch.ops.aten.var_mean.correction(add_445, [2], correction = 0, keepdim = True)
        getitem_294: "f32[4, 64, 1]" = var_mean_86[0]
        getitem_295: "f32[4, 64, 1]" = var_mean_86[1];  var_mean_86 = None
        add_446: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_86: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_86: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_445, getitem_295);  getitem_295 = None
        mul_456: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
        mul_457: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_456, primals_857)
        add_447: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_457, primals_858);  mul_457 = primals_858 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_487: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_859, [1, 0]);  primals_859 = None
        view_1220: "f32[256, 1280]" = torch.ops.aten.view.default(add_447, [256, 1280]);  add_447 = None
        mm_261: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1220, permute_487)
        view_1221: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_261, [4, 64, 1280]);  mm_261 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_488: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_860, [1, 0]);  primals_860 = None
        mm_262: "f32[256, 80]" = torch.ops.aten.mm.default(view_1220, permute_488)
        permute_489: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_861, [1, 0]);  primals_861 = None
        mm_263: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_262, permute_489)
        view_1225: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_263, [4, 64, 1280]);  mm_263 = None
        mul_458: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1225, 1.0);  view_1225 = None
        add_448: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1221, mul_458);  view_1221 = mul_458 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_490: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_862, [1, 0]);  primals_862 = None
        mm_264: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_490);  permute_490 = None
        view_1229: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_264, [4, 77, 1280]);  mm_264 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_491: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_863, [1, 0]);  primals_863 = None
        mm_265: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_491);  permute_491 = None
        permute_492: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_864, [1, 0]);  primals_864 = None
        mm_266: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_265, permute_492)
        view_1233: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_266, [4, 77, 1280]);  mm_266 = None
        mul_459: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1233, 1.0);  view_1233 = None
        add_449: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1229, mul_459);  view_1229 = mul_459 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_493: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_865, [1, 0]);  primals_865 = None
        mm_267: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_493);  permute_493 = None
        view_1237: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_267, [4, 77, 1280]);  mm_267 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_494: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_866, [1, 0]);  primals_866 = None
        mm_268: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_494);  permute_494 = None
        permute_495: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_867, [1, 0]);  primals_867 = None
        mm_269: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_268, permute_495)
        view_1241: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_269, [4, 77, 1280]);  mm_269 = None
        mul_460: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1241, 1.0);  view_1241 = None
        add_450: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1237, mul_460);  view_1237 = mul_460 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1248: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_448, [4, -1, 20, 64]);  add_448 = None
        permute_499: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1248, [0, 2, 1, 3]);  view_1248 = None
        view_1250: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_449, [4, -1, 20, 64]);  add_449 = None
        permute_500: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1250, [0, 2, 1, 3]);  view_1250 = None
        view_1252: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_450, [4, -1, 20, 64]);  add_450 = None
        permute_501: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1252, [0, 2, 1, 3]);  view_1252 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_499, permute_500, permute_501, None, True)
        getitem_296: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_18[0]
        getitem_297: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_18[1]
        getitem_298: "i64[]" = _scaled_dot_product_efficient_attention_18[2]
        getitem_299: "i64[]" = _scaled_dot_product_efficient_attention_18[3];  _scaled_dot_product_efficient_attention_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_502: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_296, [0, 2, 1, 3])
        view_1253: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_502, [4, -1, 1280]);  permute_502 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1254: "f32[256, 1280]" = torch.ops.aten.view.default(view_1253, [256, 1280]);  view_1253 = None
        permute_503: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_868, [1, 0]);  primals_868 = None
        addmm_59: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_869, view_1254, permute_503);  primals_869 = None
        view_1255: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_59, [4, 64, 1280]);  addmm_59 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_504: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_870, [1, 0]);  primals_870 = None
        mm_270: "f32[256, 80]" = torch.ops.aten.mm.default(view_1254, permute_504);  view_1254 = None
        permute_505: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_871, [1, 0]);  primals_871 = None
        mm_271: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_270, permute_505)
        view_1259: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_271, [4, 64, 1280]);  mm_271 = None
        mul_461: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1259, 1.0);  view_1259 = None
        add_451: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1255, mul_461);  view_1255 = mul_461 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_44: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_451, 1.0);  add_451 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_452: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_44, add_445);  div_44 = add_445 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_87 = torch.ops.aten.var_mean.correction(add_452, [2], correction = 0, keepdim = True)
        getitem_300: "f32[4, 64, 1]" = var_mean_87[0]
        getitem_301: "f32[4, 64, 1]" = var_mean_87[1];  var_mean_87 = None
        add_453: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_300, 1e-05);  getitem_300 = None
        rsqrt_87: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_453);  add_453 = None
        sub_87: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_452, getitem_301);  getitem_301 = None
        mul_462: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        mul_463: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_462, primals_872)
        add_454: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_463, primals_873);  mul_463 = primals_873 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1263: "f32[256, 1280]" = torch.ops.aten.view.default(add_454, [256, 1280]);  add_454 = None
        permute_506: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_874, [1, 0]);  primals_874 = None
        addmm_60: "f32[256, 10240]" = torch.ops.aten.addmm.default(primals_875, view_1263, permute_506);  primals_875 = None
        view_1264: "f32[4, 64, 10240]" = torch.ops.aten.view.default(addmm_60, [4, 64, 10240]);  addmm_60 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_507: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_876, [1, 0]);  primals_876 = None
        mm_272: "f32[256, 80]" = torch.ops.aten.mm.default(view_1263, permute_507)
        permute_508: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_877, [1, 0]);  primals_877 = None
        mm_273: "f32[256, 10240]" = torch.ops.aten.mm.default(mm_272, permute_508)
        view_1268: "f32[4, 64, 10240]" = torch.ops.aten.view.default(mm_273, [4, 64, 10240]);  mm_273 = None
        mul_464: "f32[4, 64, 10240]" = torch.ops.aten.mul.Tensor(view_1268, 1.0);  view_1268 = None
        add_455: "f32[4, 64, 10240]" = torch.ops.aten.add.Tensor(view_1264, mul_464);  view_1264 = mul_464 = None
        view_1269: "f32[256, 10240]" = torch.ops.aten.view.default(add_455, [256, 10240]);  add_455 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1272: "f32[4, 64, 10240]" = torch.ops.aten.view.default(view_1269, [4, 64, 10240]);  view_1269 = None
        split_26 = torch.ops.aten.split.Tensor(view_1272, 5120, -1);  view_1272 = None
        getitem_305: "f32[4, 64, 5120]" = split_26[1]
        mul_465: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_305, 0.5)
        mul_466: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_305, 0.7071067811865476)
        erf_8: "f32[4, 64, 5120]" = torch.ops.aten.erf.default(mul_466);  mul_466 = None
        add_456: "f32[4, 64, 5120]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_467: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_465, add_456);  mul_465 = add_456 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_306: "f32[4, 64, 5120]" = split_26[0];  split_26 = None
        mul_468: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_306, mul_467);  mul_467 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1274: "f32[256, 5120]" = torch.ops.aten.view.default(mul_468, [256, 5120]);  mul_468 = None
        permute_509: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_878, [1, 0]);  primals_878 = None
        addmm_61: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_879, view_1274, permute_509);  primals_879 = None
        view_1275: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_61, [4, 64, 1280]);  addmm_61 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_510: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_880, [1, 0]);  primals_880 = None
        mm_274: "f32[256, 80]" = torch.ops.aten.mm.default(view_1274, permute_510)
        permute_511: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_881, [1, 0]);  primals_881 = None
        mm_275: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_274, permute_511)
        view_1279: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_275, [4, 64, 1280]);  mm_275 = None
        mul_469: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1279, 1.0);  view_1279 = None
        add_457: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1275, mul_469);  view_1275 = mul_469 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_458: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_457, add_452);  add_457 = add_452 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1283: "f32[256, 1280]" = torch.ops.aten.view.default(add_458, [256, 1280]);  add_458 = None
        permute_512: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_882, [1, 0]);  primals_882 = None
        addmm_62: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_883, view_1283, permute_512);  primals_883 = None
        view_1284: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_62, [4, 64, 1280]);  addmm_62 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_513: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_884, [1, 0]);  primals_884 = None
        mm_276: "f32[256, 80]" = torch.ops.aten.mm.default(view_1283, permute_513)
        permute_514: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_885, [1, 0]);  primals_885 = None
        mm_277: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_276, permute_514)
        view_1288: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_277, [4, 64, 1280]);  mm_277 = None
        mul_470: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1288, 1.0);  view_1288 = None
        add_459: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1284, mul_470);  view_1284 = mul_470 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1294: "f32[4, 8, 8, 1280]" = torch.ops.aten.view.default(add_459, [4, 8, 8, 1280]);  add_459 = None
        permute_516: "f32[4, 1280, 8, 8]" = torch.ops.aten.permute.default(view_1294, [0, 3, 1, 2]);  view_1294 = None
        clone_75: "f32[4, 1280, 8, 8]" = torch.ops.aten.clone.default(permute_516, memory_format = torch.contiguous_format);  permute_516 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_460: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(clone_75, div_42);  clone_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_7: "f32[4, 1920, 8, 8]" = torch.ops.aten.cat.default([add_460, add_230], 1);  add_460 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1295: "f32[4, 32, 60, 64]" = torch.ops.aten.view.default(cat_7, [4, 32, 60, 64])
        var_mean_88 = torch.ops.aten.var_mean.correction(view_1295, [2, 3], correction = 0, keepdim = True)
        getitem_308: "f32[4, 32, 1, 1]" = var_mean_88[0]
        getitem_309: "f32[4, 32, 1, 1]" = var_mean_88[1];  var_mean_88 = None
        add_461: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_88: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
        sub_88: "f32[4, 32, 60, 64]" = torch.ops.aten.sub.Tensor(view_1295, getitem_309);  view_1295 = None
        mul_471: "f32[4, 32, 60, 64]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        view_1296: "f32[4, 1920, 8, 8]" = torch.ops.aten.view.default(mul_471, [4, 1920, 8, 8]);  mul_471 = None
        unsqueeze_397: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(primals_887, 0)
        unsqueeze_398: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(primals_886, 0)
        unsqueeze_401: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        mul_472: "f32[4, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(view_1296, unsqueeze_402);  view_1296 = unsqueeze_402 = None
        add_462: "f32[4, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_472, unsqueeze_399);  mul_472 = unsqueeze_399 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_67: "f32[4, 1920, 8, 8]" = torch.ops.aten.sigmoid.default(add_462)
        mul_473: "f32[4, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(add_462, sigmoid_67);  add_462 = sigmoid_67 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_208: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_473, primals_888, primals_889, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_889 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_209: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_473, primals_890, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_210: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_209, primals_891, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_474: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_210, 1.0);  convolution_210 = None
        add_463: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_208, mul_474);  convolution_208 = mul_474 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_517: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_892, [1, 0]);  primals_892 = None
        addmm_63: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_893, mul_109, permute_517);  primals_893 = permute_517 = None
        unsqueeze_403: "f32[4, 1280, 1]" = torch.ops.aten.unsqueeze.default(addmm_63, 2);  addmm_63 = None
        unsqueeze_404: "f32[4, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_464: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_463, unsqueeze_404);  add_463 = unsqueeze_404 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1297: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(add_464, [4, 32, 40, 64])
        var_mean_89 = torch.ops.aten.var_mean.correction(view_1297, [2, 3], correction = 0, keepdim = True)
        getitem_310: "f32[4, 32, 1, 1]" = var_mean_89[0]
        getitem_311: "f32[4, 32, 1, 1]" = var_mean_89[1];  var_mean_89 = None
        add_465: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_89: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_465);  add_465 = None
        sub_89: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1297, getitem_311);  view_1297 = None
        mul_476: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
        view_1298: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_476, [4, 1280, 8, 8]);  mul_476 = None
        unsqueeze_405: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_895, 0)
        unsqueeze_406: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_894, 0)
        unsqueeze_409: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        mul_477: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1298, unsqueeze_410);  view_1298 = unsqueeze_410 = None
        add_466: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_477, unsqueeze_407);  mul_477 = unsqueeze_407 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_69: "f32[4, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_466)
        mul_478: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_466, sigmoid_69);  add_466 = sigmoid_69 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_211: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_478, primals_896, primals_897, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_897 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_212: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(mul_478, primals_898, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_213: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_212, primals_899, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_479: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_213, 1.0);  convolution_213 = None
        add_467: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_211, mul_479);  convolution_211 = mul_479 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_214: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(cat_7, primals_900, primals_901, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_901 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_215: "f32[4, 80, 8, 8]" = torch.ops.aten.convolution.default(cat_7, primals_902, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_216: "f32[4, 1280, 8, 8]" = torch.ops.aten.convolution.default(convolution_215, primals_903, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_480: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(convolution_216, 1.0);  convolution_216 = None
        add_468: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(convolution_214, mul_480);  convolution_214 = mul_480 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_469: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_468, add_467);  add_468 = add_467 = None
        div_45: "f32[4, 1280, 8, 8]" = torch.ops.aten.div.Tensor(add_469, 1.0);  add_469 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1299: "f32[4, 32, 40, 64]" = torch.ops.aten.view.default(div_45, [4, 32, 40, 64])
        var_mean_90 = torch.ops.aten.var_mean.correction(view_1299, [2, 3], correction = 0, keepdim = True)
        getitem_312: "f32[4, 32, 1, 1]" = var_mean_90[0]
        getitem_313: "f32[4, 32, 1, 1]" = var_mean_90[1];  var_mean_90 = None
        add_470: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
        rsqrt_90: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_470);  add_470 = None
        sub_90: "f32[4, 32, 40, 64]" = torch.ops.aten.sub.Tensor(view_1299, getitem_313);  view_1299 = None
        mul_481: "f32[4, 32, 40, 64]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
        view_1300: "f32[4, 1280, 8, 8]" = torch.ops.aten.view.default(mul_481, [4, 1280, 8, 8]);  mul_481 = None
        unsqueeze_411: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_905, 0);  primals_905 = None
        unsqueeze_412: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_904, 0)
        unsqueeze_415: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_482: "f32[4, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(view_1300, unsqueeze_416);  view_1300 = unsqueeze_416 = None
        add_471: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_413);  mul_482 = unsqueeze_413 = None
        squeeze_126: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_313, [2, 3]);  getitem_313 = None
        squeeze_127: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_90, [2, 3]);  rsqrt_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_518: "f32[4, 8, 8, 1280]" = torch.ops.aten.permute.default(add_471, [0, 2, 3, 1]);  add_471 = None
        view_1301: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_518, [4, 64, 1280]);  permute_518 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_519: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_906, [1, 0])
        expand_25: "f32[4, 64, 1280]" = torch.ops.aten.expand.default(view_1301, [4, 64, 1280])
        expand_26: "f32[4, 1280, 1280]" = torch.ops.aten.expand.default(permute_519, [4, 1280, 1280]);  permute_519 = None
        bmm_12: "f32[4, 64, 1280]" = torch.ops.aten.bmm.default(expand_25, expand_26);  expand_25 = expand_26 = None
        add_472: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(bmm_12, primals_907);  bmm_12 = primals_907 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_520: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_908, [1, 0]);  primals_908 = None
        clone_77: "f32[4, 64, 1280]" = torch.ops.aten.clone.default(view_1301, memory_format = torch.contiguous_format);  view_1301 = None
        view_1305: "f32[256, 1280]" = torch.ops.aten.view.default(clone_77, [256, 1280]);  clone_77 = None
        mm_278: "f32[256, 80]" = torch.ops.aten.mm.default(view_1305, permute_520)
        permute_521: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_909, [1, 0]);  primals_909 = None
        mm_279: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_278, permute_521)
        view_1308: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_279, [4, 64, 1280]);  mm_279 = None
        mul_483: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1308, 1.0);  view_1308 = None
        add_473: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_472, mul_483);  add_472 = mul_483 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_91 = torch.ops.aten.var_mean.correction(add_473, [2], correction = 0, keepdim = True)
        getitem_314: "f32[4, 64, 1]" = var_mean_91[0]
        getitem_315: "f32[4, 64, 1]" = var_mean_91[1];  var_mean_91 = None
        add_474: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-05);  getitem_314 = None
        rsqrt_91: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
        sub_91: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_473, getitem_315);  getitem_315 = None
        mul_484: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        mul_485: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_484, primals_910)
        add_475: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_485, primals_911);  mul_485 = primals_911 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_522: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_912, [1, 0]);  primals_912 = None
        view_1309: "f32[256, 1280]" = torch.ops.aten.view.default(add_475, [256, 1280]);  add_475 = None
        mm_280: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1309, permute_522)
        view_1310: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_280, [4, 64, 1280]);  mm_280 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_523: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_913, [1, 0]);  primals_913 = None
        mm_281: "f32[256, 80]" = torch.ops.aten.mm.default(view_1309, permute_523)
        permute_524: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_914, [1, 0]);  primals_914 = None
        mm_282: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_281, permute_524)
        view_1314: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_282, [4, 64, 1280]);  mm_282 = None
        mul_486: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1314, 1.0);  view_1314 = None
        add_476: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1310, mul_486);  view_1310 = mul_486 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_525: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_915, [1, 0]);  primals_915 = None
        mm_283: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1309, permute_525)
        view_1318: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_283, [4, 64, 1280]);  mm_283 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_526: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_916, [1, 0]);  primals_916 = None
        mm_284: "f32[256, 80]" = torch.ops.aten.mm.default(view_1309, permute_526)
        permute_527: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_917, [1, 0]);  primals_917 = None
        mm_285: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_284, permute_527)
        view_1322: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_285, [4, 64, 1280]);  mm_285 = None
        mul_487: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1322, 1.0);  view_1322 = None
        add_477: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1318, mul_487);  view_1318 = mul_487 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_528: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_918, [1, 0]);  primals_918 = None
        mm_286: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1309, permute_528)
        view_1326: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_286, [4, 64, 1280]);  mm_286 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_529: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_919, [1, 0]);  primals_919 = None
        mm_287: "f32[256, 80]" = torch.ops.aten.mm.default(view_1309, permute_529)
        permute_530: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_920, [1, 0]);  primals_920 = None
        mm_288: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_287, permute_530)
        view_1330: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_288, [4, 64, 1280]);  mm_288 = None
        mul_488: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1330, 1.0);  view_1330 = None
        add_478: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1326, mul_488);  view_1326 = mul_488 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1337: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_476, [4, -1, 20, 64]);  add_476 = None
        permute_534: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1337, [0, 2, 1, 3]);  view_1337 = None
        view_1339: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_477, [4, -1, 20, 64]);  add_477 = None
        permute_535: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1339, [0, 2, 1, 3]);  view_1339 = None
        view_1341: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_478, [4, -1, 20, 64]);  add_478 = None
        permute_536: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1341, [0, 2, 1, 3]);  view_1341 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_534, permute_535, permute_536, None, True)
        getitem_316: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_19[0]
        getitem_317: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_19[1]
        getitem_318: "i64[]" = _scaled_dot_product_efficient_attention_19[2]
        getitem_319: "i64[]" = _scaled_dot_product_efficient_attention_19[3];  _scaled_dot_product_efficient_attention_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_537: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_316, [0, 2, 1, 3])
        view_1342: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_537, [4, -1, 1280]);  permute_537 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1343: "f32[256, 1280]" = torch.ops.aten.view.default(view_1342, [256, 1280]);  view_1342 = None
        permute_538: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_921, [1, 0]);  primals_921 = None
        addmm_64: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_922, view_1343, permute_538);  primals_922 = None
        view_1344: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_64, [4, 64, 1280]);  addmm_64 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_539: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_923, [1, 0]);  primals_923 = None
        mm_289: "f32[256, 80]" = torch.ops.aten.mm.default(view_1343, permute_539);  view_1343 = None
        permute_540: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_924, [1, 0]);  primals_924 = None
        mm_290: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_289, permute_540)
        view_1348: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_290, [4, 64, 1280]);  mm_290 = None
        mul_489: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1348, 1.0);  view_1348 = None
        add_479: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1344, mul_489);  view_1344 = mul_489 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_46: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_479, 1.0);  add_479 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_480: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_46, add_473);  div_46 = add_473 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_92 = torch.ops.aten.var_mean.correction(add_480, [2], correction = 0, keepdim = True)
        getitem_320: "f32[4, 64, 1]" = var_mean_92[0]
        getitem_321: "f32[4, 64, 1]" = var_mean_92[1];  var_mean_92 = None
        add_481: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-05);  getitem_320 = None
        rsqrt_92: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_92: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_480, getitem_321);  getitem_321 = None
        mul_490: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        mul_491: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_490, primals_925)
        add_482: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_491, primals_926);  mul_491 = primals_926 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_541: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_927, [1, 0]);  primals_927 = None
        view_1352: "f32[256, 1280]" = torch.ops.aten.view.default(add_482, [256, 1280]);  add_482 = None
        mm_291: "f32[256, 1280]" = torch.ops.aten.mm.default(view_1352, permute_541)
        view_1353: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_291, [4, 64, 1280]);  mm_291 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_542: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_928, [1, 0]);  primals_928 = None
        mm_292: "f32[256, 80]" = torch.ops.aten.mm.default(view_1352, permute_542)
        permute_543: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_929, [1, 0]);  primals_929 = None
        mm_293: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_292, permute_543)
        view_1357: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_293, [4, 64, 1280]);  mm_293 = None
        mul_492: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1357, 1.0);  view_1357 = None
        add_483: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1353, mul_492);  view_1353 = mul_492 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_544: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_930, [1, 0]);  primals_930 = None
        mm_294: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_544);  permute_544 = None
        view_1361: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_294, [4, 77, 1280]);  mm_294 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_545: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_931, [1, 0]);  primals_931 = None
        mm_295: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_545);  permute_545 = None
        permute_546: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_932, [1, 0]);  primals_932 = None
        mm_296: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_295, permute_546)
        view_1365: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_296, [4, 77, 1280]);  mm_296 = None
        mul_493: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1365, 1.0);  view_1365 = None
        add_484: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1361, mul_493);  view_1361 = mul_493 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_547: "f32[1024, 1280]" = torch.ops.aten.permute.default(primals_933, [1, 0]);  primals_933 = None
        mm_297: "f32[308, 1280]" = torch.ops.aten.mm.default(view_148, permute_547);  permute_547 = None
        view_1369: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_297, [4, 77, 1280]);  mm_297 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_548: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_934, [1, 0]);  primals_934 = None
        mm_298: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_548);  permute_548 = None
        permute_549: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_935, [1, 0]);  primals_935 = None
        mm_299: "f32[308, 1280]" = torch.ops.aten.mm.default(mm_298, permute_549)
        view_1373: "f32[4, 77, 1280]" = torch.ops.aten.view.default(mm_299, [4, 77, 1280]);  mm_299 = None
        mul_494: "f32[4, 77, 1280]" = torch.ops.aten.mul.Tensor(view_1373, 1.0);  view_1373 = None
        add_485: "f32[4, 77, 1280]" = torch.ops.aten.add.Tensor(view_1369, mul_494);  view_1369 = mul_494 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1380: "f32[4, 64, 20, 64]" = torch.ops.aten.view.default(add_483, [4, -1, 20, 64]);  add_483 = None
        permute_553: "f32[4, 20, 64, 64]" = torch.ops.aten.permute.default(view_1380, [0, 2, 1, 3]);  view_1380 = None
        view_1382: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_484, [4, -1, 20, 64]);  add_484 = None
        permute_554: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1382, [0, 2, 1, 3]);  view_1382 = None
        view_1384: "f32[4, 77, 20, 64]" = torch.ops.aten.view.default(add_485, [4, -1, 20, 64]);  add_485 = None
        permute_555: "f32[4, 20, 77, 64]" = torch.ops.aten.permute.default(view_1384, [0, 2, 1, 3]);  view_1384 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_553, permute_554, permute_555, None, True)
        getitem_322: "f32[4, 20, 64, 64]" = _scaled_dot_product_efficient_attention_20[0]
        getitem_323: "f32[4, 20, 64]" = _scaled_dot_product_efficient_attention_20[1]
        getitem_324: "i64[]" = _scaled_dot_product_efficient_attention_20[2]
        getitem_325: "i64[]" = _scaled_dot_product_efficient_attention_20[3];  _scaled_dot_product_efficient_attention_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_556: "f32[4, 64, 20, 64]" = torch.ops.aten.permute.default(getitem_322, [0, 2, 1, 3])
        view_1385: "f32[4, 64, 1280]" = torch.ops.aten.view.default(permute_556, [4, -1, 1280]);  permute_556 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1386: "f32[256, 1280]" = torch.ops.aten.view.default(view_1385, [256, 1280]);  view_1385 = None
        permute_557: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_936, [1, 0]);  primals_936 = None
        addmm_65: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_937, view_1386, permute_557);  primals_937 = None
        view_1387: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_65, [4, 64, 1280]);  addmm_65 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_558: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_938, [1, 0]);  primals_938 = None
        mm_300: "f32[256, 80]" = torch.ops.aten.mm.default(view_1386, permute_558);  view_1386 = None
        permute_559: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_939, [1, 0]);  primals_939 = None
        mm_301: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_300, permute_559)
        view_1391: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_301, [4, 64, 1280]);  mm_301 = None
        mul_495: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1391, 1.0);  view_1391 = None
        add_486: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1387, mul_495);  view_1387 = mul_495 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_47: "f32[4, 64, 1280]" = torch.ops.aten.div.Tensor(add_486, 1.0);  add_486 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_487: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(div_47, add_480);  div_47 = add_480 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_93 = torch.ops.aten.var_mean.correction(add_487, [2], correction = 0, keepdim = True)
        getitem_326: "f32[4, 64, 1]" = var_mean_93[0]
        getitem_327: "f32[4, 64, 1]" = var_mean_93[1];  var_mean_93 = None
        add_488: "f32[4, 64, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-05);  getitem_326 = None
        rsqrt_93: "f32[4, 64, 1]" = torch.ops.aten.rsqrt.default(add_488);  add_488 = None
        sub_93: "f32[4, 64, 1280]" = torch.ops.aten.sub.Tensor(add_487, getitem_327);  getitem_327 = None
        mul_496: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        mul_497: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_496, primals_940)
        add_489: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(mul_497, primals_941);  mul_497 = primals_941 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1395: "f32[256, 1280]" = torch.ops.aten.view.default(add_489, [256, 1280]);  add_489 = None
        permute_560: "f32[1280, 10240]" = torch.ops.aten.permute.default(primals_942, [1, 0]);  primals_942 = None
        addmm_66: "f32[256, 10240]" = torch.ops.aten.addmm.default(primals_943, view_1395, permute_560);  primals_943 = None
        view_1396: "f32[4, 64, 10240]" = torch.ops.aten.view.default(addmm_66, [4, 64, 10240]);  addmm_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_561: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_944, [1, 0]);  primals_944 = None
        mm_302: "f32[256, 80]" = torch.ops.aten.mm.default(view_1395, permute_561)
        permute_562: "f32[80, 10240]" = torch.ops.aten.permute.default(primals_945, [1, 0]);  primals_945 = None
        mm_303: "f32[256, 10240]" = torch.ops.aten.mm.default(mm_302, permute_562)
        view_1400: "f32[4, 64, 10240]" = torch.ops.aten.view.default(mm_303, [4, 64, 10240]);  mm_303 = None
        mul_498: "f32[4, 64, 10240]" = torch.ops.aten.mul.Tensor(view_1400, 1.0);  view_1400 = None
        add_490: "f32[4, 64, 10240]" = torch.ops.aten.add.Tensor(view_1396, mul_498);  view_1396 = mul_498 = None
        view_1401: "f32[256, 10240]" = torch.ops.aten.view.default(add_490, [256, 10240]);  add_490 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1404: "f32[4, 64, 10240]" = torch.ops.aten.view.default(view_1401, [4, 64, 10240]);  view_1401 = None
        split_29 = torch.ops.aten.split.Tensor(view_1404, 5120, -1);  view_1404 = None
        getitem_331: "f32[4, 64, 5120]" = split_29[1]
        mul_499: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_331, 0.5)
        mul_500: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_331, 0.7071067811865476)
        erf_9: "f32[4, 64, 5120]" = torch.ops.aten.erf.default(mul_500);  mul_500 = None
        add_491: "f32[4, 64, 5120]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_501: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_499, add_491);  mul_499 = add_491 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_332: "f32[4, 64, 5120]" = split_29[0];  split_29 = None
        mul_502: "f32[4, 64, 5120]" = torch.ops.aten.mul.Tensor(getitem_332, mul_501);  mul_501 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1406: "f32[256, 5120]" = torch.ops.aten.view.default(mul_502, [256, 5120]);  mul_502 = None
        permute_563: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_946, [1, 0]);  primals_946 = None
        addmm_67: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_947, view_1406, permute_563);  primals_947 = None
        view_1407: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_67, [4, 64, 1280]);  addmm_67 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_564: "f32[5120, 80]" = torch.ops.aten.permute.default(primals_948, [1, 0]);  primals_948 = None
        mm_304: "f32[256, 80]" = torch.ops.aten.mm.default(view_1406, permute_564)
        permute_565: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_949, [1, 0]);  primals_949 = None
        mm_305: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_304, permute_565)
        view_1411: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_305, [4, 64, 1280]);  mm_305 = None
        mul_503: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1411, 1.0);  view_1411 = None
        add_492: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1407, mul_503);  view_1407 = mul_503 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_493: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(add_492, add_487);  add_492 = add_487 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1415: "f32[256, 1280]" = torch.ops.aten.view.default(add_493, [256, 1280]);  add_493 = None
        permute_566: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_950, [1, 0]);  primals_950 = None
        addmm_68: "f32[256, 1280]" = torch.ops.aten.addmm.default(primals_951, view_1415, permute_566);  primals_951 = None
        view_1416: "f32[4, 64, 1280]" = torch.ops.aten.view.default(addmm_68, [4, 64, 1280]);  addmm_68 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_567: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_952, [1, 0]);  primals_952 = None
        mm_306: "f32[256, 80]" = torch.ops.aten.mm.default(view_1415, permute_567)
        permute_568: "f32[80, 1280]" = torch.ops.aten.permute.default(primals_953, [1, 0]);  primals_953 = None
        mm_307: "f32[256, 1280]" = torch.ops.aten.mm.default(mm_306, permute_568)
        view_1420: "f32[4, 64, 1280]" = torch.ops.aten.view.default(mm_307, [4, 64, 1280]);  mm_307 = None
        mul_504: "f32[4, 64, 1280]" = torch.ops.aten.mul.Tensor(view_1420, 1.0);  view_1420 = None
        add_494: "f32[4, 64, 1280]" = torch.ops.aten.add.Tensor(view_1416, mul_504);  view_1416 = mul_504 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1426: "f32[4, 8, 8, 1280]" = torch.ops.aten.view.default(add_494, [4, 8, 8, 1280]);  add_494 = None
        permute_570: "f32[4, 1280, 8, 8]" = torch.ops.aten.permute.default(view_1426, [0, 3, 1, 2]);  view_1426 = None
        clone_81: "f32[4, 1280, 8, 8]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_495: "f32[4, 1280, 8, 8]" = torch.ops.aten.add.Tensor(clone_81, div_45);  clone_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota_3: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_505: "i64[16]" = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_496: "i64[16]" = torch.ops.aten.add.Tensor(mul_505, 0);  mul_505 = None
        convert_element_type_6: "f32[16]" = torch.ops.prims.convert_element_type.default(add_496, torch.float32);  add_496 = None
        add_497: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_6, 0.0);  convert_element_type_6 = None
        mul_506: "f32[16]" = torch.ops.aten.mul.Tensor(add_497, 0.5);  add_497 = None
        convert_element_type_7: "i64[16]" = torch.ops.prims.convert_element_type.default(mul_506, torch.int64);  mul_506 = None
        unsqueeze_417: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_7, -1)
        _unsafe_index_1: "f32[4, 1280, 16, 16]" = torch.ops.aten._unsafe_index.Tensor(add_495, [None, None, unsqueeze_417, convert_element_type_7]);  add_495 = unsqueeze_417 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_217: "f32[4, 1280, 16, 16]" = torch.ops.aten.convolution.default(_unsafe_index_1, primals_954, primals_955, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_955 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_218: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(_unsafe_index_1, primals_956, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_219: "f32[4, 1280, 16, 16]" = torch.ops.aten.convolution.default(convolution_218, primals_957, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_509: "f32[4, 1280, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_219, 1.0);  convolution_219 = None
        add_500: "f32[4, 1280, 16, 16]" = torch.ops.aten.add.Tensor(convolution_217, mul_509);  convolution_217 = mul_509 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_8: "f32[4, 1920, 16, 16]" = torch.ops.aten.cat.default([add_500, add_229], 1);  add_500 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1427: "f32[4, 32, 60, 256]" = torch.ops.aten.view.default(cat_8, [4, 32, 60, 256])
        var_mean_94 = torch.ops.aten.var_mean.correction(view_1427, [2, 3], correction = 0, keepdim = True)
        getitem_334: "f32[4, 32, 1, 1]" = var_mean_94[0]
        getitem_335: "f32[4, 32, 1, 1]" = var_mean_94[1];  var_mean_94 = None
        add_501: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_334, 1e-05);  getitem_334 = None
        rsqrt_94: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_501);  add_501 = None
        sub_94: "f32[4, 32, 60, 256]" = torch.ops.aten.sub.Tensor(view_1427, getitem_335);  view_1427 = None
        mul_510: "f32[4, 32, 60, 256]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
        view_1428: "f32[4, 1920, 16, 16]" = torch.ops.aten.view.default(mul_510, [4, 1920, 16, 16]);  mul_510 = None
        unsqueeze_418: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(primals_959, 0)
        unsqueeze_419: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(primals_958, 0)
        unsqueeze_422: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_511: "f32[4, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(view_1428, unsqueeze_423);  view_1428 = unsqueeze_423 = None
        add_502: "f32[4, 1920, 16, 16]" = torch.ops.aten.add.Tensor(mul_511, unsqueeze_420);  mul_511 = unsqueeze_420 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_70: "f32[4, 1920, 16, 16]" = torch.ops.aten.sigmoid.default(add_502)
        mul_512: "f32[4, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(add_502, sigmoid_70);  add_502 = sigmoid_70 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_220: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_512, primals_960, primals_961, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_961 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_221: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_512, primals_962, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_222: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_221, primals_963, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_513: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_222, 1.0);  convolution_222 = None
        add_503: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_220, mul_513);  convolution_220 = mul_513 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_571: "f32[1280, 640]" = torch.ops.aten.permute.default(primals_964, [1, 0]);  primals_964 = None
        addmm_69: "f32[4, 640]" = torch.ops.aten.addmm.default(primals_965, mul_109, permute_571);  primals_965 = permute_571 = None
        unsqueeze_424: "f32[4, 640, 1]" = torch.ops.aten.unsqueeze.default(addmm_69, 2);  addmm_69 = None
        unsqueeze_425: "f32[4, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_504: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_503, unsqueeze_425);  add_503 = unsqueeze_425 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1429: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_504, [4, 32, 20, 256])
        var_mean_95 = torch.ops.aten.var_mean.correction(view_1429, [2, 3], correction = 0, keepdim = True)
        getitem_336: "f32[4, 32, 1, 1]" = var_mean_95[0]
        getitem_337: "f32[4, 32, 1, 1]" = var_mean_95[1];  var_mean_95 = None
        add_505: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_95: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_505);  add_505 = None
        sub_95: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1429, getitem_337);  view_1429 = None
        mul_515: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
        view_1430: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_515, [4, 640, 16, 16]);  mul_515 = None
        unsqueeze_426: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_967, 0)
        unsqueeze_427: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_966, 0)
        unsqueeze_430: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_516: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1430, unsqueeze_431);  view_1430 = unsqueeze_431 = None
        add_506: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_516, unsqueeze_428);  mul_516 = unsqueeze_428 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_72: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_506)
        mul_517: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_506, sigmoid_72);  add_506 = sigmoid_72 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_223: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_517, primals_968, primals_969, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_969 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_224: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_517, primals_970, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_225: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_224, primals_971, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_518: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_225, 1.0);  convolution_225 = None
        add_507: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_223, mul_518);  convolution_223 = mul_518 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_226: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(cat_8, primals_972, primals_973, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_973 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_227: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(cat_8, primals_974, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_228: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_227, primals_975, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_519: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_228, 1.0);  convolution_228 = None
        add_508: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_226, mul_519);  convolution_226 = mul_519 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_509: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_508, add_507);  add_508 = add_507 = None
        div_48: "f32[4, 640, 16, 16]" = torch.ops.aten.div.Tensor(add_509, 1.0);  add_509 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1431: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(div_48, [4, 32, 20, 256])
        var_mean_96 = torch.ops.aten.var_mean.correction(view_1431, [2, 3], correction = 0, keepdim = True)
        getitem_338: "f32[4, 32, 1, 1]" = var_mean_96[0]
        getitem_339: "f32[4, 32, 1, 1]" = var_mean_96[1];  var_mean_96 = None
        add_510: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_96: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_510);  add_510 = None
        sub_96: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1431, getitem_339);  view_1431 = None
        mul_520: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
        view_1432: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_520, [4, 640, 16, 16]);  mul_520 = None
        unsqueeze_432: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_977, 0);  primals_977 = None
        unsqueeze_433: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_976, 0)
        unsqueeze_436: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        mul_521: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1432, unsqueeze_437);  view_1432 = unsqueeze_437 = None
        add_511: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_521, unsqueeze_434);  mul_521 = unsqueeze_434 = None
        squeeze_132: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_339, [2, 3]);  getitem_339 = None
        squeeze_133: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_96, [2, 3]);  rsqrt_96 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_572: "f32[4, 16, 16, 640]" = torch.ops.aten.permute.default(add_511, [0, 2, 3, 1]);  add_511 = None
        view_1433: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_572, [4, 256, 640]);  permute_572 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_573: "f32[640, 640]" = torch.ops.aten.permute.default(primals_978, [1, 0])
        expand_27: "f32[4, 256, 640]" = torch.ops.aten.expand.default(view_1433, [4, 256, 640])
        expand_28: "f32[4, 640, 640]" = torch.ops.aten.expand.default(permute_573, [4, 640, 640]);  permute_573 = None
        bmm_13: "f32[4, 256, 640]" = torch.ops.aten.bmm.default(expand_27, expand_28);  expand_27 = expand_28 = None
        add_512: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(bmm_13, primals_979);  bmm_13 = primals_979 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_574: "f32[640, 80]" = torch.ops.aten.permute.default(primals_980, [1, 0]);  primals_980 = None
        clone_83: "f32[4, 256, 640]" = torch.ops.aten.clone.default(view_1433, memory_format = torch.contiguous_format);  view_1433 = None
        view_1437: "f32[1024, 640]" = torch.ops.aten.view.default(clone_83, [1024, 640]);  clone_83 = None
        mm_308: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1437, permute_574)
        permute_575: "f32[80, 640]" = torch.ops.aten.permute.default(primals_981, [1, 0]);  primals_981 = None
        mm_309: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_308, permute_575)
        view_1440: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_309, [4, 256, 640]);  mm_309 = None
        mul_522: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1440, 1.0);  view_1440 = None
        add_513: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_512, mul_522);  add_512 = mul_522 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_97 = torch.ops.aten.var_mean.correction(add_513, [2], correction = 0, keepdim = True)
        getitem_340: "f32[4, 256, 1]" = var_mean_97[0]
        getitem_341: "f32[4, 256, 1]" = var_mean_97[1];  var_mean_97 = None
        add_514: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_97: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_514);  add_514 = None
        sub_97: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_513, getitem_341);  getitem_341 = None
        mul_523: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
        mul_524: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_523, primals_982)
        add_515: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_524, primals_983);  mul_524 = primals_983 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_576: "f32[640, 640]" = torch.ops.aten.permute.default(primals_984, [1, 0]);  primals_984 = None
        view_1441: "f32[1024, 640]" = torch.ops.aten.view.default(add_515, [1024, 640]);  add_515 = None
        mm_310: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1441, permute_576)
        view_1442: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_310, [4, 256, 640]);  mm_310 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_577: "f32[640, 80]" = torch.ops.aten.permute.default(primals_985, [1, 0]);  primals_985 = None
        mm_311: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1441, permute_577)
        permute_578: "f32[80, 640]" = torch.ops.aten.permute.default(primals_986, [1, 0]);  primals_986 = None
        mm_312: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_311, permute_578)
        view_1446: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_312, [4, 256, 640]);  mm_312 = None
        mul_525: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1446, 1.0);  view_1446 = None
        add_516: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1442, mul_525);  view_1442 = mul_525 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_579: "f32[640, 640]" = torch.ops.aten.permute.default(primals_987, [1, 0]);  primals_987 = None
        mm_313: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1441, permute_579)
        view_1450: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_313, [4, 256, 640]);  mm_313 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_580: "f32[640, 80]" = torch.ops.aten.permute.default(primals_988, [1, 0]);  primals_988 = None
        mm_314: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1441, permute_580)
        permute_581: "f32[80, 640]" = torch.ops.aten.permute.default(primals_989, [1, 0]);  primals_989 = None
        mm_315: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_314, permute_581)
        view_1454: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_315, [4, 256, 640]);  mm_315 = None
        mul_526: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1454, 1.0);  view_1454 = None
        add_517: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1450, mul_526);  view_1450 = mul_526 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_582: "f32[640, 640]" = torch.ops.aten.permute.default(primals_990, [1, 0]);  primals_990 = None
        mm_316: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1441, permute_582)
        view_1458: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_316, [4, 256, 640]);  mm_316 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_583: "f32[640, 80]" = torch.ops.aten.permute.default(primals_991, [1, 0]);  primals_991 = None
        mm_317: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1441, permute_583)
        permute_584: "f32[80, 640]" = torch.ops.aten.permute.default(primals_992, [1, 0]);  primals_992 = None
        mm_318: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_317, permute_584)
        view_1462: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_318, [4, 256, 640]);  mm_318 = None
        mul_527: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1462, 1.0);  view_1462 = None
        add_518: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1458, mul_527);  view_1458 = mul_527 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1469: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_516, [4, -1, 10, 64]);  add_516 = None
        permute_588: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1469, [0, 2, 1, 3]);  view_1469 = None
        view_1471: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_517, [4, -1, 10, 64]);  add_517 = None
        permute_589: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1471, [0, 2, 1, 3]);  view_1471 = None
        view_1473: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_518, [4, -1, 10, 64]);  add_518 = None
        permute_590: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1473, [0, 2, 1, 3]);  view_1473 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_588, permute_589, permute_590, None, True)
        getitem_342: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_21[0]
        getitem_343: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_21[1]
        getitem_344: "i64[]" = _scaled_dot_product_efficient_attention_21[2]
        getitem_345: "i64[]" = _scaled_dot_product_efficient_attention_21[3];  _scaled_dot_product_efficient_attention_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_591: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_342, [0, 2, 1, 3])
        view_1474: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_591, [4, -1, 640]);  permute_591 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1475: "f32[1024, 640]" = torch.ops.aten.view.default(view_1474, [1024, 640]);  view_1474 = None
        permute_592: "f32[640, 640]" = torch.ops.aten.permute.default(primals_993, [1, 0]);  primals_993 = None
        addmm_70: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_994, view_1475, permute_592);  primals_994 = None
        view_1476: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_70, [4, 256, 640]);  addmm_70 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_593: "f32[640, 80]" = torch.ops.aten.permute.default(primals_995, [1, 0]);  primals_995 = None
        mm_319: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1475, permute_593);  view_1475 = None
        permute_594: "f32[80, 640]" = torch.ops.aten.permute.default(primals_996, [1, 0]);  primals_996 = None
        mm_320: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_319, permute_594)
        view_1480: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_320, [4, 256, 640]);  mm_320 = None
        mul_528: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1480, 1.0);  view_1480 = None
        add_519: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1476, mul_528);  view_1476 = mul_528 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_49: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_519, 1.0);  add_519 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_520: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_49, add_513);  div_49 = add_513 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_98 = torch.ops.aten.var_mean.correction(add_520, [2], correction = 0, keepdim = True)
        getitem_346: "f32[4, 256, 1]" = var_mean_98[0]
        getitem_347: "f32[4, 256, 1]" = var_mean_98[1];  var_mean_98 = None
        add_521: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_346, 1e-05);  getitem_346 = None
        rsqrt_98: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_521);  add_521 = None
        sub_98: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_520, getitem_347);  getitem_347 = None
        mul_529: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
        mul_530: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_529, primals_997)
        add_522: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_530, primals_998);  mul_530 = primals_998 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_595: "f32[640, 640]" = torch.ops.aten.permute.default(primals_999, [1, 0]);  primals_999 = None
        view_1484: "f32[1024, 640]" = torch.ops.aten.view.default(add_522, [1024, 640]);  add_522 = None
        mm_321: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1484, permute_595)
        view_1485: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_321, [4, 256, 640]);  mm_321 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_596: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1000, [1, 0]);  primals_1000 = None
        mm_322: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1484, permute_596)
        permute_597: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1001, [1, 0]);  primals_1001 = None
        mm_323: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_322, permute_597)
        view_1489: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_323, [4, 256, 640]);  mm_323 = None
        mul_531: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1489, 1.0);  view_1489 = None
        add_523: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1485, mul_531);  view_1485 = mul_531 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_598: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1002, [1, 0]);  primals_1002 = None
        mm_324: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_598);  permute_598 = None
        view_1493: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_324, [4, 77, 640]);  mm_324 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_599: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1003, [1, 0]);  primals_1003 = None
        mm_325: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_599);  permute_599 = None
        permute_600: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1004, [1, 0]);  primals_1004 = None
        mm_326: "f32[308, 640]" = torch.ops.aten.mm.default(mm_325, permute_600)
        view_1497: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_326, [4, 77, 640]);  mm_326 = None
        mul_532: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1497, 1.0);  view_1497 = None
        add_524: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1493, mul_532);  view_1493 = mul_532 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_601: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1005, [1, 0]);  primals_1005 = None
        mm_327: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_601);  permute_601 = None
        view_1501: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_327, [4, 77, 640]);  mm_327 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_602: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1006, [1, 0]);  primals_1006 = None
        mm_328: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_602);  permute_602 = None
        permute_603: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1007, [1, 0]);  primals_1007 = None
        mm_329: "f32[308, 640]" = torch.ops.aten.mm.default(mm_328, permute_603)
        view_1505: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_329, [4, 77, 640]);  mm_329 = None
        mul_533: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1505, 1.0);  view_1505 = None
        add_525: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1501, mul_533);  view_1501 = mul_533 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1512: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_523, [4, -1, 10, 64]);  add_523 = None
        permute_607: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1512, [0, 2, 1, 3]);  view_1512 = None
        view_1514: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_524, [4, -1, 10, 64]);  add_524 = None
        permute_608: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1514, [0, 2, 1, 3]);  view_1514 = None
        view_1516: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_525, [4, -1, 10, 64]);  add_525 = None
        permute_609: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1516, [0, 2, 1, 3]);  view_1516 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_607, permute_608, permute_609, None, True)
        getitem_348: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_22[0]
        getitem_349: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_22[1]
        getitem_350: "i64[]" = _scaled_dot_product_efficient_attention_22[2]
        getitem_351: "i64[]" = _scaled_dot_product_efficient_attention_22[3];  _scaled_dot_product_efficient_attention_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_610: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_348, [0, 2, 1, 3])
        view_1517: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_610, [4, -1, 640]);  permute_610 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1518: "f32[1024, 640]" = torch.ops.aten.view.default(view_1517, [1024, 640]);  view_1517 = None
        permute_611: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1008, [1, 0]);  primals_1008 = None
        addmm_71: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1009, view_1518, permute_611);  primals_1009 = None
        view_1519: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_71, [4, 256, 640]);  addmm_71 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_612: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1010, [1, 0]);  primals_1010 = None
        mm_330: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1518, permute_612);  view_1518 = None
        permute_613: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1011, [1, 0]);  primals_1011 = None
        mm_331: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_330, permute_613)
        view_1523: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_331, [4, 256, 640]);  mm_331 = None
        mul_534: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1523, 1.0);  view_1523 = None
        add_526: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1519, mul_534);  view_1519 = mul_534 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_50: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_526, 1.0);  add_526 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_527: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_50, add_520);  div_50 = add_520 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_99 = torch.ops.aten.var_mean.correction(add_527, [2], correction = 0, keepdim = True)
        getitem_352: "f32[4, 256, 1]" = var_mean_99[0]
        getitem_353: "f32[4, 256, 1]" = var_mean_99[1];  var_mean_99 = None
        add_528: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-05);  getitem_352 = None
        rsqrt_99: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_528);  add_528 = None
        sub_99: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_527, getitem_353);  getitem_353 = None
        mul_535: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
        mul_536: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_535, primals_1012)
        add_529: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_536, primals_1013);  mul_536 = primals_1013 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1527: "f32[1024, 640]" = torch.ops.aten.view.default(add_529, [1024, 640]);  add_529 = None
        permute_614: "f32[640, 5120]" = torch.ops.aten.permute.default(primals_1014, [1, 0]);  primals_1014 = None
        addmm_72: "f32[1024, 5120]" = torch.ops.aten.addmm.default(primals_1015, view_1527, permute_614);  primals_1015 = None
        view_1528: "f32[4, 256, 5120]" = torch.ops.aten.view.default(addmm_72, [4, 256, 5120]);  addmm_72 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_615: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1016, [1, 0]);  primals_1016 = None
        mm_332: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1527, permute_615)
        permute_616: "f32[80, 5120]" = torch.ops.aten.permute.default(primals_1017, [1, 0]);  primals_1017 = None
        mm_333: "f32[1024, 5120]" = torch.ops.aten.mm.default(mm_332, permute_616)
        view_1532: "f32[4, 256, 5120]" = torch.ops.aten.view.default(mm_333, [4, 256, 5120]);  mm_333 = None
        mul_537: "f32[4, 256, 5120]" = torch.ops.aten.mul.Tensor(view_1532, 1.0);  view_1532 = None
        add_530: "f32[4, 256, 5120]" = torch.ops.aten.add.Tensor(view_1528, mul_537);  view_1528 = mul_537 = None
        view_1533: "f32[1024, 5120]" = torch.ops.aten.view.default(add_530, [1024, 5120]);  add_530 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1536: "f32[4, 256, 5120]" = torch.ops.aten.view.default(view_1533, [4, 256, 5120]);  view_1533 = None
        split_32 = torch.ops.aten.split.Tensor(view_1536, 2560, -1);  view_1536 = None
        getitem_357: "f32[4, 256, 2560]" = split_32[1]
        mul_538: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_357, 0.5)
        mul_539: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_357, 0.7071067811865476)
        erf_10: "f32[4, 256, 2560]" = torch.ops.aten.erf.default(mul_539);  mul_539 = None
        add_531: "f32[4, 256, 2560]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_540: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_538, add_531);  mul_538 = add_531 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_358: "f32[4, 256, 2560]" = split_32[0];  split_32 = None
        mul_541: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_358, mul_540);  mul_540 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1538: "f32[1024, 2560]" = torch.ops.aten.view.default(mul_541, [1024, 2560]);  mul_541 = None
        permute_617: "f32[2560, 640]" = torch.ops.aten.permute.default(primals_1018, [1, 0]);  primals_1018 = None
        addmm_73: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1019, view_1538, permute_617);  primals_1019 = None
        view_1539: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_73, [4, 256, 640]);  addmm_73 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_618: "f32[2560, 80]" = torch.ops.aten.permute.default(primals_1020, [1, 0]);  primals_1020 = None
        mm_334: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1538, permute_618)
        permute_619: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1021, [1, 0]);  primals_1021 = None
        mm_335: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_334, permute_619)
        view_1543: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_335, [4, 256, 640]);  mm_335 = None
        mul_542: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1543, 1.0);  view_1543 = None
        add_532: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1539, mul_542);  view_1539 = mul_542 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_533: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_532, add_527);  add_532 = add_527 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1547: "f32[1024, 640]" = torch.ops.aten.view.default(add_533, [1024, 640]);  add_533 = None
        permute_620: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1022, [1, 0]);  primals_1022 = None
        addmm_74: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1023, view_1547, permute_620);  primals_1023 = None
        view_1548: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_74, [4, 256, 640]);  addmm_74 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_621: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1024, [1, 0]);  primals_1024 = None
        mm_336: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1547, permute_621)
        permute_622: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1025, [1, 0]);  primals_1025 = None
        mm_337: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_336, permute_622)
        view_1552: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_337, [4, 256, 640]);  mm_337 = None
        mul_543: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1552, 1.0);  view_1552 = None
        add_534: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1548, mul_543);  view_1548 = mul_543 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1558: "f32[4, 16, 16, 640]" = torch.ops.aten.view.default(add_534, [4, 16, 16, 640]);  add_534 = None
        permute_624: "f32[4, 640, 16, 16]" = torch.ops.aten.permute.default(view_1558, [0, 3, 1, 2]);  view_1558 = None
        clone_87: "f32[4, 640, 16, 16]" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_535: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(clone_87, div_48);  clone_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_9: "f32[4, 1280, 16, 16]" = torch.ops.aten.cat.default([add_535, add_195], 1);  add_535 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1559: "f32[4, 32, 40, 256]" = torch.ops.aten.view.default(cat_9, [4, 32, 40, 256])
        var_mean_100 = torch.ops.aten.var_mean.correction(view_1559, [2, 3], correction = 0, keepdim = True)
        getitem_360: "f32[4, 32, 1, 1]" = var_mean_100[0]
        getitem_361: "f32[4, 32, 1, 1]" = var_mean_100[1];  var_mean_100 = None
        add_536: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_360, 1e-05);  getitem_360 = None
        rsqrt_100: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_536);  add_536 = None
        sub_100: "f32[4, 32, 40, 256]" = torch.ops.aten.sub.Tensor(view_1559, getitem_361);  view_1559 = None
        mul_544: "f32[4, 32, 40, 256]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
        view_1560: "f32[4, 1280, 16, 16]" = torch.ops.aten.view.default(mul_544, [4, 1280, 16, 16]);  mul_544 = None
        unsqueeze_438: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_1027, 0)
        unsqueeze_439: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_1026, 0)
        unsqueeze_442: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_545: "f32[4, 1280, 16, 16]" = torch.ops.aten.mul.Tensor(view_1560, unsqueeze_443);  view_1560 = unsqueeze_443 = None
        add_537: "f32[4, 1280, 16, 16]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_440);  mul_545 = unsqueeze_440 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_73: "f32[4, 1280, 16, 16]" = torch.ops.aten.sigmoid.default(add_537)
        mul_546: "f32[4, 1280, 16, 16]" = torch.ops.aten.mul.Tensor(add_537, sigmoid_73);  add_537 = sigmoid_73 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_229: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_546, primals_1028, primals_1029, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1029 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_230: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_546, primals_1030, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_231: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_230, primals_1031, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_547: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_231, 1.0);  convolution_231 = None
        add_538: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_229, mul_547);  convolution_229 = mul_547 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_625: "f32[1280, 640]" = torch.ops.aten.permute.default(primals_1032, [1, 0]);  primals_1032 = None
        addmm_75: "f32[4, 640]" = torch.ops.aten.addmm.default(primals_1033, mul_109, permute_625);  primals_1033 = permute_625 = None
        unsqueeze_444: "f32[4, 640, 1]" = torch.ops.aten.unsqueeze.default(addmm_75, 2);  addmm_75 = None
        unsqueeze_445: "f32[4, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_539: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_538, unsqueeze_445);  add_538 = unsqueeze_445 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1561: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_539, [4, 32, 20, 256])
        var_mean_101 = torch.ops.aten.var_mean.correction(view_1561, [2, 3], correction = 0, keepdim = True)
        getitem_362: "f32[4, 32, 1, 1]" = var_mean_101[0]
        getitem_363: "f32[4, 32, 1, 1]" = var_mean_101[1];  var_mean_101 = None
        add_540: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-05);  getitem_362 = None
        rsqrt_101: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
        sub_101: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1561, getitem_363);  view_1561 = None
        mul_549: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
        view_1562: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_549, [4, 640, 16, 16]);  mul_549 = None
        unsqueeze_446: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1035, 0)
        unsqueeze_447: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        unsqueeze_449: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1034, 0)
        unsqueeze_450: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_550: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1562, unsqueeze_451);  view_1562 = unsqueeze_451 = None
        add_541: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_550, unsqueeze_448);  mul_550 = unsqueeze_448 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_75: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_541)
        mul_551: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_541, sigmoid_75);  add_541 = sigmoid_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_232: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_551, primals_1036, primals_1037, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1037 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_233: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_551, primals_1038, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_234: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_233, primals_1039, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_552: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_234, 1.0);  convolution_234 = None
        add_542: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_232, mul_552);  convolution_232 = mul_552 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_235: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(cat_9, primals_1040, primals_1041, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_1041 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_236: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(cat_9, primals_1042, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_237: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_236, primals_1043, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_553: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_237, 1.0);  convolution_237 = None
        add_543: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_235, mul_553);  convolution_235 = mul_553 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_544: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_543, add_542);  add_543 = add_542 = None
        div_51: "f32[4, 640, 16, 16]" = torch.ops.aten.div.Tensor(add_544, 1.0);  add_544 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1563: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(div_51, [4, 32, 20, 256])
        var_mean_102 = torch.ops.aten.var_mean.correction(view_1563, [2, 3], correction = 0, keepdim = True)
        getitem_364: "f32[4, 32, 1, 1]" = var_mean_102[0]
        getitem_365: "f32[4, 32, 1, 1]" = var_mean_102[1];  var_mean_102 = None
        add_545: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_364, 1e-06);  getitem_364 = None
        rsqrt_102: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_545);  add_545 = None
        sub_102: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1563, getitem_365);  view_1563 = None
        mul_554: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
        view_1564: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_554, [4, 640, 16, 16]);  mul_554 = None
        unsqueeze_452: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1045, 0);  primals_1045 = None
        unsqueeze_453: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        unsqueeze_455: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1044, 0)
        unsqueeze_456: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_555: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1564, unsqueeze_457);  view_1564 = unsqueeze_457 = None
        add_546: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_555, unsqueeze_454);  mul_555 = unsqueeze_454 = None
        squeeze_138: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_365, [2, 3]);  getitem_365 = None
        squeeze_139: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_102, [2, 3]);  rsqrt_102 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_626: "f32[4, 16, 16, 640]" = torch.ops.aten.permute.default(add_546, [0, 2, 3, 1]);  add_546 = None
        view_1565: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_626, [4, 256, 640]);  permute_626 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_627: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1046, [1, 0])
        expand_29: "f32[4, 256, 640]" = torch.ops.aten.expand.default(view_1565, [4, 256, 640])
        expand_30: "f32[4, 640, 640]" = torch.ops.aten.expand.default(permute_627, [4, 640, 640]);  permute_627 = None
        bmm_14: "f32[4, 256, 640]" = torch.ops.aten.bmm.default(expand_29, expand_30);  expand_29 = expand_30 = None
        add_547: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(bmm_14, primals_1047);  bmm_14 = primals_1047 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_628: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1048, [1, 0]);  primals_1048 = None
        clone_89: "f32[4, 256, 640]" = torch.ops.aten.clone.default(view_1565, memory_format = torch.contiguous_format);  view_1565 = None
        view_1569: "f32[1024, 640]" = torch.ops.aten.view.default(clone_89, [1024, 640]);  clone_89 = None
        mm_338: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1569, permute_628)
        permute_629: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1049, [1, 0]);  primals_1049 = None
        mm_339: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_338, permute_629)
        view_1572: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_339, [4, 256, 640]);  mm_339 = None
        mul_556: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1572, 1.0);  view_1572 = None
        add_548: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_547, mul_556);  add_547 = mul_556 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_103 = torch.ops.aten.var_mean.correction(add_548, [2], correction = 0, keepdim = True)
        getitem_366: "f32[4, 256, 1]" = var_mean_103[0]
        getitem_367: "f32[4, 256, 1]" = var_mean_103[1];  var_mean_103 = None
        add_549: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-05);  getitem_366 = None
        rsqrt_103: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_549);  add_549 = None
        sub_103: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_548, getitem_367);  getitem_367 = None
        mul_557: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
        mul_558: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_557, primals_1050)
        add_550: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_558, primals_1051);  mul_558 = primals_1051 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_630: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1052, [1, 0]);  primals_1052 = None
        view_1573: "f32[1024, 640]" = torch.ops.aten.view.default(add_550, [1024, 640]);  add_550 = None
        mm_340: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1573, permute_630)
        view_1574: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_340, [4, 256, 640]);  mm_340 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_631: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1053, [1, 0]);  primals_1053 = None
        mm_341: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1573, permute_631)
        permute_632: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1054, [1, 0]);  primals_1054 = None
        mm_342: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_341, permute_632)
        view_1578: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_342, [4, 256, 640]);  mm_342 = None
        mul_559: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1578, 1.0);  view_1578 = None
        add_551: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1574, mul_559);  view_1574 = mul_559 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_633: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1055, [1, 0]);  primals_1055 = None
        mm_343: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1573, permute_633)
        view_1582: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_343, [4, 256, 640]);  mm_343 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_634: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1056, [1, 0]);  primals_1056 = None
        mm_344: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1573, permute_634)
        permute_635: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1057, [1, 0]);  primals_1057 = None
        mm_345: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_344, permute_635)
        view_1586: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_345, [4, 256, 640]);  mm_345 = None
        mul_560: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1586, 1.0);  view_1586 = None
        add_552: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1582, mul_560);  view_1582 = mul_560 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_636: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1058, [1, 0]);  primals_1058 = None
        mm_346: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1573, permute_636)
        view_1590: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_346, [4, 256, 640]);  mm_346 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_637: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1059, [1, 0]);  primals_1059 = None
        mm_347: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1573, permute_637)
        permute_638: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1060, [1, 0]);  primals_1060 = None
        mm_348: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_347, permute_638)
        view_1594: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_348, [4, 256, 640]);  mm_348 = None
        mul_561: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1594, 1.0);  view_1594 = None
        add_553: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1590, mul_561);  view_1590 = mul_561 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1601: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_551, [4, -1, 10, 64]);  add_551 = None
        permute_642: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1601, [0, 2, 1, 3]);  view_1601 = None
        view_1603: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_552, [4, -1, 10, 64]);  add_552 = None
        permute_643: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1603, [0, 2, 1, 3]);  view_1603 = None
        view_1605: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_553, [4, -1, 10, 64]);  add_553 = None
        permute_644: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1605, [0, 2, 1, 3]);  view_1605 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_642, permute_643, permute_644, None, True)
        getitem_368: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_23[0]
        getitem_369: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_23[1]
        getitem_370: "i64[]" = _scaled_dot_product_efficient_attention_23[2]
        getitem_371: "i64[]" = _scaled_dot_product_efficient_attention_23[3];  _scaled_dot_product_efficient_attention_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_645: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_368, [0, 2, 1, 3])
        view_1606: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_645, [4, -1, 640]);  permute_645 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1607: "f32[1024, 640]" = torch.ops.aten.view.default(view_1606, [1024, 640]);  view_1606 = None
        permute_646: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1061, [1, 0]);  primals_1061 = None
        addmm_76: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1062, view_1607, permute_646);  primals_1062 = None
        view_1608: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_76, [4, 256, 640]);  addmm_76 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_647: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1063, [1, 0]);  primals_1063 = None
        mm_349: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1607, permute_647);  view_1607 = None
        permute_648: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1064, [1, 0]);  primals_1064 = None
        mm_350: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_349, permute_648)
        view_1612: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_350, [4, 256, 640]);  mm_350 = None
        mul_562: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1612, 1.0);  view_1612 = None
        add_554: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1608, mul_562);  view_1608 = mul_562 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_52: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_554, 1.0);  add_554 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_555: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_52, add_548);  div_52 = add_548 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_104 = torch.ops.aten.var_mean.correction(add_555, [2], correction = 0, keepdim = True)
        getitem_372: "f32[4, 256, 1]" = var_mean_104[0]
        getitem_373: "f32[4, 256, 1]" = var_mean_104[1];  var_mean_104 = None
        add_556: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_372, 1e-05);  getitem_372 = None
        rsqrt_104: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_556);  add_556 = None
        sub_104: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_555, getitem_373);  getitem_373 = None
        mul_563: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
        mul_564: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_563, primals_1065)
        add_557: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_564, primals_1066);  mul_564 = primals_1066 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_649: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1067, [1, 0]);  primals_1067 = None
        view_1616: "f32[1024, 640]" = torch.ops.aten.view.default(add_557, [1024, 640]);  add_557 = None
        mm_351: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1616, permute_649)
        view_1617: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_351, [4, 256, 640]);  mm_351 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_650: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1068, [1, 0]);  primals_1068 = None
        mm_352: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1616, permute_650)
        permute_651: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1069, [1, 0]);  primals_1069 = None
        mm_353: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_352, permute_651)
        view_1621: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_353, [4, 256, 640]);  mm_353 = None
        mul_565: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1621, 1.0);  view_1621 = None
        add_558: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1617, mul_565);  view_1617 = mul_565 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_652: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1070, [1, 0]);  primals_1070 = None
        mm_354: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_652);  permute_652 = None
        view_1625: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_354, [4, 77, 640]);  mm_354 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_653: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1071, [1, 0]);  primals_1071 = None
        mm_355: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_653);  permute_653 = None
        permute_654: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1072, [1, 0]);  primals_1072 = None
        mm_356: "f32[308, 640]" = torch.ops.aten.mm.default(mm_355, permute_654)
        view_1629: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_356, [4, 77, 640]);  mm_356 = None
        mul_566: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1629, 1.0);  view_1629 = None
        add_559: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1625, mul_566);  view_1625 = mul_566 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_655: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1073, [1, 0]);  primals_1073 = None
        mm_357: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_655);  permute_655 = None
        view_1633: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_357, [4, 77, 640]);  mm_357 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_656: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1074, [1, 0]);  primals_1074 = None
        mm_358: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_656);  permute_656 = None
        permute_657: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1075, [1, 0]);  primals_1075 = None
        mm_359: "f32[308, 640]" = torch.ops.aten.mm.default(mm_358, permute_657)
        view_1637: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_359, [4, 77, 640]);  mm_359 = None
        mul_567: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1637, 1.0);  view_1637 = None
        add_560: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1633, mul_567);  view_1633 = mul_567 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1644: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_558, [4, -1, 10, 64]);  add_558 = None
        permute_661: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1644, [0, 2, 1, 3]);  view_1644 = None
        view_1646: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_559, [4, -1, 10, 64]);  add_559 = None
        permute_662: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1646, [0, 2, 1, 3]);  view_1646 = None
        view_1648: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_560, [4, -1, 10, 64]);  add_560 = None
        permute_663: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1648, [0, 2, 1, 3]);  view_1648 = None
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_661, permute_662, permute_663, None, True)
        getitem_374: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_24[0]
        getitem_375: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_24[1]
        getitem_376: "i64[]" = _scaled_dot_product_efficient_attention_24[2]
        getitem_377: "i64[]" = _scaled_dot_product_efficient_attention_24[3];  _scaled_dot_product_efficient_attention_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_664: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_374, [0, 2, 1, 3])
        view_1649: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_664, [4, -1, 640]);  permute_664 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1650: "f32[1024, 640]" = torch.ops.aten.view.default(view_1649, [1024, 640]);  view_1649 = None
        permute_665: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1076, [1, 0]);  primals_1076 = None
        addmm_77: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1077, view_1650, permute_665);  primals_1077 = None
        view_1651: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_77, [4, 256, 640]);  addmm_77 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_666: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1078, [1, 0]);  primals_1078 = None
        mm_360: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1650, permute_666);  view_1650 = None
        permute_667: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1079, [1, 0]);  primals_1079 = None
        mm_361: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_360, permute_667)
        view_1655: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_361, [4, 256, 640]);  mm_361 = None
        mul_568: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1655, 1.0);  view_1655 = None
        add_561: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1651, mul_568);  view_1651 = mul_568 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_53: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_561, 1.0);  add_561 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_562: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_53, add_555);  div_53 = add_555 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_105 = torch.ops.aten.var_mean.correction(add_562, [2], correction = 0, keepdim = True)
        getitem_378: "f32[4, 256, 1]" = var_mean_105[0]
        getitem_379: "f32[4, 256, 1]" = var_mean_105[1];  var_mean_105 = None
        add_563: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_378, 1e-05);  getitem_378 = None
        rsqrt_105: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_563);  add_563 = None
        sub_105: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_562, getitem_379);  getitem_379 = None
        mul_569: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = None
        mul_570: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_569, primals_1080)
        add_564: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_570, primals_1081);  mul_570 = primals_1081 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1659: "f32[1024, 640]" = torch.ops.aten.view.default(add_564, [1024, 640]);  add_564 = None
        permute_668: "f32[640, 5120]" = torch.ops.aten.permute.default(primals_1082, [1, 0]);  primals_1082 = None
        addmm_78: "f32[1024, 5120]" = torch.ops.aten.addmm.default(primals_1083, view_1659, permute_668);  primals_1083 = None
        view_1660: "f32[4, 256, 5120]" = torch.ops.aten.view.default(addmm_78, [4, 256, 5120]);  addmm_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_669: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1084, [1, 0]);  primals_1084 = None
        mm_362: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1659, permute_669)
        permute_670: "f32[80, 5120]" = torch.ops.aten.permute.default(primals_1085, [1, 0]);  primals_1085 = None
        mm_363: "f32[1024, 5120]" = torch.ops.aten.mm.default(mm_362, permute_670)
        view_1664: "f32[4, 256, 5120]" = torch.ops.aten.view.default(mm_363, [4, 256, 5120]);  mm_363 = None
        mul_571: "f32[4, 256, 5120]" = torch.ops.aten.mul.Tensor(view_1664, 1.0);  view_1664 = None
        add_565: "f32[4, 256, 5120]" = torch.ops.aten.add.Tensor(view_1660, mul_571);  view_1660 = mul_571 = None
        view_1665: "f32[1024, 5120]" = torch.ops.aten.view.default(add_565, [1024, 5120]);  add_565 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1668: "f32[4, 256, 5120]" = torch.ops.aten.view.default(view_1665, [4, 256, 5120]);  view_1665 = None
        split_35 = torch.ops.aten.split.Tensor(view_1668, 2560, -1);  view_1668 = None
        getitem_383: "f32[4, 256, 2560]" = split_35[1]
        mul_572: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_383, 0.5)
        mul_573: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_383, 0.7071067811865476)
        erf_11: "f32[4, 256, 2560]" = torch.ops.aten.erf.default(mul_573);  mul_573 = None
        add_566: "f32[4, 256, 2560]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_574: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_572, add_566);  mul_572 = add_566 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_384: "f32[4, 256, 2560]" = split_35[0];  split_35 = None
        mul_575: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_384, mul_574);  mul_574 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1670: "f32[1024, 2560]" = torch.ops.aten.view.default(mul_575, [1024, 2560]);  mul_575 = None
        permute_671: "f32[2560, 640]" = torch.ops.aten.permute.default(primals_1086, [1, 0]);  primals_1086 = None
        addmm_79: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1087, view_1670, permute_671);  primals_1087 = None
        view_1671: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_79, [4, 256, 640]);  addmm_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_672: "f32[2560, 80]" = torch.ops.aten.permute.default(primals_1088, [1, 0]);  primals_1088 = None
        mm_364: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1670, permute_672)
        permute_673: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1089, [1, 0]);  primals_1089 = None
        mm_365: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_364, permute_673)
        view_1675: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_365, [4, 256, 640]);  mm_365 = None
        mul_576: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1675, 1.0);  view_1675 = None
        add_567: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1671, mul_576);  view_1671 = mul_576 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_568: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_567, add_562);  add_567 = add_562 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1679: "f32[1024, 640]" = torch.ops.aten.view.default(add_568, [1024, 640]);  add_568 = None
        permute_674: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1090, [1, 0]);  primals_1090 = None
        addmm_80: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1091, view_1679, permute_674);  primals_1091 = None
        view_1680: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_80, [4, 256, 640]);  addmm_80 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_675: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1092, [1, 0]);  primals_1092 = None
        mm_366: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1679, permute_675)
        permute_676: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1093, [1, 0]);  primals_1093 = None
        mm_367: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_366, permute_676)
        view_1684: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_367, [4, 256, 640]);  mm_367 = None
        mul_577: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1684, 1.0);  view_1684 = None
        add_569: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1680, mul_577);  view_1680 = mul_577 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1690: "f32[4, 16, 16, 640]" = torch.ops.aten.view.default(add_569, [4, 16, 16, 640]);  add_569 = None
        permute_678: "f32[4, 640, 16, 16]" = torch.ops.aten.permute.default(view_1690, [0, 3, 1, 2]);  view_1690 = None
        clone_93: "f32[4, 640, 16, 16]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_570: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(clone_93, div_51);  clone_93 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_10: "f32[4, 960, 16, 16]" = torch.ops.aten.cat.default([add_570, add_160], 1);  add_570 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1691: "f32[4, 32, 30, 256]" = torch.ops.aten.view.default(cat_10, [4, 32, 30, 256])
        var_mean_106 = torch.ops.aten.var_mean.correction(view_1691, [2, 3], correction = 0, keepdim = True)
        getitem_386: "f32[4, 32, 1, 1]" = var_mean_106[0]
        getitem_387: "f32[4, 32, 1, 1]" = var_mean_106[1];  var_mean_106 = None
        add_571: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_386, 1e-05);  getitem_386 = None
        rsqrt_106: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_571);  add_571 = None
        sub_106: "f32[4, 32, 30, 256]" = torch.ops.aten.sub.Tensor(view_1691, getitem_387);  view_1691 = None
        mul_578: "f32[4, 32, 30, 256]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = None
        view_1692: "f32[4, 960, 16, 16]" = torch.ops.aten.view.default(mul_578, [4, 960, 16, 16]);  mul_578 = None
        unsqueeze_458: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_1095, 0)
        unsqueeze_459: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        unsqueeze_461: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_1094, 0)
        unsqueeze_462: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_579: "f32[4, 960, 16, 16]" = torch.ops.aten.mul.Tensor(view_1692, unsqueeze_463);  view_1692 = unsqueeze_463 = None
        add_572: "f32[4, 960, 16, 16]" = torch.ops.aten.add.Tensor(mul_579, unsqueeze_460);  mul_579 = unsqueeze_460 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_76: "f32[4, 960, 16, 16]" = torch.ops.aten.sigmoid.default(add_572)
        mul_580: "f32[4, 960, 16, 16]" = torch.ops.aten.mul.Tensor(add_572, sigmoid_76);  add_572 = sigmoid_76 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_238: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_580, primals_1096, primals_1097, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1097 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_239: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_580, primals_1098, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_240: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_239, primals_1099, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_581: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_240, 1.0);  convolution_240 = None
        add_573: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_238, mul_581);  convolution_238 = mul_581 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_679: "f32[1280, 640]" = torch.ops.aten.permute.default(primals_1100, [1, 0]);  primals_1100 = None
        addmm_81: "f32[4, 640]" = torch.ops.aten.addmm.default(primals_1101, mul_109, permute_679);  primals_1101 = permute_679 = None
        unsqueeze_464: "f32[4, 640, 1]" = torch.ops.aten.unsqueeze.default(addmm_81, 2);  addmm_81 = None
        unsqueeze_465: "f32[4, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_574: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_573, unsqueeze_465);  add_573 = unsqueeze_465 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1693: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(add_574, [4, 32, 20, 256])
        var_mean_107 = torch.ops.aten.var_mean.correction(view_1693, [2, 3], correction = 0, keepdim = True)
        getitem_388: "f32[4, 32, 1, 1]" = var_mean_107[0]
        getitem_389: "f32[4, 32, 1, 1]" = var_mean_107[1];  var_mean_107 = None
        add_575: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_388, 1e-05);  getitem_388 = None
        rsqrt_107: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_575);  add_575 = None
        sub_107: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1693, getitem_389);  view_1693 = None
        mul_583: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = None
        view_1694: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_583, [4, 640, 16, 16]);  mul_583 = None
        unsqueeze_466: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1103, 0)
        unsqueeze_467: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1102, 0)
        unsqueeze_470: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_584: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1694, unsqueeze_471);  view_1694 = unsqueeze_471 = None
        add_576: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_468);  mul_584 = unsqueeze_468 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_78: "f32[4, 640, 16, 16]" = torch.ops.aten.sigmoid.default(add_576)
        mul_585: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(add_576, sigmoid_78);  add_576 = sigmoid_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_241: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_585, primals_1104, primals_1105, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1105 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_242: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(mul_585, primals_1106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_243: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_242, primals_1107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_586: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_243, 1.0);  convolution_243 = None
        add_577: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_241, mul_586);  convolution_241 = mul_586 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_244: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(cat_10, primals_1108, primals_1109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_1109 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_245: "f32[4, 80, 16, 16]" = torch.ops.aten.convolution.default(cat_10, primals_1110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_246: "f32[4, 640, 16, 16]" = torch.ops.aten.convolution.default(convolution_245, primals_1111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_587: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(convolution_246, 1.0);  convolution_246 = None
        add_578: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(convolution_244, mul_587);  convolution_244 = mul_587 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_579: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_578, add_577);  add_578 = add_577 = None
        div_54: "f32[4, 640, 16, 16]" = torch.ops.aten.div.Tensor(add_579, 1.0);  add_579 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1695: "f32[4, 32, 20, 256]" = torch.ops.aten.view.default(div_54, [4, 32, 20, 256])
        var_mean_108 = torch.ops.aten.var_mean.correction(view_1695, [2, 3], correction = 0, keepdim = True)
        getitem_390: "f32[4, 32, 1, 1]" = var_mean_108[0]
        getitem_391: "f32[4, 32, 1, 1]" = var_mean_108[1];  var_mean_108 = None
        add_580: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_390, 1e-06);  getitem_390 = None
        rsqrt_108: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_580);  add_580 = None
        sub_108: "f32[4, 32, 20, 256]" = torch.ops.aten.sub.Tensor(view_1695, getitem_391);  view_1695 = None
        mul_588: "f32[4, 32, 20, 256]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = None
        view_1696: "f32[4, 640, 16, 16]" = torch.ops.aten.view.default(mul_588, [4, 640, 16, 16]);  mul_588 = None
        unsqueeze_472: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1113, 0);  primals_1113 = None
        unsqueeze_473: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1112, 0)
        unsqueeze_476: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_589: "f32[4, 640, 16, 16]" = torch.ops.aten.mul.Tensor(view_1696, unsqueeze_477);  view_1696 = unsqueeze_477 = None
        add_581: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_589, unsqueeze_474);  mul_589 = unsqueeze_474 = None
        squeeze_144: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_391, [2, 3]);  getitem_391 = None
        squeeze_145: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_108, [2, 3]);  rsqrt_108 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_680: "f32[4, 16, 16, 640]" = torch.ops.aten.permute.default(add_581, [0, 2, 3, 1]);  add_581 = None
        view_1697: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_680, [4, 256, 640]);  permute_680 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_681: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1114, [1, 0])
        expand_31: "f32[4, 256, 640]" = torch.ops.aten.expand.default(view_1697, [4, 256, 640])
        expand_32: "f32[4, 640, 640]" = torch.ops.aten.expand.default(permute_681, [4, 640, 640]);  permute_681 = None
        bmm_15: "f32[4, 256, 640]" = torch.ops.aten.bmm.default(expand_31, expand_32);  expand_31 = expand_32 = None
        add_582: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(bmm_15, primals_1115);  bmm_15 = primals_1115 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_682: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1116, [1, 0]);  primals_1116 = None
        clone_95: "f32[4, 256, 640]" = torch.ops.aten.clone.default(view_1697, memory_format = torch.contiguous_format);  view_1697 = None
        view_1701: "f32[1024, 640]" = torch.ops.aten.view.default(clone_95, [1024, 640]);  clone_95 = None
        mm_368: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1701, permute_682)
        permute_683: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1117, [1, 0]);  primals_1117 = None
        mm_369: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_368, permute_683)
        view_1704: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_369, [4, 256, 640]);  mm_369 = None
        mul_590: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1704, 1.0);  view_1704 = None
        add_583: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_582, mul_590);  add_582 = mul_590 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_109 = torch.ops.aten.var_mean.correction(add_583, [2], correction = 0, keepdim = True)
        getitem_392: "f32[4, 256, 1]" = var_mean_109[0]
        getitem_393: "f32[4, 256, 1]" = var_mean_109[1];  var_mean_109 = None
        add_584: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_392, 1e-05);  getitem_392 = None
        rsqrt_109: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_584);  add_584 = None
        sub_109: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_583, getitem_393);  getitem_393 = None
        mul_591: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = None
        mul_592: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_591, primals_1118)
        add_585: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_592, primals_1119);  mul_592 = primals_1119 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_684: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1120, [1, 0]);  primals_1120 = None
        view_1705: "f32[1024, 640]" = torch.ops.aten.view.default(add_585, [1024, 640]);  add_585 = None
        mm_370: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1705, permute_684)
        view_1706: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_370, [4, 256, 640]);  mm_370 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_685: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1121, [1, 0]);  primals_1121 = None
        mm_371: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1705, permute_685)
        permute_686: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1122, [1, 0]);  primals_1122 = None
        mm_372: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_371, permute_686)
        view_1710: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_372, [4, 256, 640]);  mm_372 = None
        mul_593: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1710, 1.0);  view_1710 = None
        add_586: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1706, mul_593);  view_1706 = mul_593 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_687: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1123, [1, 0]);  primals_1123 = None
        mm_373: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1705, permute_687)
        view_1714: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_373, [4, 256, 640]);  mm_373 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_688: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1124, [1, 0]);  primals_1124 = None
        mm_374: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1705, permute_688)
        permute_689: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1125, [1, 0]);  primals_1125 = None
        mm_375: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_374, permute_689)
        view_1718: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_375, [4, 256, 640]);  mm_375 = None
        mul_594: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1718, 1.0);  view_1718 = None
        add_587: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1714, mul_594);  view_1714 = mul_594 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_690: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1126, [1, 0]);  primals_1126 = None
        mm_376: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1705, permute_690)
        view_1722: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_376, [4, 256, 640]);  mm_376 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_691: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1127, [1, 0]);  primals_1127 = None
        mm_377: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1705, permute_691)
        permute_692: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1128, [1, 0]);  primals_1128 = None
        mm_378: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_377, permute_692)
        view_1726: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_378, [4, 256, 640]);  mm_378 = None
        mul_595: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1726, 1.0);  view_1726 = None
        add_588: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1722, mul_595);  view_1722 = mul_595 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1733: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_586, [4, -1, 10, 64]);  add_586 = None
        permute_696: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1733, [0, 2, 1, 3]);  view_1733 = None
        view_1735: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_587, [4, -1, 10, 64]);  add_587 = None
        permute_697: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1735, [0, 2, 1, 3]);  view_1735 = None
        view_1737: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_588, [4, -1, 10, 64]);  add_588 = None
        permute_698: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1737, [0, 2, 1, 3]);  view_1737 = None
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_696, permute_697, permute_698, None, True)
        getitem_394: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_25[0]
        getitem_395: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_25[1]
        getitem_396: "i64[]" = _scaled_dot_product_efficient_attention_25[2]
        getitem_397: "i64[]" = _scaled_dot_product_efficient_attention_25[3];  _scaled_dot_product_efficient_attention_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_699: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_394, [0, 2, 1, 3])
        view_1738: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_699, [4, -1, 640]);  permute_699 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1739: "f32[1024, 640]" = torch.ops.aten.view.default(view_1738, [1024, 640]);  view_1738 = None
        permute_700: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1129, [1, 0]);  primals_1129 = None
        addmm_82: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1130, view_1739, permute_700);  primals_1130 = None
        view_1740: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_82, [4, 256, 640]);  addmm_82 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_701: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1131, [1, 0]);  primals_1131 = None
        mm_379: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1739, permute_701);  view_1739 = None
        permute_702: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1132, [1, 0]);  primals_1132 = None
        mm_380: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_379, permute_702)
        view_1744: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_380, [4, 256, 640]);  mm_380 = None
        mul_596: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1744, 1.0);  view_1744 = None
        add_589: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1740, mul_596);  view_1740 = mul_596 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_55: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_589, 1.0);  add_589 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_590: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_55, add_583);  div_55 = add_583 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_110 = torch.ops.aten.var_mean.correction(add_590, [2], correction = 0, keepdim = True)
        getitem_398: "f32[4, 256, 1]" = var_mean_110[0]
        getitem_399: "f32[4, 256, 1]" = var_mean_110[1];  var_mean_110 = None
        add_591: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_398, 1e-05);  getitem_398 = None
        rsqrt_110: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_591);  add_591 = None
        sub_110: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_590, getitem_399);  getitem_399 = None
        mul_597: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = None
        mul_598: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_597, primals_1133)
        add_592: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_598, primals_1134);  mul_598 = primals_1134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_703: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1135, [1, 0]);  primals_1135 = None
        view_1748: "f32[1024, 640]" = torch.ops.aten.view.default(add_592, [1024, 640]);  add_592 = None
        mm_381: "f32[1024, 640]" = torch.ops.aten.mm.default(view_1748, permute_703)
        view_1749: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_381, [4, 256, 640]);  mm_381 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_704: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1136, [1, 0]);  primals_1136 = None
        mm_382: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1748, permute_704)
        permute_705: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1137, [1, 0]);  primals_1137 = None
        mm_383: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_382, permute_705)
        view_1753: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_383, [4, 256, 640]);  mm_383 = None
        mul_599: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1753, 1.0);  view_1753 = None
        add_593: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1749, mul_599);  view_1749 = mul_599 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_706: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1138, [1, 0]);  primals_1138 = None
        mm_384: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_706);  permute_706 = None
        view_1757: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_384, [4, 77, 640]);  mm_384 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_707: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1139, [1, 0]);  primals_1139 = None
        mm_385: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_707);  permute_707 = None
        permute_708: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1140, [1, 0]);  primals_1140 = None
        mm_386: "f32[308, 640]" = torch.ops.aten.mm.default(mm_385, permute_708)
        view_1761: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_386, [4, 77, 640]);  mm_386 = None
        mul_600: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1761, 1.0);  view_1761 = None
        add_594: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1757, mul_600);  view_1757 = mul_600 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_709: "f32[1024, 640]" = torch.ops.aten.permute.default(primals_1141, [1, 0]);  primals_1141 = None
        mm_387: "f32[308, 640]" = torch.ops.aten.mm.default(view_148, permute_709);  permute_709 = None
        view_1765: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_387, [4, 77, 640]);  mm_387 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_710: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1142, [1, 0]);  primals_1142 = None
        mm_388: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_710);  permute_710 = None
        permute_711: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1143, [1, 0]);  primals_1143 = None
        mm_389: "f32[308, 640]" = torch.ops.aten.mm.default(mm_388, permute_711)
        view_1769: "f32[4, 77, 640]" = torch.ops.aten.view.default(mm_389, [4, 77, 640]);  mm_389 = None
        mul_601: "f32[4, 77, 640]" = torch.ops.aten.mul.Tensor(view_1769, 1.0);  view_1769 = None
        add_595: "f32[4, 77, 640]" = torch.ops.aten.add.Tensor(view_1765, mul_601);  view_1765 = mul_601 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1776: "f32[4, 256, 10, 64]" = torch.ops.aten.view.default(add_593, [4, -1, 10, 64]);  add_593 = None
        permute_715: "f32[4, 10, 256, 64]" = torch.ops.aten.permute.default(view_1776, [0, 2, 1, 3]);  view_1776 = None
        view_1778: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_594, [4, -1, 10, 64]);  add_594 = None
        permute_716: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1778, [0, 2, 1, 3]);  view_1778 = None
        view_1780: "f32[4, 77, 10, 64]" = torch.ops.aten.view.default(add_595, [4, -1, 10, 64]);  add_595 = None
        permute_717: "f32[4, 10, 77, 64]" = torch.ops.aten.permute.default(view_1780, [0, 2, 1, 3]);  view_1780 = None
        _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_715, permute_716, permute_717, None, True)
        getitem_400: "f32[4, 10, 256, 64]" = _scaled_dot_product_efficient_attention_26[0]
        getitem_401: "f32[4, 10, 256]" = _scaled_dot_product_efficient_attention_26[1]
        getitem_402: "i64[]" = _scaled_dot_product_efficient_attention_26[2]
        getitem_403: "i64[]" = _scaled_dot_product_efficient_attention_26[3];  _scaled_dot_product_efficient_attention_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_718: "f32[4, 256, 10, 64]" = torch.ops.aten.permute.default(getitem_400, [0, 2, 1, 3])
        view_1781: "f32[4, 256, 640]" = torch.ops.aten.view.default(permute_718, [4, -1, 640]);  permute_718 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1782: "f32[1024, 640]" = torch.ops.aten.view.default(view_1781, [1024, 640]);  view_1781 = None
        permute_719: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1144, [1, 0]);  primals_1144 = None
        addmm_83: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1145, view_1782, permute_719);  primals_1145 = None
        view_1783: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_83, [4, 256, 640]);  addmm_83 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_720: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1146, [1, 0]);  primals_1146 = None
        mm_390: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1782, permute_720);  view_1782 = None
        permute_721: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1147, [1, 0]);  primals_1147 = None
        mm_391: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_390, permute_721)
        view_1787: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_391, [4, 256, 640]);  mm_391 = None
        mul_602: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1787, 1.0);  view_1787 = None
        add_596: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1783, mul_602);  view_1783 = mul_602 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_56: "f32[4, 256, 640]" = torch.ops.aten.div.Tensor(add_596, 1.0);  add_596 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_597: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(div_56, add_590);  div_56 = add_590 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_111 = torch.ops.aten.var_mean.correction(add_597, [2], correction = 0, keepdim = True)
        getitem_404: "f32[4, 256, 1]" = var_mean_111[0]
        getitem_405: "f32[4, 256, 1]" = var_mean_111[1];  var_mean_111 = None
        add_598: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(getitem_404, 1e-05);  getitem_404 = None
        rsqrt_111: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_598);  add_598 = None
        sub_111: "f32[4, 256, 640]" = torch.ops.aten.sub.Tensor(add_597, getitem_405);  getitem_405 = None
        mul_603: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = None
        mul_604: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(mul_603, primals_1148)
        add_599: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(mul_604, primals_1149);  mul_604 = primals_1149 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1791: "f32[1024, 640]" = torch.ops.aten.view.default(add_599, [1024, 640]);  add_599 = None
        permute_722: "f32[640, 5120]" = torch.ops.aten.permute.default(primals_1150, [1, 0]);  primals_1150 = None
        addmm_84: "f32[1024, 5120]" = torch.ops.aten.addmm.default(primals_1151, view_1791, permute_722);  primals_1151 = None
        view_1792: "f32[4, 256, 5120]" = torch.ops.aten.view.default(addmm_84, [4, 256, 5120]);  addmm_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_723: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1152, [1, 0]);  primals_1152 = None
        mm_392: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1791, permute_723)
        permute_724: "f32[80, 5120]" = torch.ops.aten.permute.default(primals_1153, [1, 0]);  primals_1153 = None
        mm_393: "f32[1024, 5120]" = torch.ops.aten.mm.default(mm_392, permute_724)
        view_1796: "f32[4, 256, 5120]" = torch.ops.aten.view.default(mm_393, [4, 256, 5120]);  mm_393 = None
        mul_605: "f32[4, 256, 5120]" = torch.ops.aten.mul.Tensor(view_1796, 1.0);  view_1796 = None
        add_600: "f32[4, 256, 5120]" = torch.ops.aten.add.Tensor(view_1792, mul_605);  view_1792 = mul_605 = None
        view_1797: "f32[1024, 5120]" = torch.ops.aten.view.default(add_600, [1024, 5120]);  add_600 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1800: "f32[4, 256, 5120]" = torch.ops.aten.view.default(view_1797, [4, 256, 5120]);  view_1797 = None
        split_38 = torch.ops.aten.split.Tensor(view_1800, 2560, -1);  view_1800 = None
        getitem_409: "f32[4, 256, 2560]" = split_38[1]
        mul_606: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_409, 0.5)
        mul_607: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_409, 0.7071067811865476)
        erf_12: "f32[4, 256, 2560]" = torch.ops.aten.erf.default(mul_607);  mul_607 = None
        add_601: "f32[4, 256, 2560]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_608: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_606, add_601);  mul_606 = add_601 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_410: "f32[4, 256, 2560]" = split_38[0];  split_38 = None
        mul_609: "f32[4, 256, 2560]" = torch.ops.aten.mul.Tensor(getitem_410, mul_608);  mul_608 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1802: "f32[1024, 2560]" = torch.ops.aten.view.default(mul_609, [1024, 2560]);  mul_609 = None
        permute_725: "f32[2560, 640]" = torch.ops.aten.permute.default(primals_1154, [1, 0]);  primals_1154 = None
        addmm_85: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1155, view_1802, permute_725);  primals_1155 = None
        view_1803: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_85, [4, 256, 640]);  addmm_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_726: "f32[2560, 80]" = torch.ops.aten.permute.default(primals_1156, [1, 0]);  primals_1156 = None
        mm_394: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1802, permute_726)
        permute_727: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1157, [1, 0]);  primals_1157 = None
        mm_395: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_394, permute_727)
        view_1807: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_395, [4, 256, 640]);  mm_395 = None
        mul_610: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1807, 1.0);  view_1807 = None
        add_602: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1803, mul_610);  view_1803 = mul_610 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_603: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(add_602, add_597);  add_602 = add_597 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1811: "f32[1024, 640]" = torch.ops.aten.view.default(add_603, [1024, 640]);  add_603 = None
        permute_728: "f32[640, 640]" = torch.ops.aten.permute.default(primals_1158, [1, 0]);  primals_1158 = None
        addmm_86: "f32[1024, 640]" = torch.ops.aten.addmm.default(primals_1159, view_1811, permute_728);  primals_1159 = None
        view_1812: "f32[4, 256, 640]" = torch.ops.aten.view.default(addmm_86, [4, 256, 640]);  addmm_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_729: "f32[640, 80]" = torch.ops.aten.permute.default(primals_1160, [1, 0]);  primals_1160 = None
        mm_396: "f32[1024, 80]" = torch.ops.aten.mm.default(view_1811, permute_729)
        permute_730: "f32[80, 640]" = torch.ops.aten.permute.default(primals_1161, [1, 0]);  primals_1161 = None
        mm_397: "f32[1024, 640]" = torch.ops.aten.mm.default(mm_396, permute_730)
        view_1816: "f32[4, 256, 640]" = torch.ops.aten.view.default(mm_397, [4, 256, 640]);  mm_397 = None
        mul_611: "f32[4, 256, 640]" = torch.ops.aten.mul.Tensor(view_1816, 1.0);  view_1816 = None
        add_604: "f32[4, 256, 640]" = torch.ops.aten.add.Tensor(view_1812, mul_611);  view_1812 = mul_611 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1822: "f32[4, 16, 16, 640]" = torch.ops.aten.view.default(add_604, [4, 16, 16, 640]);  add_604 = None
        permute_732: "f32[4, 640, 16, 16]" = torch.ops.aten.permute.default(view_1822, [0, 3, 1, 2]);  view_1822 = None
        clone_99: "f32[4, 640, 16, 16]" = torch.ops.aten.clone.default(permute_732, memory_format = torch.contiguous_format);  permute_732 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_605: "f32[4, 640, 16, 16]" = torch.ops.aten.add.Tensor(clone_99, div_54);  clone_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota_5: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_612: "i64[32]" = torch.ops.aten.mul.Tensor(iota_5, 1);  iota_5 = None
        add_606: "i64[32]" = torch.ops.aten.add.Tensor(mul_612, 0);  mul_612 = None
        convert_element_type_10: "f32[32]" = torch.ops.prims.convert_element_type.default(add_606, torch.float32);  add_606 = None
        add_607: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_10, 0.0);  convert_element_type_10 = None
        mul_613: "f32[32]" = torch.ops.aten.mul.Tensor(add_607, 0.5);  add_607 = None
        convert_element_type_11: "i64[32]" = torch.ops.prims.convert_element_type.default(mul_613, torch.int64);  mul_613 = None
        unsqueeze_478: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, -1)
        _unsafe_index_2: "f32[4, 640, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(add_605, [None, None, unsqueeze_478, convert_element_type_11]);  add_605 = unsqueeze_478 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_247: "f32[4, 640, 32, 32]" = torch.ops.aten.convolution.default(_unsafe_index_2, primals_1162, primals_1163, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1163 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_248: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(_unsafe_index_2, primals_1164, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_249: "f32[4, 640, 32, 32]" = torch.ops.aten.convolution.default(convolution_248, primals_1165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_616: "f32[4, 640, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_249, 1.0);  convolution_249 = None
        add_610: "f32[4, 640, 32, 32]" = torch.ops.aten.add.Tensor(convolution_247, mul_616);  convolution_247 = mul_616 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_11: "f32[4, 960, 32, 32]" = torch.ops.aten.cat.default([add_610, add_159], 1);  add_610 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1823: "f32[4, 32, 30, 1024]" = torch.ops.aten.view.default(cat_11, [4, 32, 30, 1024])
        var_mean_112 = torch.ops.aten.var_mean.correction(view_1823, [2, 3], correction = 0, keepdim = True)
        getitem_412: "f32[4, 32, 1, 1]" = var_mean_112[0]
        getitem_413: "f32[4, 32, 1, 1]" = var_mean_112[1];  var_mean_112 = None
        add_611: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_412, 1e-05);  getitem_412 = None
        rsqrt_112: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_611);  add_611 = None
        sub_112: "f32[4, 32, 30, 1024]" = torch.ops.aten.sub.Tensor(view_1823, getitem_413);  view_1823 = None
        mul_617: "f32[4, 32, 30, 1024]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = None
        view_1824: "f32[4, 960, 32, 32]" = torch.ops.aten.view.default(mul_617, [4, 960, 32, 32]);  mul_617 = None
        unsqueeze_479: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_1167, 0)
        unsqueeze_480: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        unsqueeze_482: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_1166, 0)
        unsqueeze_483: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_618: "f32[4, 960, 32, 32]" = torch.ops.aten.mul.Tensor(view_1824, unsqueeze_484);  view_1824 = unsqueeze_484 = None
        add_612: "f32[4, 960, 32, 32]" = torch.ops.aten.add.Tensor(mul_618, unsqueeze_481);  mul_618 = unsqueeze_481 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_79: "f32[4, 960, 32, 32]" = torch.ops.aten.sigmoid.default(add_612)
        mul_619: "f32[4, 960, 32, 32]" = torch.ops.aten.mul.Tensor(add_612, sigmoid_79);  add_612 = sigmoid_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_250: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_619, primals_1168, primals_1169, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1169 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_251: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_619, primals_1170, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_252: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_251, primals_1171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_620: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_252, 1.0);  convolution_252 = None
        add_613: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_250, mul_620);  convolution_250 = mul_620 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_733: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1172, [1, 0]);  primals_1172 = None
        addmm_87: "f32[4, 320]" = torch.ops.aten.addmm.default(primals_1173, mul_109, permute_733);  primals_1173 = permute_733 = None
        unsqueeze_485: "f32[4, 320, 1]" = torch.ops.aten.unsqueeze.default(addmm_87, 2);  addmm_87 = None
        unsqueeze_486: "f32[4, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_614: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_613, unsqueeze_486);  add_613 = unsqueeze_486 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1825: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_614, [4, 32, 10, 1024])
        var_mean_113 = torch.ops.aten.var_mean.correction(view_1825, [2, 3], correction = 0, keepdim = True)
        getitem_414: "f32[4, 32, 1, 1]" = var_mean_113[0]
        getitem_415: "f32[4, 32, 1, 1]" = var_mean_113[1];  var_mean_113 = None
        add_615: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_414, 1e-05);  getitem_414 = None
        rsqrt_113: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_615);  add_615 = None
        sub_113: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_1825, getitem_415);  view_1825 = None
        mul_622: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = None
        view_1826: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_622, [4, 320, 32, 32]);  mul_622 = None
        unsqueeze_487: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1175, 0)
        unsqueeze_488: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1174, 0)
        unsqueeze_491: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_623: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_1826, unsqueeze_492);  view_1826 = unsqueeze_492 = None
        add_616: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_489);  mul_623 = unsqueeze_489 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_81: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_616)
        mul_624: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_616, sigmoid_81);  add_616 = sigmoid_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_253: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_624, primals_1176, primals_1177, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1177 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_254: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_624, primals_1178, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_255: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_254, primals_1179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_625: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_255, 1.0);  convolution_255 = None
        add_617: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_253, mul_625);  convolution_253 = mul_625 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_256: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(cat_11, primals_1180, primals_1181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_1181 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_257: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(cat_11, primals_1182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_258: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_257, primals_1183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_626: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_258, 1.0);  convolution_258 = None
        add_618: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_256, mul_626);  convolution_256 = mul_626 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_619: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_618, add_617);  add_618 = add_617 = None
        div_57: "f32[4, 320, 32, 32]" = torch.ops.aten.div.Tensor(add_619, 1.0);  add_619 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1827: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(div_57, [4, 32, 10, 1024])
        var_mean_114 = torch.ops.aten.var_mean.correction(view_1827, [2, 3], correction = 0, keepdim = True)
        getitem_416: "f32[4, 32, 1, 1]" = var_mean_114[0]
        getitem_417: "f32[4, 32, 1, 1]" = var_mean_114[1];  var_mean_114 = None
        add_620: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-06);  getitem_416 = None
        rsqrt_114: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_620);  add_620 = None
        sub_114: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_1827, getitem_417);  view_1827 = None
        mul_627: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = None
        view_1828: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_627, [4, 320, 32, 32]);  mul_627 = None
        unsqueeze_493: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1185, 0);  primals_1185 = None
        unsqueeze_494: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1184, 0)
        unsqueeze_497: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        mul_628: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_1828, unsqueeze_498);  view_1828 = unsqueeze_498 = None
        add_621: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_628, unsqueeze_495);  mul_628 = unsqueeze_495 = None
        squeeze_150: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_417, [2, 3]);  getitem_417 = None
        squeeze_151: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_114, [2, 3]);  rsqrt_114 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_734: "f32[4, 32, 32, 320]" = torch.ops.aten.permute.default(add_621, [0, 2, 3, 1]);  add_621 = None
        view_1829: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_734, [4, 1024, 320]);  permute_734 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_735: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1186, [1, 0])
        expand_33: "f32[4, 1024, 320]" = torch.ops.aten.expand.default(view_1829, [4, 1024, 320])
        expand_34: "f32[4, 320, 320]" = torch.ops.aten.expand.default(permute_735, [4, 320, 320]);  permute_735 = None
        bmm_16: "f32[4, 1024, 320]" = torch.ops.aten.bmm.default(expand_33, expand_34);  expand_33 = expand_34 = None
        add_622: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(bmm_16, primals_1187);  bmm_16 = primals_1187 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_736: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1188, [1, 0]);  primals_1188 = None
        clone_101: "f32[4, 1024, 320]" = torch.ops.aten.clone.default(view_1829, memory_format = torch.contiguous_format);  view_1829 = None
        view_1833: "f32[4096, 320]" = torch.ops.aten.view.default(clone_101, [4096, 320]);  clone_101 = None
        mm_398: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1833, permute_736)
        permute_737: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1189, [1, 0]);  primals_1189 = None
        mm_399: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_398, permute_737)
        view_1836: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_399, [4, 1024, 320]);  mm_399 = None
        mul_629: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1836, 1.0);  view_1836 = None
        add_623: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_622, mul_629);  add_622 = mul_629 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_115 = torch.ops.aten.var_mean.correction(add_623, [2], correction = 0, keepdim = True)
        getitem_418: "f32[4, 1024, 1]" = var_mean_115[0]
        getitem_419: "f32[4, 1024, 1]" = var_mean_115[1];  var_mean_115 = None
        add_624: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_418, 1e-05);  getitem_418 = None
        rsqrt_115: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_624);  add_624 = None
        sub_115: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_623, getitem_419);  getitem_419 = None
        mul_630: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = None
        mul_631: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_630, primals_1190)
        add_625: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_631, primals_1191);  mul_631 = primals_1191 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_738: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1192, [1, 0]);  primals_1192 = None
        view_1837: "f32[4096, 320]" = torch.ops.aten.view.default(add_625, [4096, 320]);  add_625 = None
        mm_400: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1837, permute_738)
        view_1838: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_400, [4, 1024, 320]);  mm_400 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_739: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1193, [1, 0]);  primals_1193 = None
        mm_401: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1837, permute_739)
        permute_740: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1194, [1, 0]);  primals_1194 = None
        mm_402: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_401, permute_740)
        view_1842: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_402, [4, 1024, 320]);  mm_402 = None
        mul_632: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1842, 1.0);  view_1842 = None
        add_626: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1838, mul_632);  view_1838 = mul_632 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_741: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1195, [1, 0]);  primals_1195 = None
        mm_403: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1837, permute_741)
        view_1846: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_403, [4, 1024, 320]);  mm_403 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_742: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1196, [1, 0]);  primals_1196 = None
        mm_404: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1837, permute_742)
        permute_743: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1197, [1, 0]);  primals_1197 = None
        mm_405: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_404, permute_743)
        view_1850: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_405, [4, 1024, 320]);  mm_405 = None
        mul_633: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1850, 1.0);  view_1850 = None
        add_627: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1846, mul_633);  view_1846 = mul_633 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_744: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1198, [1, 0]);  primals_1198 = None
        mm_406: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1837, permute_744)
        view_1854: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_406, [4, 1024, 320]);  mm_406 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_745: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1199, [1, 0]);  primals_1199 = None
        mm_407: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1837, permute_745)
        permute_746: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1200, [1, 0]);  primals_1200 = None
        mm_408: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_407, permute_746)
        view_1858: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_408, [4, 1024, 320]);  mm_408 = None
        mul_634: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1858, 1.0);  view_1858 = None
        add_628: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1854, mul_634);  view_1854 = mul_634 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1865: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_626, [4, -1, 5, 64]);  add_626 = None
        permute_750: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1865, [0, 2, 1, 3]);  view_1865 = None
        view_1867: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_627, [4, -1, 5, 64]);  add_627 = None
        permute_751: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1867, [0, 2, 1, 3]);  view_1867 = None
        view_1869: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_628, [4, -1, 5, 64]);  add_628 = None
        permute_752: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1869, [0, 2, 1, 3]);  view_1869 = None
        _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_750, permute_751, permute_752, None, True)
        getitem_420: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_27[0]
        getitem_421: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_27[1]
        getitem_422: "i64[]" = _scaled_dot_product_efficient_attention_27[2]
        getitem_423: "i64[]" = _scaled_dot_product_efficient_attention_27[3];  _scaled_dot_product_efficient_attention_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_753: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_420, [0, 2, 1, 3])
        view_1870: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_753, [4, -1, 320]);  permute_753 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1871: "f32[4096, 320]" = torch.ops.aten.view.default(view_1870, [4096, 320]);  view_1870 = None
        permute_754: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1201, [1, 0]);  primals_1201 = None
        addmm_88: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1202, view_1871, permute_754);  primals_1202 = None
        view_1872: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_88, [4, 1024, 320]);  addmm_88 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_755: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1203, [1, 0]);  primals_1203 = None
        mm_409: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1871, permute_755);  view_1871 = None
        permute_756: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1204, [1, 0]);  primals_1204 = None
        mm_410: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_409, permute_756)
        view_1876: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_410, [4, 1024, 320]);  mm_410 = None
        mul_635: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1876, 1.0);  view_1876 = None
        add_629: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1872, mul_635);  view_1872 = mul_635 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_58: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_629, 1.0);  add_629 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_630: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_58, add_623);  div_58 = add_623 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_116 = torch.ops.aten.var_mean.correction(add_630, [2], correction = 0, keepdim = True)
        getitem_424: "f32[4, 1024, 1]" = var_mean_116[0]
        getitem_425: "f32[4, 1024, 1]" = var_mean_116[1];  var_mean_116 = None
        add_631: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_424, 1e-05);  getitem_424 = None
        rsqrt_116: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_631);  add_631 = None
        sub_116: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_630, getitem_425);  getitem_425 = None
        mul_636: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = None
        mul_637: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_636, primals_1205)
        add_632: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_637, primals_1206);  mul_637 = primals_1206 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_757: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1207, [1, 0]);  primals_1207 = None
        view_1880: "f32[4096, 320]" = torch.ops.aten.view.default(add_632, [4096, 320]);  add_632 = None
        mm_411: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1880, permute_757)
        view_1881: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_411, [4, 1024, 320]);  mm_411 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_758: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1208, [1, 0]);  primals_1208 = None
        mm_412: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1880, permute_758)
        permute_759: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1209, [1, 0]);  primals_1209 = None
        mm_413: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_412, permute_759)
        view_1885: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_413, [4, 1024, 320]);  mm_413 = None
        mul_638: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1885, 1.0);  view_1885 = None
        add_633: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1881, mul_638);  view_1881 = mul_638 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_760: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1210, [1, 0]);  primals_1210 = None
        mm_414: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_760);  permute_760 = None
        view_1889: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_414, [4, 77, 320]);  mm_414 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_761: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1211, [1, 0]);  primals_1211 = None
        mm_415: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_761);  permute_761 = None
        permute_762: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1212, [1, 0]);  primals_1212 = None
        mm_416: "f32[308, 320]" = torch.ops.aten.mm.default(mm_415, permute_762)
        view_1893: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_416, [4, 77, 320]);  mm_416 = None
        mul_639: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_1893, 1.0);  view_1893 = None
        add_634: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_1889, mul_639);  view_1889 = mul_639 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_763: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1213, [1, 0]);  primals_1213 = None
        mm_417: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_763);  permute_763 = None
        view_1897: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_417, [4, 77, 320]);  mm_417 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_764: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1214, [1, 0]);  primals_1214 = None
        mm_418: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_764);  permute_764 = None
        permute_765: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1215, [1, 0]);  primals_1215 = None
        mm_419: "f32[308, 320]" = torch.ops.aten.mm.default(mm_418, permute_765)
        view_1901: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_419, [4, 77, 320]);  mm_419 = None
        mul_640: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_1901, 1.0);  view_1901 = None
        add_635: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_1897, mul_640);  view_1897 = mul_640 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1908: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_633, [4, -1, 5, 64]);  add_633 = None
        permute_769: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1908, [0, 2, 1, 3]);  view_1908 = None
        view_1910: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_634, [4, -1, 5, 64]);  add_634 = None
        permute_770: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_1910, [0, 2, 1, 3]);  view_1910 = None
        view_1912: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_635, [4, -1, 5, 64]);  add_635 = None
        permute_771: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_1912, [0, 2, 1, 3]);  view_1912 = None
        _scaled_dot_product_efficient_attention_28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_769, permute_770, permute_771, None, True)
        getitem_426: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_28[0]
        getitem_427: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_28[1]
        getitem_428: "i64[]" = _scaled_dot_product_efficient_attention_28[2]
        getitem_429: "i64[]" = _scaled_dot_product_efficient_attention_28[3];  _scaled_dot_product_efficient_attention_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_772: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_426, [0, 2, 1, 3])
        view_1913: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_772, [4, -1, 320]);  permute_772 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1914: "f32[4096, 320]" = torch.ops.aten.view.default(view_1913, [4096, 320]);  view_1913 = None
        permute_773: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1216, [1, 0]);  primals_1216 = None
        addmm_89: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1217, view_1914, permute_773);  primals_1217 = None
        view_1915: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_89, [4, 1024, 320]);  addmm_89 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_774: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1218, [1, 0]);  primals_1218 = None
        mm_420: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1914, permute_774);  view_1914 = None
        permute_775: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1219, [1, 0]);  primals_1219 = None
        mm_421: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_420, permute_775)
        view_1919: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_421, [4, 1024, 320]);  mm_421 = None
        mul_641: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1919, 1.0);  view_1919 = None
        add_636: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1915, mul_641);  view_1915 = mul_641 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_59: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_636, 1.0);  add_636 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_637: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_59, add_630);  div_59 = add_630 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_117 = torch.ops.aten.var_mean.correction(add_637, [2], correction = 0, keepdim = True)
        getitem_430: "f32[4, 1024, 1]" = var_mean_117[0]
        getitem_431: "f32[4, 1024, 1]" = var_mean_117[1];  var_mean_117 = None
        add_638: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-05);  getitem_430 = None
        rsqrt_117: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_638);  add_638 = None
        sub_117: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_637, getitem_431);  getitem_431 = None
        mul_642: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = None
        mul_643: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_642, primals_1220)
        add_639: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_643, primals_1221);  mul_643 = primals_1221 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1923: "f32[4096, 320]" = torch.ops.aten.view.default(add_639, [4096, 320]);  add_639 = None
        permute_776: "f32[320, 2560]" = torch.ops.aten.permute.default(primals_1222, [1, 0]);  primals_1222 = None
        addmm_90: "f32[4096, 2560]" = torch.ops.aten.addmm.default(primals_1223, view_1923, permute_776);  primals_1223 = None
        view_1924: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(addmm_90, [4, 1024, 2560]);  addmm_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_777: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1224, [1, 0]);  primals_1224 = None
        mm_422: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1923, permute_777)
        permute_778: "f32[80, 2560]" = torch.ops.aten.permute.default(primals_1225, [1, 0]);  primals_1225 = None
        mm_423: "f32[4096, 2560]" = torch.ops.aten.mm.default(mm_422, permute_778)
        view_1928: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(mm_423, [4, 1024, 2560]);  mm_423 = None
        mul_644: "f32[4, 1024, 2560]" = torch.ops.aten.mul.Tensor(view_1928, 1.0);  view_1928 = None
        add_640: "f32[4, 1024, 2560]" = torch.ops.aten.add.Tensor(view_1924, mul_644);  view_1924 = mul_644 = None
        view_1929: "f32[4096, 2560]" = torch.ops.aten.view.default(add_640, [4096, 2560]);  add_640 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_1932: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(view_1929, [4, 1024, 2560]);  view_1929 = None
        split_41 = torch.ops.aten.split.Tensor(view_1932, 1280, -1);  view_1932 = None
        getitem_435: "f32[4, 1024, 1280]" = split_41[1]
        mul_645: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_435, 0.5)
        mul_646: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_435, 0.7071067811865476)
        erf_13: "f32[4, 1024, 1280]" = torch.ops.aten.erf.default(mul_646);  mul_646 = None
        add_641: "f32[4, 1024, 1280]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_647: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_645, add_641);  mul_645 = add_641 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_436: "f32[4, 1024, 1280]" = split_41[0];  split_41 = None
        mul_648: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_436, mul_647);  mul_647 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1934: "f32[4096, 1280]" = torch.ops.aten.view.default(mul_648, [4096, 1280]);  mul_648 = None
        permute_779: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1226, [1, 0]);  primals_1226 = None
        addmm_91: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1227, view_1934, permute_779);  primals_1227 = None
        view_1935: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_91, [4, 1024, 320]);  addmm_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_780: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_1228, [1, 0]);  primals_1228 = None
        mm_424: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1934, permute_780)
        permute_781: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1229, [1, 0]);  primals_1229 = None
        mm_425: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_424, permute_781)
        view_1939: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_425, [4, 1024, 320]);  mm_425 = None
        mul_649: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1939, 1.0);  view_1939 = None
        add_642: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1935, mul_649);  view_1935 = mul_649 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_643: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_642, add_637);  add_642 = add_637 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_1943: "f32[4096, 320]" = torch.ops.aten.view.default(add_643, [4096, 320]);  add_643 = None
        permute_782: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1230, [1, 0]);  primals_1230 = None
        addmm_92: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1231, view_1943, permute_782);  primals_1231 = None
        view_1944: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_92, [4, 1024, 320]);  addmm_92 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_783: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1232, [1, 0]);  primals_1232 = None
        mm_426: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1943, permute_783)
        permute_784: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1233, [1, 0]);  primals_1233 = None
        mm_427: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_426, permute_784)
        view_1948: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_427, [4, 1024, 320]);  mm_427 = None
        mul_650: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1948, 1.0);  view_1948 = None
        add_644: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1944, mul_650);  view_1944 = mul_650 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_1954: "f32[4, 32, 32, 320]" = torch.ops.aten.view.default(add_644, [4, 32, 32, 320]);  add_644 = None
        permute_786: "f32[4, 320, 32, 32]" = torch.ops.aten.permute.default(view_1954, [0, 3, 1, 2]);  view_1954 = None
        clone_105: "f32[4, 320, 32, 32]" = torch.ops.aten.clone.default(permute_786, memory_format = torch.contiguous_format);  permute_786 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_645: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(clone_105, div_57);  clone_105 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_12: "f32[4, 640, 32, 32]" = torch.ops.aten.cat.default([add_645, add_125], 1);  add_645 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1955: "f32[4, 32, 20, 1024]" = torch.ops.aten.view.default(cat_12, [4, 32, 20, 1024])
        var_mean_118 = torch.ops.aten.var_mean.correction(view_1955, [2, 3], correction = 0, keepdim = True)
        getitem_438: "f32[4, 32, 1, 1]" = var_mean_118[0]
        getitem_439: "f32[4, 32, 1, 1]" = var_mean_118[1];  var_mean_118 = None
        add_646: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_438, 1e-05);  getitem_438 = None
        rsqrt_118: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_646);  add_646 = None
        sub_118: "f32[4, 32, 20, 1024]" = torch.ops.aten.sub.Tensor(view_1955, getitem_439);  view_1955 = None
        mul_651: "f32[4, 32, 20, 1024]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = None
        view_1956: "f32[4, 640, 32, 32]" = torch.ops.aten.view.default(mul_651, [4, 640, 32, 32]);  mul_651 = None
        unsqueeze_499: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1235, 0)
        unsqueeze_500: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1234, 0)
        unsqueeze_503: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_652: "f32[4, 640, 32, 32]" = torch.ops.aten.mul.Tensor(view_1956, unsqueeze_504);  view_1956 = unsqueeze_504 = None
        add_647: "f32[4, 640, 32, 32]" = torch.ops.aten.add.Tensor(mul_652, unsqueeze_501);  mul_652 = unsqueeze_501 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_82: "f32[4, 640, 32, 32]" = torch.ops.aten.sigmoid.default(add_647)
        mul_653: "f32[4, 640, 32, 32]" = torch.ops.aten.mul.Tensor(add_647, sigmoid_82);  add_647 = sigmoid_82 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_259: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_653, primals_1236, primals_1237, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1237 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_260: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_653, primals_1238, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_261: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_260, primals_1239, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_654: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_261, 1.0);  convolution_261 = None
        add_648: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_259, mul_654);  convolution_259 = mul_654 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_787: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1240, [1, 0]);  primals_1240 = None
        addmm_93: "f32[4, 320]" = torch.ops.aten.addmm.default(primals_1241, mul_109, permute_787);  primals_1241 = permute_787 = None
        unsqueeze_505: "f32[4, 320, 1]" = torch.ops.aten.unsqueeze.default(addmm_93, 2);  addmm_93 = None
        unsqueeze_506: "f32[4, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_649: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_648, unsqueeze_506);  add_648 = unsqueeze_506 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_1957: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_649, [4, 32, 10, 1024])
        var_mean_119 = torch.ops.aten.var_mean.correction(view_1957, [2, 3], correction = 0, keepdim = True)
        getitem_440: "f32[4, 32, 1, 1]" = var_mean_119[0]
        getitem_441: "f32[4, 32, 1, 1]" = var_mean_119[1];  var_mean_119 = None
        add_650: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_440, 1e-05);  getitem_440 = None
        rsqrt_119: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_650);  add_650 = None
        sub_119: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_1957, getitem_441);  view_1957 = None
        mul_656: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = None
        view_1958: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_656, [4, 320, 32, 32]);  mul_656 = None
        unsqueeze_507: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1243, 0)
        unsqueeze_508: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1242, 0)
        unsqueeze_511: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        mul_657: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_1958, unsqueeze_512);  view_1958 = unsqueeze_512 = None
        add_651: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_509);  mul_657 = unsqueeze_509 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_84: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_651)
        mul_658: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_651, sigmoid_84);  add_651 = sigmoid_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_262: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_658, primals_1244, primals_1245, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1245 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_263: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_658, primals_1246, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_264: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_263, primals_1247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_659: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_264, 1.0);  convolution_264 = None
        add_652: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_262, mul_659);  convolution_262 = mul_659 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_265: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(cat_12, primals_1248, primals_1249, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_1249 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_266: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(cat_12, primals_1250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_267: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_266, primals_1251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_660: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_267, 1.0);  convolution_267 = None
        add_653: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_265, mul_660);  convolution_265 = mul_660 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_654: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_653, add_652);  add_653 = add_652 = None
        div_60: "f32[4, 320, 32, 32]" = torch.ops.aten.div.Tensor(add_654, 1.0);  add_654 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_1959: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(div_60, [4, 32, 10, 1024])
        var_mean_120 = torch.ops.aten.var_mean.correction(view_1959, [2, 3], correction = 0, keepdim = True)
        getitem_442: "f32[4, 32, 1, 1]" = var_mean_120[0]
        getitem_443: "f32[4, 32, 1, 1]" = var_mean_120[1];  var_mean_120 = None
        add_655: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_442, 1e-06);  getitem_442 = None
        rsqrt_120: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_655);  add_655 = None
        sub_120: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_1959, getitem_443);  view_1959 = None
        mul_661: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
        view_1960: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_661, [4, 320, 32, 32]);  mul_661 = None
        unsqueeze_513: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1253, 0);  primals_1253 = None
        unsqueeze_514: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1252, 0)
        unsqueeze_517: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        mul_662: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_1960, unsqueeze_518);  view_1960 = unsqueeze_518 = None
        add_656: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_662, unsqueeze_515);  mul_662 = unsqueeze_515 = None
        squeeze_156: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_443, [2, 3]);  getitem_443 = None
        squeeze_157: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_120, [2, 3]);  rsqrt_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_788: "f32[4, 32, 32, 320]" = torch.ops.aten.permute.default(add_656, [0, 2, 3, 1]);  add_656 = None
        view_1961: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_788, [4, 1024, 320]);  permute_788 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_789: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1254, [1, 0])
        expand_35: "f32[4, 1024, 320]" = torch.ops.aten.expand.default(view_1961, [4, 1024, 320])
        expand_36: "f32[4, 320, 320]" = torch.ops.aten.expand.default(permute_789, [4, 320, 320]);  permute_789 = None
        bmm_17: "f32[4, 1024, 320]" = torch.ops.aten.bmm.default(expand_35, expand_36);  expand_35 = expand_36 = None
        add_657: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(bmm_17, primals_1255);  bmm_17 = primals_1255 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_790: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1256, [1, 0]);  primals_1256 = None
        clone_107: "f32[4, 1024, 320]" = torch.ops.aten.clone.default(view_1961, memory_format = torch.contiguous_format);  view_1961 = None
        view_1965: "f32[4096, 320]" = torch.ops.aten.view.default(clone_107, [4096, 320]);  clone_107 = None
        mm_428: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1965, permute_790)
        permute_791: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1257, [1, 0]);  primals_1257 = None
        mm_429: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_428, permute_791)
        view_1968: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_429, [4, 1024, 320]);  mm_429 = None
        mul_663: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1968, 1.0);  view_1968 = None
        add_658: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_657, mul_663);  add_657 = mul_663 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_121 = torch.ops.aten.var_mean.correction(add_658, [2], correction = 0, keepdim = True)
        getitem_444: "f32[4, 1024, 1]" = var_mean_121[0]
        getitem_445: "f32[4, 1024, 1]" = var_mean_121[1];  var_mean_121 = None
        add_659: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_444, 1e-05);  getitem_444 = None
        rsqrt_121: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_659);  add_659 = None
        sub_121: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_658, getitem_445);  getitem_445 = None
        mul_664: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = None
        mul_665: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_664, primals_1258)
        add_660: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_665, primals_1259);  mul_665 = primals_1259 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_792: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1260, [1, 0]);  primals_1260 = None
        view_1969: "f32[4096, 320]" = torch.ops.aten.view.default(add_660, [4096, 320]);  add_660 = None
        mm_430: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1969, permute_792)
        view_1970: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_430, [4, 1024, 320]);  mm_430 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_793: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1261, [1, 0]);  primals_1261 = None
        mm_431: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1969, permute_793)
        permute_794: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1262, [1, 0]);  primals_1262 = None
        mm_432: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_431, permute_794)
        view_1974: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_432, [4, 1024, 320]);  mm_432 = None
        mul_666: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1974, 1.0);  view_1974 = None
        add_661: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1970, mul_666);  view_1970 = mul_666 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_795: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1263, [1, 0]);  primals_1263 = None
        mm_433: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1969, permute_795)
        view_1978: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_433, [4, 1024, 320]);  mm_433 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_796: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1264, [1, 0]);  primals_1264 = None
        mm_434: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1969, permute_796)
        permute_797: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1265, [1, 0]);  primals_1265 = None
        mm_435: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_434, permute_797)
        view_1982: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_435, [4, 1024, 320]);  mm_435 = None
        mul_667: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1982, 1.0);  view_1982 = None
        add_662: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1978, mul_667);  view_1978 = mul_667 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_798: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1266, [1, 0]);  primals_1266 = None
        mm_436: "f32[4096, 320]" = torch.ops.aten.mm.default(view_1969, permute_798)
        view_1986: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_436, [4, 1024, 320]);  mm_436 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_799: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1267, [1, 0]);  primals_1267 = None
        mm_437: "f32[4096, 80]" = torch.ops.aten.mm.default(view_1969, permute_799)
        permute_800: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1268, [1, 0]);  primals_1268 = None
        mm_438: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_437, permute_800)
        view_1990: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_438, [4, 1024, 320]);  mm_438 = None
        mul_668: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_1990, 1.0);  view_1990 = None
        add_663: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_1986, mul_668);  view_1986 = mul_668 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_1997: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_661, [4, -1, 5, 64]);  add_661 = None
        permute_804: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1997, [0, 2, 1, 3]);  view_1997 = None
        view_1999: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_662, [4, -1, 5, 64]);  add_662 = None
        permute_805: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_1999, [0, 2, 1, 3]);  view_1999 = None
        view_2001: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_663, [4, -1, 5, 64]);  add_663 = None
        permute_806: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2001, [0, 2, 1, 3]);  view_2001 = None
        _scaled_dot_product_efficient_attention_29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_804, permute_805, permute_806, None, True)
        getitem_446: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_29[0]
        getitem_447: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_29[1]
        getitem_448: "i64[]" = _scaled_dot_product_efficient_attention_29[2]
        getitem_449: "i64[]" = _scaled_dot_product_efficient_attention_29[3];  _scaled_dot_product_efficient_attention_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_807: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_446, [0, 2, 1, 3])
        view_2002: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_807, [4, -1, 320]);  permute_807 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2003: "f32[4096, 320]" = torch.ops.aten.view.default(view_2002, [4096, 320]);  view_2002 = None
        permute_808: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1269, [1, 0]);  primals_1269 = None
        addmm_94: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1270, view_2003, permute_808);  primals_1270 = None
        view_2004: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_94, [4, 1024, 320]);  addmm_94 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_809: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1271, [1, 0]);  primals_1271 = None
        mm_439: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2003, permute_809);  view_2003 = None
        permute_810: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1272, [1, 0]);  primals_1272 = None
        mm_440: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_439, permute_810)
        view_2008: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_440, [4, 1024, 320]);  mm_440 = None
        mul_669: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2008, 1.0);  view_2008 = None
        add_664: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2004, mul_669);  view_2004 = mul_669 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_61: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_664, 1.0);  add_664 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_665: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_61, add_658);  div_61 = add_658 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_122 = torch.ops.aten.var_mean.correction(add_665, [2], correction = 0, keepdim = True)
        getitem_450: "f32[4, 1024, 1]" = var_mean_122[0]
        getitem_451: "f32[4, 1024, 1]" = var_mean_122[1];  var_mean_122 = None
        add_666: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_450, 1e-05);  getitem_450 = None
        rsqrt_122: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
        sub_122: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_665, getitem_451);  getitem_451 = None
        mul_670: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = None
        mul_671: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_670, primals_1273)
        add_667: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_671, primals_1274);  mul_671 = primals_1274 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_811: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1275, [1, 0]);  primals_1275 = None
        view_2012: "f32[4096, 320]" = torch.ops.aten.view.default(add_667, [4096, 320]);  add_667 = None
        mm_441: "f32[4096, 320]" = torch.ops.aten.mm.default(view_2012, permute_811)
        view_2013: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_441, [4, 1024, 320]);  mm_441 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_812: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1276, [1, 0]);  primals_1276 = None
        mm_442: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2012, permute_812)
        permute_813: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1277, [1, 0]);  primals_1277 = None
        mm_443: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_442, permute_813)
        view_2017: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_443, [4, 1024, 320]);  mm_443 = None
        mul_672: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2017, 1.0);  view_2017 = None
        add_668: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2013, mul_672);  view_2013 = mul_672 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_814: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1278, [1, 0]);  primals_1278 = None
        mm_444: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_814);  permute_814 = None
        view_2021: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_444, [4, 77, 320]);  mm_444 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_815: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1279, [1, 0]);  primals_1279 = None
        mm_445: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_815);  permute_815 = None
        permute_816: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1280, [1, 0]);  primals_1280 = None
        mm_446: "f32[308, 320]" = torch.ops.aten.mm.default(mm_445, permute_816)
        view_2025: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_446, [4, 77, 320]);  mm_446 = None
        mul_673: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_2025, 1.0);  view_2025 = None
        add_669: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_2021, mul_673);  view_2021 = mul_673 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_817: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1281, [1, 0]);  primals_1281 = None
        mm_447: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_817);  permute_817 = None
        view_2029: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_447, [4, 77, 320]);  mm_447 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_818: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1282, [1, 0]);  primals_1282 = None
        mm_448: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_818);  permute_818 = None
        permute_819: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1283, [1, 0]);  primals_1283 = None
        mm_449: "f32[308, 320]" = torch.ops.aten.mm.default(mm_448, permute_819)
        view_2033: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_449, [4, 77, 320]);  mm_449 = None
        mul_674: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_2033, 1.0);  view_2033 = None
        add_670: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_2029, mul_674);  view_2029 = mul_674 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_2040: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_668, [4, -1, 5, 64]);  add_668 = None
        permute_823: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2040, [0, 2, 1, 3]);  view_2040 = None
        view_2042: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_669, [4, -1, 5, 64]);  add_669 = None
        permute_824: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_2042, [0, 2, 1, 3]);  view_2042 = None
        view_2044: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_670, [4, -1, 5, 64]);  add_670 = None
        permute_825: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_2044, [0, 2, 1, 3]);  view_2044 = None
        _scaled_dot_product_efficient_attention_30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_823, permute_824, permute_825, None, True)
        getitem_452: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_30[0]
        getitem_453: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_30[1]
        getitem_454: "i64[]" = _scaled_dot_product_efficient_attention_30[2]
        getitem_455: "i64[]" = _scaled_dot_product_efficient_attention_30[3];  _scaled_dot_product_efficient_attention_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_826: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_452, [0, 2, 1, 3])
        view_2045: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_826, [4, -1, 320]);  permute_826 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2046: "f32[4096, 320]" = torch.ops.aten.view.default(view_2045, [4096, 320]);  view_2045 = None
        permute_827: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1284, [1, 0]);  primals_1284 = None
        addmm_95: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1285, view_2046, permute_827);  primals_1285 = None
        view_2047: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_95, [4, 1024, 320]);  addmm_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_828: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1286, [1, 0]);  primals_1286 = None
        mm_450: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2046, permute_828);  view_2046 = None
        permute_829: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1287, [1, 0]);  primals_1287 = None
        mm_451: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_450, permute_829)
        view_2051: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_451, [4, 1024, 320]);  mm_451 = None
        mul_675: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2051, 1.0);  view_2051 = None
        add_671: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2047, mul_675);  view_2047 = mul_675 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_62: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_671, 1.0);  add_671 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_672: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_62, add_665);  div_62 = add_665 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_123 = torch.ops.aten.var_mean.correction(add_672, [2], correction = 0, keepdim = True)
        getitem_456: "f32[4, 1024, 1]" = var_mean_123[0]
        getitem_457: "f32[4, 1024, 1]" = var_mean_123[1];  var_mean_123 = None
        add_673: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_456, 1e-05);  getitem_456 = None
        rsqrt_123: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_673);  add_673 = None
        sub_123: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_672, getitem_457);  getitem_457 = None
        mul_676: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = None
        mul_677: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_676, primals_1288)
        add_674: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_677, primals_1289);  mul_677 = primals_1289 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2055: "f32[4096, 320]" = torch.ops.aten.view.default(add_674, [4096, 320]);  add_674 = None
        permute_830: "f32[320, 2560]" = torch.ops.aten.permute.default(primals_1290, [1, 0]);  primals_1290 = None
        addmm_96: "f32[4096, 2560]" = torch.ops.aten.addmm.default(primals_1291, view_2055, permute_830);  primals_1291 = None
        view_2056: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(addmm_96, [4, 1024, 2560]);  addmm_96 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_831: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1292, [1, 0]);  primals_1292 = None
        mm_452: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2055, permute_831)
        permute_832: "f32[80, 2560]" = torch.ops.aten.permute.default(primals_1293, [1, 0]);  primals_1293 = None
        mm_453: "f32[4096, 2560]" = torch.ops.aten.mm.default(mm_452, permute_832)
        view_2060: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(mm_453, [4, 1024, 2560]);  mm_453 = None
        mul_678: "f32[4, 1024, 2560]" = torch.ops.aten.mul.Tensor(view_2060, 1.0);  view_2060 = None
        add_675: "f32[4, 1024, 2560]" = torch.ops.aten.add.Tensor(view_2056, mul_678);  view_2056 = mul_678 = None
        view_2061: "f32[4096, 2560]" = torch.ops.aten.view.default(add_675, [4096, 2560]);  add_675 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_2064: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(view_2061, [4, 1024, 2560]);  view_2061 = None
        split_44 = torch.ops.aten.split.Tensor(view_2064, 1280, -1);  view_2064 = None
        getitem_461: "f32[4, 1024, 1280]" = split_44[1]
        mul_679: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_461, 0.5)
        mul_680: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_461, 0.7071067811865476)
        erf_14: "f32[4, 1024, 1280]" = torch.ops.aten.erf.default(mul_680);  mul_680 = None
        add_676: "f32[4, 1024, 1280]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_681: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_679, add_676);  mul_679 = add_676 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_462: "f32[4, 1024, 1280]" = split_44[0];  split_44 = None
        mul_682: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_462, mul_681);  mul_681 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2066: "f32[4096, 1280]" = torch.ops.aten.view.default(mul_682, [4096, 1280]);  mul_682 = None
        permute_833: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1294, [1, 0]);  primals_1294 = None
        addmm_97: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1295, view_2066, permute_833);  primals_1295 = None
        view_2067: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_97, [4, 1024, 320]);  addmm_97 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_834: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_1296, [1, 0]);  primals_1296 = None
        mm_454: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2066, permute_834)
        permute_835: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1297, [1, 0]);  primals_1297 = None
        mm_455: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_454, permute_835)
        view_2071: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_455, [4, 1024, 320]);  mm_455 = None
        mul_683: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2071, 1.0);  view_2071 = None
        add_677: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2067, mul_683);  view_2067 = mul_683 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_678: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_677, add_672);  add_677 = add_672 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2075: "f32[4096, 320]" = torch.ops.aten.view.default(add_678, [4096, 320]);  add_678 = None
        permute_836: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1298, [1, 0]);  primals_1298 = None
        addmm_98: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1299, view_2075, permute_836);  primals_1299 = None
        view_2076: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_98, [4, 1024, 320]);  addmm_98 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_837: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1300, [1, 0]);  primals_1300 = None
        mm_456: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2075, permute_837)
        permute_838: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1301, [1, 0]);  primals_1301 = None
        mm_457: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_456, permute_838)
        view_2080: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_457, [4, 1024, 320]);  mm_457 = None
        mul_684: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2080, 1.0);  view_2080 = None
        add_679: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2076, mul_684);  view_2076 = mul_684 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_2086: "f32[4, 32, 32, 320]" = torch.ops.aten.view.default(add_679, [4, 32, 32, 320]);  add_679 = None
        permute_840: "f32[4, 320, 32, 32]" = torch.ops.aten.permute.default(view_2086, [0, 3, 1, 2]);  view_2086 = None
        clone_111: "f32[4, 320, 32, 32]" = torch.ops.aten.clone.default(permute_840, memory_format = torch.contiguous_format);  permute_840 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_680: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(clone_111, div_60);  clone_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_blocks.py:2588 in forward, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_13: "f32[4, 640, 32, 32]" = torch.ops.aten.cat.default([add_680, add_91], 1);  add_680 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_2087: "f32[4, 32, 20, 1024]" = torch.ops.aten.view.default(cat_13, [4, 32, 20, 1024])
        var_mean_124 = torch.ops.aten.var_mean.correction(view_2087, [2, 3], correction = 0, keepdim = True)
        getitem_464: "f32[4, 32, 1, 1]" = var_mean_124[0]
        getitem_465: "f32[4, 32, 1, 1]" = var_mean_124[1];  var_mean_124 = None
        add_681: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_464, 1e-05);  getitem_464 = None
        rsqrt_124: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_681);  add_681 = None
        sub_124: "f32[4, 32, 20, 1024]" = torch.ops.aten.sub.Tensor(view_2087, getitem_465);  view_2087 = None
        mul_685: "f32[4, 32, 20, 1024]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = None
        view_2088: "f32[4, 640, 32, 32]" = torch.ops.aten.view.default(mul_685, [4, 640, 32, 32]);  mul_685 = None
        unsqueeze_519: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1303, 0)
        unsqueeze_520: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_1302, 0)
        unsqueeze_523: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        mul_686: "f32[4, 640, 32, 32]" = torch.ops.aten.mul.Tensor(view_2088, unsqueeze_524);  view_2088 = unsqueeze_524 = None
        add_682: "f32[4, 640, 32, 32]" = torch.ops.aten.add.Tensor(mul_686, unsqueeze_521);  mul_686 = unsqueeze_521 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_85: "f32[4, 640, 32, 32]" = torch.ops.aten.sigmoid.default(add_682)
        mul_687: "f32[4, 640, 32, 32]" = torch.ops.aten.mul.Tensor(add_682, sigmoid_85);  add_682 = sigmoid_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_268: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_687, primals_1304, primals_1305, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1305 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_269: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_687, primals_1306, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_270: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_269, primals_1307, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_688: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_270, 1.0);  convolution_270 = None
        add_683: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_268, mul_688);  convolution_268 = mul_688 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:346 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute_841: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1308, [1, 0]);  primals_1308 = None
        addmm_99: "f32[4, 320]" = torch.ops.aten.addmm.default(primals_1309, mul_109, permute_841);  primals_1309 = mul_109 = permute_841 = None
        unsqueeze_525: "f32[4, 320, 1]" = torch.ops.aten.unsqueeze.default(addmm_99, 2);  addmm_99 = None
        unsqueeze_526: "f32[4, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = hidden_states + temb
        add_684: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_683, unsqueeze_526);  add_683 = unsqueeze_526 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_2089: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_684, [4, 32, 10, 1024])
        var_mean_125 = torch.ops.aten.var_mean.correction(view_2089, [2, 3], correction = 0, keepdim = True)
        getitem_466: "f32[4, 32, 1, 1]" = var_mean_125[0]
        getitem_467: "f32[4, 32, 1, 1]" = var_mean_125[1];  var_mean_125 = None
        add_685: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_466, 1e-05);  getitem_466 = None
        rsqrt_125: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_685);  add_685 = None
        sub_125: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_2089, getitem_467);  view_2089 = None
        mul_690: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = None
        view_2090: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_690, [4, 320, 32, 32]);  mul_690 = None
        unsqueeze_527: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1311, 0)
        unsqueeze_528: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
        unsqueeze_529: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
        unsqueeze_530: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1310, 0)
        unsqueeze_531: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
        unsqueeze_532: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
        mul_691: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_2090, unsqueeze_532);  view_2090 = unsqueeze_532 = None
        add_686: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_691, unsqueeze_529);  mul_691 = unsqueeze_529 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_87: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_686)
        mul_692: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_686, sigmoid_87);  add_686 = sigmoid_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_271: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(mul_692, primals_1312, primals_1313, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1313 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_272: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_692, primals_1314, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_273: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_272, primals_1315, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_693: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_273, 1.0);  convolution_273 = None
        add_687: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_271, mul_693);  convolution_271 = mul_693 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_274: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(cat_13, primals_1316, primals_1317, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_1317 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_275: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(cat_13, primals_1318, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_276: "f32[4, 320, 32, 32]" = torch.ops.aten.convolution.default(convolution_275, primals_1319, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_694: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_276, 1.0);  convolution_276 = None
        add_688: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(convolution_274, mul_694);  convolution_274 = mul_694 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_689: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(add_688, add_687);  add_688 = add_687 = None
        div_63: "f32[4, 320, 32, 32]" = torch.ops.aten.div.Tensor(add_689, 1.0);  add_689 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:481 in _operate_on_continuous_inputs, code: hidden_states = self.norm(hidden_states)
        view_2091: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(div_63, [4, 32, 10, 1024])
        var_mean_126 = torch.ops.aten.var_mean.correction(view_2091, [2, 3], correction = 0, keepdim = True)
        getitem_468: "f32[4, 32, 1, 1]" = var_mean_126[0]
        getitem_469: "f32[4, 32, 1, 1]" = var_mean_126[1];  var_mean_126 = None
        add_690: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_468, 1e-06);  getitem_468 = None
        rsqrt_126: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_690);  add_690 = None
        sub_126: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_2091, getitem_469);  view_2091 = None
        mul_695: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = None
        view_2092: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_695, [4, 320, 32, 32]);  mul_695 = None
        unsqueeze_533: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1321, 0);  primals_1321 = None
        unsqueeze_534: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
        unsqueeze_535: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
        unsqueeze_536: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1320, 0)
        unsqueeze_537: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
        unsqueeze_538: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
        mul_696: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_2092, unsqueeze_538);  view_2092 = unsqueeze_538 = None
        add_691: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_696, unsqueeze_535);  mul_696 = unsqueeze_535 = None
        squeeze_162: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_469, [2, 3]);  getitem_469 = None
        squeeze_163: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_126, [2, 3]);  rsqrt_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:489 in _operate_on_continuous_inputs, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_842: "f32[4, 32, 32, 320]" = torch.ops.aten.permute.default(add_691, [0, 2, 3, 1]);  add_691 = None
        view_2093: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_842, [4, 1024, 320]);  permute_842 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_843: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1322, [1, 0])
        expand_37: "f32[4, 1024, 320]" = torch.ops.aten.expand.default(view_2093, [4, 1024, 320])
        expand_38: "f32[4, 320, 320]" = torch.ops.aten.expand.default(permute_843, [4, 320, 320]);  permute_843 = None
        bmm_18: "f32[4, 1024, 320]" = torch.ops.aten.bmm.default(expand_37, expand_38);  expand_37 = expand_38 = None
        add_692: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(bmm_18, primals_1323);  bmm_18 = primals_1323 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_844: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1324, [1, 0]);  primals_1324 = None
        clone_113: "f32[4, 1024, 320]" = torch.ops.aten.clone.default(view_2093, memory_format = torch.contiguous_format);  view_2093 = None
        view_2097: "f32[4096, 320]" = torch.ops.aten.view.default(clone_113, [4096, 320]);  clone_113 = None
        mm_458: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2097, permute_844)
        permute_845: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1325, [1, 0]);  primals_1325 = None
        mm_459: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_458, permute_845)
        view_2100: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_459, [4, 1024, 320]);  mm_459 = None
        mul_697: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2100, 1.0);  view_2100 = None
        add_693: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_692, mul_697);  add_692 = mul_697 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        var_mean_127 = torch.ops.aten.var_mean.correction(add_693, [2], correction = 0, keepdim = True)
        getitem_470: "f32[4, 1024, 1]" = var_mean_127[0]
        getitem_471: "f32[4, 1024, 1]" = var_mean_127[1];  var_mean_127 = None
        add_694: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_470, 1e-05);  getitem_470 = None
        rsqrt_127: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_694);  add_694 = None
        sub_127: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_693, getitem_471);  getitem_471 = None
        mul_698: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = None
        mul_699: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_698, primals_1326)
        add_695: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_699, primals_1327);  mul_699 = primals_1327 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_846: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1328, [1, 0]);  primals_1328 = None
        view_2101: "f32[4096, 320]" = torch.ops.aten.view.default(add_695, [4096, 320]);  add_695 = None
        mm_460: "f32[4096, 320]" = torch.ops.aten.mm.default(view_2101, permute_846)
        view_2102: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_460, [4, 1024, 320]);  mm_460 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_847: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1329, [1, 0]);  primals_1329 = None
        mm_461: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2101, permute_847)
        permute_848: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1330, [1, 0]);  primals_1330 = None
        mm_462: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_461, permute_848)
        view_2106: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_462, [4, 1024, 320]);  mm_462 = None
        mul_700: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2106, 1.0);  view_2106 = None
        add_696: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2102, mul_700);  view_2102 = mul_700 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_849: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1331, [1, 0]);  primals_1331 = None
        mm_463: "f32[4096, 320]" = torch.ops.aten.mm.default(view_2101, permute_849)
        view_2110: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_463, [4, 1024, 320]);  mm_463 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_850: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1332, [1, 0]);  primals_1332 = None
        mm_464: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2101, permute_850)
        permute_851: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1333, [1, 0]);  primals_1333 = None
        mm_465: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_464, permute_851)
        view_2114: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_465, [4, 1024, 320]);  mm_465 = None
        mul_701: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2114, 1.0);  view_2114 = None
        add_697: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2110, mul_701);  view_2110 = mul_701 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_852: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1334, [1, 0]);  primals_1334 = None
        mm_466: "f32[4096, 320]" = torch.ops.aten.mm.default(view_2101, permute_852)
        view_2118: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_466, [4, 1024, 320]);  mm_466 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_853: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1335, [1, 0]);  primals_1335 = None
        mm_467: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2101, permute_853)
        permute_854: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1336, [1, 0]);  primals_1336 = None
        mm_468: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_467, permute_854)
        view_2122: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_468, [4, 1024, 320]);  mm_468 = None
        mul_702: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2122, 1.0);  view_2122 = None
        add_698: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2118, mul_702);  view_2118 = mul_702 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_2129: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_696, [4, -1, 5, 64]);  add_696 = None
        permute_858: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2129, [0, 2, 1, 3]);  view_2129 = None
        view_2131: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_697, [4, -1, 5, 64]);  add_697 = None
        permute_859: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2131, [0, 2, 1, 3]);  view_2131 = None
        view_2133: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_698, [4, -1, 5, 64]);  add_698 = None
        permute_860: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2133, [0, 2, 1, 3]);  view_2133 = None
        _scaled_dot_product_efficient_attention_31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_858, permute_859, permute_860, None, True)
        getitem_472: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_31[0]
        getitem_473: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_31[1]
        getitem_474: "i64[]" = _scaled_dot_product_efficient_attention_31[2]
        getitem_475: "i64[]" = _scaled_dot_product_efficient_attention_31[3];  _scaled_dot_product_efficient_attention_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_861: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_472, [0, 2, 1, 3])
        view_2134: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_861, [4, -1, 320]);  permute_861 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2135: "f32[4096, 320]" = torch.ops.aten.view.default(view_2134, [4096, 320]);  view_2134 = None
        permute_862: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1337, [1, 0]);  primals_1337 = None
        addmm_100: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1338, view_2135, permute_862);  primals_1338 = None
        view_2136: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_100, [4, 1024, 320]);  addmm_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_863: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1339, [1, 0]);  primals_1339 = None
        mm_469: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2135, permute_863);  view_2135 = None
        permute_864: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1340, [1, 0]);  primals_1340 = None
        mm_470: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_469, permute_864)
        view_2140: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_470, [4, 1024, 320]);  mm_470 = None
        mul_703: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2140, 1.0);  view_2140 = None
        add_699: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2136, mul_703);  view_2136 = mul_703 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_64: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_699, 1.0);  add_699 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:526 in forward, code: hidden_states = attn_output + hidden_states
        add_700: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_64, add_693);  div_64 = add_693 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        var_mean_128 = torch.ops.aten.var_mean.correction(add_700, [2], correction = 0, keepdim = True)
        getitem_476: "f32[4, 1024, 1]" = var_mean_128[0]
        getitem_477: "f32[4, 1024, 1]" = var_mean_128[1];  var_mean_128 = None
        add_701: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_476, 1e-05);  getitem_476 = None
        rsqrt_128: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_701);  add_701 = None
        sub_128: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_700, getitem_477);  getitem_477 = None
        mul_704: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = None
        mul_705: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_704, primals_1341)
        add_702: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_705, primals_1342);  mul_705 = primals_1342 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_865: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1343, [1, 0]);  primals_1343 = None
        view_2144: "f32[4096, 320]" = torch.ops.aten.view.default(add_702, [4096, 320]);  add_702 = None
        mm_471: "f32[4096, 320]" = torch.ops.aten.mm.default(view_2144, permute_865)
        view_2145: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_471, [4, 1024, 320]);  mm_471 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_866: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1344, [1, 0]);  primals_1344 = None
        mm_472: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2144, permute_866)
        permute_867: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1345, [1, 0]);  primals_1345 = None
        mm_473: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_472, permute_867)
        view_2149: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_473, [4, 1024, 320]);  mm_473 = None
        mul_706: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2149, 1.0);  view_2149 = None
        add_703: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2145, mul_706);  view_2145 = mul_706 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_868: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1346, [1, 0]);  primals_1346 = None
        mm_474: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_868);  permute_868 = None
        view_2153: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_474, [4, 77, 320]);  mm_474 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_869: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1347, [1, 0]);  primals_1347 = None
        mm_475: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_869);  permute_869 = None
        permute_870: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1348, [1, 0]);  primals_1348 = None
        mm_476: "f32[308, 320]" = torch.ops.aten.mm.default(mm_475, permute_870)
        view_2157: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_476, [4, 77, 320]);  mm_476 = None
        mul_707: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_2157, 1.0);  view_2157 = None
        add_704: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_2153, mul_707);  view_2153 = mul_707 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_871: "f32[1024, 320]" = torch.ops.aten.permute.default(primals_1349, [1, 0]);  primals_1349 = None
        mm_477: "f32[308, 320]" = torch.ops.aten.mm.default(view_148, permute_871);  permute_871 = None
        view_2161: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_477, [4, 77, 320]);  mm_477 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_872: "f32[1024, 80]" = torch.ops.aten.permute.default(primals_1350, [1, 0]);  primals_1350 = None
        mm_478: "f32[308, 80]" = torch.ops.aten.mm.default(view_148, permute_872);  permute_872 = None
        permute_873: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1351, [1, 0]);  primals_1351 = None
        mm_479: "f32[308, 320]" = torch.ops.aten.mm.default(mm_478, permute_873)
        view_2165: "f32[4, 77, 320]" = torch.ops.aten.view.default(mm_479, [4, 77, 320]);  mm_479 = None
        mul_708: "f32[4, 77, 320]" = torch.ops.aten.mul.Tensor(view_2165, 1.0);  view_2165 = None
        add_705: "f32[4, 77, 320]" = torch.ops.aten.add.Tensor(view_2161, mul_708);  view_2161 = mul_708 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_2172: "f32[4, 1024, 5, 64]" = torch.ops.aten.view.default(add_703, [4, -1, 5, 64]);  add_703 = None
        permute_877: "f32[4, 5, 1024, 64]" = torch.ops.aten.permute.default(view_2172, [0, 2, 1, 3]);  view_2172 = None
        view_2174: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_704, [4, -1, 5, 64]);  add_704 = None
        permute_878: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_2174, [0, 2, 1, 3]);  view_2174 = None
        view_2176: "f32[4, 77, 5, 64]" = torch.ops.aten.view.default(add_705, [4, -1, 5, 64]);  add_705 = None
        permute_879: "f32[4, 5, 77, 64]" = torch.ops.aten.permute.default(view_2176, [0, 2, 1, 3]);  view_2176 = None
        _scaled_dot_product_efficient_attention_32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_877, permute_878, permute_879, None, True)
        getitem_478: "f32[4, 5, 1024, 64]" = _scaled_dot_product_efficient_attention_32[0]
        getitem_479: "f32[4, 5, 1024]" = _scaled_dot_product_efficient_attention_32[1]
        getitem_480: "i64[]" = _scaled_dot_product_efficient_attention_32[2]
        getitem_481: "i64[]" = _scaled_dot_product_efficient_attention_32[3];  _scaled_dot_product_efficient_attention_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_880: "f32[4, 1024, 5, 64]" = torch.ops.aten.permute.default(getitem_478, [0, 2, 1, 3])
        view_2177: "f32[4, 1024, 320]" = torch.ops.aten.view.default(permute_880, [4, -1, 320]);  permute_880 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2178: "f32[4096, 320]" = torch.ops.aten.view.default(view_2177, [4096, 320]);  view_2177 = None
        permute_881: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1352, [1, 0]);  primals_1352 = None
        addmm_101: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1353, view_2178, permute_881);  primals_1353 = None
        view_2179: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_101, [4, 1024, 320]);  addmm_101 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_882: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1354, [1, 0]);  primals_1354 = None
        mm_480: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2178, permute_882);  view_2178 = None
        permute_883: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1355, [1, 0]);  primals_1355 = None
        mm_481: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_480, permute_883)
        view_2183: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_481, [4, 1024, 320]);  mm_481 = None
        mul_709: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2183, 1.0);  view_2183 = None
        add_706: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2179, mul_709);  view_2179 = mul_709 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_65: "f32[4, 1024, 320]" = torch.ops.aten.div.Tensor(add_706, 1.0);  add_706 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:558 in forward, code: hidden_states = attn_output + hidden_states
        add_707: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(div_65, add_700);  div_65 = add_700 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        var_mean_129 = torch.ops.aten.var_mean.correction(add_707, [2], correction = 0, keepdim = True)
        getitem_482: "f32[4, 1024, 1]" = var_mean_129[0]
        getitem_483: "f32[4, 1024, 1]" = var_mean_129[1];  var_mean_129 = None
        add_708: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_482, 1e-05);  getitem_482 = None
        rsqrt_129: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_708);  add_708 = None
        sub_129: "f32[4, 1024, 320]" = torch.ops.aten.sub.Tensor(add_707, getitem_483);  getitem_483 = None
        mul_710: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = None
        mul_711: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(mul_710, primals_1356)
        add_709: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(mul_711, primals_1357);  mul_711 = primals_1357 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2187: "f32[4096, 320]" = torch.ops.aten.view.default(add_709, [4096, 320]);  add_709 = None
        permute_884: "f32[320, 2560]" = torch.ops.aten.permute.default(primals_1358, [1, 0]);  primals_1358 = None
        addmm_102: "f32[4096, 2560]" = torch.ops.aten.addmm.default(primals_1359, view_2187, permute_884);  primals_1359 = None
        view_2188: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(addmm_102, [4, 1024, 2560]);  addmm_102 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_885: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1360, [1, 0]);  primals_1360 = None
        mm_482: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2187, permute_885)
        permute_886: "f32[80, 2560]" = torch.ops.aten.permute.default(primals_1361, [1, 0]);  primals_1361 = None
        mm_483: "f32[4096, 2560]" = torch.ops.aten.mm.default(mm_482, permute_886)
        view_2192: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(mm_483, [4, 1024, 2560]);  mm_483 = None
        mul_712: "f32[4, 1024, 2560]" = torch.ops.aten.mul.Tensor(view_2192, 1.0);  view_2192 = None
        add_710: "f32[4, 1024, 2560]" = torch.ops.aten.add.Tensor(view_2188, mul_712);  view_2188 = mul_712 = None
        view_2193: "f32[4096, 2560]" = torch.ops.aten.view.default(add_710, [4096, 2560]);  add_710 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:111 in gelu, code: return F.gelu(gate)
        view_2196: "f32[4, 1024, 2560]" = torch.ops.aten.view.default(view_2193, [4, 1024, 2560]);  view_2193 = None
        split_47 = torch.ops.aten.split.Tensor(view_2196, 1280, -1);  view_2196 = None
        getitem_487: "f32[4, 1024, 1280]" = split_47[1]
        mul_713: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_487, 0.5)
        mul_714: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_487, 0.7071067811865476)
        erf_15: "f32[4, 1024, 1280]" = torch.ops.aten.erf.default(mul_714);  mul_714 = None
        add_711: "f32[4, 1024, 1280]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_715: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(mul_713, add_711);  mul_713 = add_711 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/activations.py:123 in forward, code: return hidden_states * self.gelu(gate)
        getitem_488: "f32[4, 1024, 1280]" = split_47[0];  split_47 = None
        mul_716: "f32[4, 1024, 1280]" = torch.ops.aten.mul.Tensor(getitem_488, mul_715);  mul_715 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2198: "f32[4096, 1280]" = torch.ops.aten.view.default(mul_716, [4096, 1280]);  mul_716 = None
        permute_887: "f32[1280, 320]" = torch.ops.aten.permute.default(primals_1362, [1, 0]);  primals_1362 = None
        addmm_103: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1363, view_2198, permute_887);  primals_1363 = None
        view_2199: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_103, [4, 1024, 320]);  addmm_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_888: "f32[1280, 80]" = torch.ops.aten.permute.default(primals_1364, [1, 0]);  primals_1364 = None
        mm_484: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2198, permute_888)
        permute_889: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1365, [1, 0]);  primals_1365 = None
        mm_485: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_484, permute_889)
        view_2203: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_485, [4, 1024, 320]);  mm_485 = None
        mul_717: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2203, 1.0);  view_2203 = None
        add_712: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2199, mul_717);  view_2199 = mul_717 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:585 in forward, code: hidden_states = ff_output + hidden_states
        add_713: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(add_712, add_707);  add_712 = add_707 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_2207: "f32[4096, 320]" = torch.ops.aten.view.default(add_713, [4096, 320]);  add_713 = None
        permute_890: "f32[320, 320]" = torch.ops.aten.permute.default(primals_1366, [1, 0]);  primals_1366 = None
        addmm_104: "f32[4096, 320]" = torch.ops.aten.addmm.default(primals_1367, view_2207, permute_890);  primals_1367 = None
        view_2208: "f32[4, 1024, 320]" = torch.ops.aten.view.default(addmm_104, [4, 1024, 320]);  addmm_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_891: "f32[320, 80]" = torch.ops.aten.permute.default(primals_1368, [1, 0]);  primals_1368 = None
        mm_486: "f32[4096, 80]" = torch.ops.aten.mm.default(view_2207, permute_891)
        permute_892: "f32[80, 320]" = torch.ops.aten.permute.default(primals_1369, [1, 0]);  primals_1369 = None
        mm_487: "f32[4096, 320]" = torch.ops.aten.mm.default(mm_486, permute_892)
        view_2212: "f32[4, 1024, 320]" = torch.ops.aten.view.default(mm_487, [4, 1024, 320]);  mm_487 = None
        mul_718: "f32[4, 1024, 320]" = torch.ops.aten.mul.Tensor(view_2212, 1.0);  view_2212 = None
        add_714: "f32[4, 1024, 320]" = torch.ops.aten.add.Tensor(view_2208, mul_718);  view_2208 = mul_718 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:523 in _get_output_for_continuous_inputs, code: hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        view_2218: "f32[4, 32, 32, 320]" = torch.ops.aten.view.default(add_714, [4, 32, 32, 320]);  add_714 = None
        permute_894: "f32[4, 320, 32, 32]" = torch.ops.aten.permute.default(view_2218, [0, 3, 1, 2]);  view_2218 = None
        clone_117: "f32[4, 320, 32, 32]" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:526 in _get_output_for_continuous_inputs, code: output = hidden_states + residual
        add_715: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(clone_117, div_63);  clone_117 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py:1301 in forward, code: sample = self.conv_norm_out(sample)
        view_2219: "f32[4, 32, 10, 1024]" = torch.ops.aten.view.default(add_715, [4, 32, 10, 1024])
        var_mean_130 = torch.ops.aten.var_mean.correction(view_2219, [2, 3], correction = 0, keepdim = True)
        getitem_490: "f32[4, 32, 1, 1]" = var_mean_130[0]
        getitem_491: "f32[4, 32, 1, 1]" = var_mean_130[1];  var_mean_130 = None
        add_716: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_490, 1e-05);  getitem_490 = None
        rsqrt_130: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_716);  add_716 = None
        sub_130: "f32[4, 32, 10, 1024]" = torch.ops.aten.sub.Tensor(view_2219, getitem_491);  view_2219 = None
        mul_719: "f32[4, 32, 10, 1024]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = None
        view_2220: "f32[4, 320, 32, 32]" = torch.ops.aten.view.default(mul_719, [4, 320, 32, 32]);  mul_719 = None
        unsqueeze_539: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1371, 0)
        unsqueeze_540: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
        unsqueeze_541: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
        unsqueeze_542: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_1370, 0)
        unsqueeze_543: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
        unsqueeze_544: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
        mul_720: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(view_2220, unsqueeze_544);  view_2220 = unsqueeze_544 = None
        add_717: "f32[4, 320, 32, 32]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_541);  mul_720 = unsqueeze_541 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py:1302 in forward, code: sample = self.conv_act(sample)
        sigmoid_88: "f32[4, 320, 32, 32]" = torch.ops.aten.sigmoid.default(add_717)
        mul_721: "f32[4, 320, 32, 32]" = torch.ops.aten.mul.Tensor(add_717, sigmoid_88);  add_717 = sigmoid_88 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_277: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_721, primals_1372, primals_1373, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_1373 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_278: "f32[4, 80, 32, 32]" = torch.ops.aten.convolution.default(mul_721, primals_1374, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_279: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(convolution_278, primals_1375, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_722: "f32[4, 4, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_279, 1.0);  convolution_279 = None
        add_718: "f32[4, 4, 32, 32]" = torch.ops.aten.add.Tensor(convolution_277, mul_722);  convolution_277 = mul_722 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_899: "f32[320, 80]" = torch.ops.aten.permute.default(permute_892, [1, 0]);  permute_892 = None
        permute_903: "f32[80, 320]" = torch.ops.aten.permute.default(permute_891, [1, 0]);  permute_891 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_905: "f32[320, 320]" = torch.ops.aten.permute.default(permute_890, [1, 0]);  permute_890 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_908: "f32[320, 80]" = torch.ops.aten.permute.default(permute_889, [1, 0]);  permute_889 = None
        permute_912: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_914: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_887, [1, 0]);  permute_887 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_917: "f32[2560, 80]" = torch.ops.aten.permute.default(permute_886, [1, 0]);  permute_886 = None
        permute_921: "f32[80, 320]" = torch.ops.aten.permute.default(permute_885, [1, 0]);  permute_885 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_923: "f32[2560, 320]" = torch.ops.aten.permute.default(permute_884, [1, 0]);  permute_884 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_66: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_129, 320);  rsqrt_129 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_926: "f32[320, 80]" = torch.ops.aten.permute.default(permute_883, [1, 0]);  permute_883 = None
        permute_930: "f32[80, 320]" = torch.ops.aten.permute.default(permute_882, [1, 0]);  permute_882 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_932: "f32[320, 320]" = torch.ops.aten.permute.default(permute_881, [1, 0]);  permute_881 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_939: "f32[320, 80]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_946: "f32[320, 80]" = torch.ops.aten.permute.default(permute_870, [1, 0]);  permute_870 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_953: "f32[320, 80]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
        permute_957: "f32[80, 320]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_959: "f32[320, 320]" = torch.ops.aten.permute.default(permute_865, [1, 0]);  permute_865 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_68: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_128, 320);  rsqrt_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_962: "f32[320, 80]" = torch.ops.aten.permute.default(permute_864, [1, 0]);  permute_864 = None
        permute_966: "f32[80, 320]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_968: "f32[320, 320]" = torch.ops.aten.permute.default(permute_862, [1, 0]);  permute_862 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_975: "f32[320, 80]" = torch.ops.aten.permute.default(permute_854, [1, 0]);  permute_854 = None
        permute_979: "f32[80, 320]" = torch.ops.aten.permute.default(permute_853, [1, 0]);  permute_853 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_981: "f32[320, 320]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_984: "f32[320, 80]" = torch.ops.aten.permute.default(permute_851, [1, 0]);  permute_851 = None
        permute_988: "f32[80, 320]" = torch.ops.aten.permute.default(permute_850, [1, 0]);  permute_850 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_990: "f32[320, 320]" = torch.ops.aten.permute.default(permute_849, [1, 0]);  permute_849 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_993: "f32[320, 80]" = torch.ops.aten.permute.default(permute_848, [1, 0]);  permute_848 = None
        permute_997: "f32[80, 320]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_999: "f32[320, 320]" = torch.ops.aten.permute.default(permute_846, [1, 0]);  permute_846 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_70: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_127, 320);  rsqrt_127 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1002: "f32[320, 80]" = torch.ops.aten.permute.default(permute_845, [1, 0]);  permute_845 = None
        permute_1006: "f32[80, 320]" = torch.ops.aten.permute.default(permute_844, [1, 0]);  permute_844 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1015: "f32[320, 80]" = torch.ops.aten.permute.default(permute_838, [1, 0]);  permute_838 = None
        permute_1019: "f32[80, 320]" = torch.ops.aten.permute.default(permute_837, [1, 0]);  permute_837 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1021: "f32[320, 320]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1024: "f32[320, 80]" = torch.ops.aten.permute.default(permute_835, [1, 0]);  permute_835 = None
        permute_1028: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1030: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_833, [1, 0]);  permute_833 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1033: "f32[2560, 80]" = torch.ops.aten.permute.default(permute_832, [1, 0]);  permute_832 = None
        permute_1037: "f32[80, 320]" = torch.ops.aten.permute.default(permute_831, [1, 0]);  permute_831 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1039: "f32[2560, 320]" = torch.ops.aten.permute.default(permute_830, [1, 0]);  permute_830 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_72: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_123, 320);  rsqrt_123 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1042: "f32[320, 80]" = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
        permute_1046: "f32[80, 320]" = torch.ops.aten.permute.default(permute_828, [1, 0]);  permute_828 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1048: "f32[320, 320]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1055: "f32[320, 80]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1062: "f32[320, 80]" = torch.ops.aten.permute.default(permute_816, [1, 0]);  permute_816 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1069: "f32[320, 80]" = torch.ops.aten.permute.default(permute_813, [1, 0]);  permute_813 = None
        permute_1073: "f32[80, 320]" = torch.ops.aten.permute.default(permute_812, [1, 0]);  permute_812 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1075: "f32[320, 320]" = torch.ops.aten.permute.default(permute_811, [1, 0]);  permute_811 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_74: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_122, 320);  rsqrt_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1078: "f32[320, 80]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
        permute_1082: "f32[80, 320]" = torch.ops.aten.permute.default(permute_809, [1, 0]);  permute_809 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1084: "f32[320, 320]" = torch.ops.aten.permute.default(permute_808, [1, 0]);  permute_808 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1091: "f32[320, 80]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
        permute_1095: "f32[80, 320]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1097: "f32[320, 320]" = torch.ops.aten.permute.default(permute_798, [1, 0]);  permute_798 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1100: "f32[320, 80]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
        permute_1104: "f32[80, 320]" = torch.ops.aten.permute.default(permute_796, [1, 0]);  permute_796 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1106: "f32[320, 320]" = torch.ops.aten.permute.default(permute_795, [1, 0]);  permute_795 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1109: "f32[320, 80]" = torch.ops.aten.permute.default(permute_794, [1, 0]);  permute_794 = None
        permute_1113: "f32[80, 320]" = torch.ops.aten.permute.default(permute_793, [1, 0]);  permute_793 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1115: "f32[320, 320]" = torch.ops.aten.permute.default(permute_792, [1, 0]);  permute_792 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_76: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_121, 320);  rsqrt_121 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1118: "f32[320, 80]" = torch.ops.aten.permute.default(permute_791, [1, 0]);  permute_791 = None
        permute_1122: "f32[80, 320]" = torch.ops.aten.permute.default(permute_790, [1, 0]);  permute_790 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1131: "f32[320, 80]" = torch.ops.aten.permute.default(permute_784, [1, 0]);  permute_784 = None
        permute_1135: "f32[80, 320]" = torch.ops.aten.permute.default(permute_783, [1, 0]);  permute_783 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1137: "f32[320, 320]" = torch.ops.aten.permute.default(permute_782, [1, 0]);  permute_782 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1140: "f32[320, 80]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
        permute_1144: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_780, [1, 0]);  permute_780 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1146: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1149: "f32[2560, 80]" = torch.ops.aten.permute.default(permute_778, [1, 0]);  permute_778 = None
        permute_1153: "f32[80, 320]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1155: "f32[2560, 320]" = torch.ops.aten.permute.default(permute_776, [1, 0]);  permute_776 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_78: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_117, 320);  rsqrt_117 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1158: "f32[320, 80]" = torch.ops.aten.permute.default(permute_775, [1, 0]);  permute_775 = None
        permute_1162: "f32[80, 320]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1164: "f32[320, 320]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1171: "f32[320, 80]" = torch.ops.aten.permute.default(permute_765, [1, 0]);  permute_765 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1178: "f32[320, 80]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1185: "f32[320, 80]" = torch.ops.aten.permute.default(permute_759, [1, 0]);  permute_759 = None
        permute_1189: "f32[80, 320]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1191: "f32[320, 320]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_80: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_116, 320);  rsqrt_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1194: "f32[320, 80]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
        permute_1198: "f32[80, 320]" = torch.ops.aten.permute.default(permute_755, [1, 0]);  permute_755 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1200: "f32[320, 320]" = torch.ops.aten.permute.default(permute_754, [1, 0]);  permute_754 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1207: "f32[320, 80]" = torch.ops.aten.permute.default(permute_746, [1, 0]);  permute_746 = None
        permute_1211: "f32[80, 320]" = torch.ops.aten.permute.default(permute_745, [1, 0]);  permute_745 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1213: "f32[320, 320]" = torch.ops.aten.permute.default(permute_744, [1, 0]);  permute_744 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1216: "f32[320, 80]" = torch.ops.aten.permute.default(permute_743, [1, 0]);  permute_743 = None
        permute_1220: "f32[80, 320]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1222: "f32[320, 320]" = torch.ops.aten.permute.default(permute_741, [1, 0]);  permute_741 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1225: "f32[320, 80]" = torch.ops.aten.permute.default(permute_740, [1, 0]);  permute_740 = None
        permute_1229: "f32[80, 320]" = torch.ops.aten.permute.default(permute_739, [1, 0]);  permute_739 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1231: "f32[320, 320]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_82: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_115, 320);  rsqrt_115 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1234: "f32[320, 80]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
        permute_1238: "f32[80, 320]" = torch.ops.aten.permute.default(permute_736, [1, 0]);  permute_736 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1247: "f32[640, 80]" = torch.ops.aten.permute.default(permute_730, [1, 0]);  permute_730 = None
        permute_1251: "f32[80, 640]" = torch.ops.aten.permute.default(permute_729, [1, 0]);  permute_729 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1253: "f32[640, 640]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1256: "f32[640, 80]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
        permute_1260: "f32[80, 2560]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1262: "f32[640, 2560]" = torch.ops.aten.permute.default(permute_725, [1, 0]);  permute_725 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1265: "f32[5120, 80]" = torch.ops.aten.permute.default(permute_724, [1, 0]);  permute_724 = None
        permute_1269: "f32[80, 640]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1271: "f32[5120, 640]" = torch.ops.aten.permute.default(permute_722, [1, 0]);  permute_722 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_84: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_111, 640);  rsqrt_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1274: "f32[640, 80]" = torch.ops.aten.permute.default(permute_721, [1, 0]);  permute_721 = None
        permute_1278: "f32[80, 640]" = torch.ops.aten.permute.default(permute_720, [1, 0]);  permute_720 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1280: "f32[640, 640]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1287: "f32[640, 80]" = torch.ops.aten.permute.default(permute_711, [1, 0]);  permute_711 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1294: "f32[640, 80]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1301: "f32[640, 80]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
        permute_1305: "f32[80, 640]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1307: "f32[640, 640]" = torch.ops.aten.permute.default(permute_703, [1, 0]);  permute_703 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_86: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_110, 640);  rsqrt_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1310: "f32[640, 80]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
        permute_1314: "f32[80, 640]" = torch.ops.aten.permute.default(permute_701, [1, 0]);  permute_701 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1316: "f32[640, 640]" = torch.ops.aten.permute.default(permute_700, [1, 0]);  permute_700 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1323: "f32[640, 80]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
        permute_1327: "f32[80, 640]" = torch.ops.aten.permute.default(permute_691, [1, 0]);  permute_691 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1329: "f32[640, 640]" = torch.ops.aten.permute.default(permute_690, [1, 0]);  permute_690 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1332: "f32[640, 80]" = torch.ops.aten.permute.default(permute_689, [1, 0]);  permute_689 = None
        permute_1336: "f32[80, 640]" = torch.ops.aten.permute.default(permute_688, [1, 0]);  permute_688 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1338: "f32[640, 640]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1341: "f32[640, 80]" = torch.ops.aten.permute.default(permute_686, [1, 0]);  permute_686 = None
        permute_1345: "f32[80, 640]" = torch.ops.aten.permute.default(permute_685, [1, 0]);  permute_685 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1347: "f32[640, 640]" = torch.ops.aten.permute.default(permute_684, [1, 0]);  permute_684 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_88: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_109, 640);  rsqrt_109 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1350: "f32[640, 80]" = torch.ops.aten.permute.default(permute_683, [1, 0]);  permute_683 = None
        permute_1354: "f32[80, 640]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1363: "f32[640, 80]" = torch.ops.aten.permute.default(permute_676, [1, 0]);  permute_676 = None
        permute_1367: "f32[80, 640]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1369: "f32[640, 640]" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1372: "f32[640, 80]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
        permute_1376: "f32[80, 2560]" = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1378: "f32[640, 2560]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1381: "f32[5120, 80]" = torch.ops.aten.permute.default(permute_670, [1, 0]);  permute_670 = None
        permute_1385: "f32[80, 640]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1387: "f32[5120, 640]" = torch.ops.aten.permute.default(permute_668, [1, 0]);  permute_668 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_90: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_105, 640);  rsqrt_105 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1390: "f32[640, 80]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
        permute_1394: "f32[80, 640]" = torch.ops.aten.permute.default(permute_666, [1, 0]);  permute_666 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1396: "f32[640, 640]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1403: "f32[640, 80]" = torch.ops.aten.permute.default(permute_657, [1, 0]);  permute_657 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1410: "f32[640, 80]" = torch.ops.aten.permute.default(permute_654, [1, 0]);  permute_654 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1417: "f32[640, 80]" = torch.ops.aten.permute.default(permute_651, [1, 0]);  permute_651 = None
        permute_1421: "f32[80, 640]" = torch.ops.aten.permute.default(permute_650, [1, 0]);  permute_650 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1423: "f32[640, 640]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_92: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_104, 640);  rsqrt_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1426: "f32[640, 80]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
        permute_1430: "f32[80, 640]" = torch.ops.aten.permute.default(permute_647, [1, 0]);  permute_647 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1432: "f32[640, 640]" = torch.ops.aten.permute.default(permute_646, [1, 0]);  permute_646 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1439: "f32[640, 80]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
        permute_1443: "f32[80, 640]" = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1445: "f32[640, 640]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1448: "f32[640, 80]" = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
        permute_1452: "f32[80, 640]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1454: "f32[640, 640]" = torch.ops.aten.permute.default(permute_633, [1, 0]);  permute_633 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1457: "f32[640, 80]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
        permute_1461: "f32[80, 640]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1463: "f32[640, 640]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_94: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_103, 640);  rsqrt_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1466: "f32[640, 80]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
        permute_1470: "f32[80, 640]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1479: "f32[640, 80]" = torch.ops.aten.permute.default(permute_622, [1, 0]);  permute_622 = None
        permute_1483: "f32[80, 640]" = torch.ops.aten.permute.default(permute_621, [1, 0]);  permute_621 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1485: "f32[640, 640]" = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1488: "f32[640, 80]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
        permute_1492: "f32[80, 2560]" = torch.ops.aten.permute.default(permute_618, [1, 0]);  permute_618 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1494: "f32[640, 2560]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1497: "f32[5120, 80]" = torch.ops.aten.permute.default(permute_616, [1, 0]);  permute_616 = None
        permute_1501: "f32[80, 640]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1503: "f32[5120, 640]" = torch.ops.aten.permute.default(permute_614, [1, 0]);  permute_614 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_96: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_99, 640);  rsqrt_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1506: "f32[640, 80]" = torch.ops.aten.permute.default(permute_613, [1, 0]);  permute_613 = None
        permute_1510: "f32[80, 640]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1512: "f32[640, 640]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1519: "f32[640, 80]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1526: "f32[640, 80]" = torch.ops.aten.permute.default(permute_600, [1, 0]);  permute_600 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1533: "f32[640, 80]" = torch.ops.aten.permute.default(permute_597, [1, 0]);  permute_597 = None
        permute_1537: "f32[80, 640]" = torch.ops.aten.permute.default(permute_596, [1, 0]);  permute_596 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1539: "f32[640, 640]" = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_98: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_98, 640);  rsqrt_98 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1542: "f32[640, 80]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
        permute_1546: "f32[80, 640]" = torch.ops.aten.permute.default(permute_593, [1, 0]);  permute_593 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1548: "f32[640, 640]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1555: "f32[640, 80]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
        permute_1559: "f32[80, 640]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1561: "f32[640, 640]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1564: "f32[640, 80]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
        permute_1568: "f32[80, 640]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1570: "f32[640, 640]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1573: "f32[640, 80]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
        permute_1577: "f32[80, 640]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1579: "f32[640, 640]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_100: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_97, 640);  rsqrt_97 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1582: "f32[640, 80]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
        permute_1586: "f32[80, 640]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1595: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
        permute_1599: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1601: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1604: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
        permute_1608: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_564, [1, 0]);  permute_564 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1610: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_563, [1, 0]);  permute_563 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1613: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
        permute_1617: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1619: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_102: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_93, 1280);  rsqrt_93 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1622: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_559, [1, 0]);  permute_559 = None
        permute_1626: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_558, [1, 0]);  permute_558 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1628: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1635: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_549, [1, 0]);  permute_549 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1642: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1649: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
        permute_1653: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1655: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_104: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_92, 1280);  rsqrt_92 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1658: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
        permute_1662: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1664: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1671: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
        permute_1675: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1677: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1680: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
        permute_1684: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1686: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_525, [1, 0]);  permute_525 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1689: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
        permute_1693: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1695: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_106: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_91, 1280);  rsqrt_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1698: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_521, [1, 0]);  permute_521 = None
        permute_1702: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1711: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_514, [1, 0]);  permute_514 = None
        permute_1715: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1717: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1720: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
        permute_1724: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1726: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_509, [1, 0]);  permute_509 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1729: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
        permute_1733: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1735: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_108: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_87, 1280);  rsqrt_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1738: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
        permute_1742: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1744: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1751: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1758: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1765: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
        permute_1769: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1771: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_110: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_86, 1280);  rsqrt_86 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1774: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
        permute_1778: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_485, [1, 0]);  permute_485 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1780: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1787: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
        permute_1791: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1793: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1796: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
        permute_1800: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1802: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1805: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
        permute_1809: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1811: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_112: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_85, 1280);  rsqrt_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1814: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
        permute_1818: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1827: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
        permute_1831: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1833: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1836: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
        permute_1840: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1842: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1845: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
        permute_1849: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_453, [1, 0]);  permute_453 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1851: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_114: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_81, 1280);  rsqrt_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1854: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
        permute_1858: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1860: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1867: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1874: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1881: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
        permute_1885: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1887: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_116: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_80, 1280);  rsqrt_80 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1890: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
        permute_1894: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1896: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1903: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
        permute_1907: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1909: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1912: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
        permute_1916: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1918: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1921: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
        permute_1925: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1927: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_118: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_79, 1280);  rsqrt_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1930: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
        permute_1934: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1951: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
        permute_1955: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1957: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1960: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        permute_1964: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1966: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1969: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
        permute_1973: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1975: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_124: "f32[4, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_67, 1280);  rsqrt_67 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1978: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
        permute_1982: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_1984: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1991: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_1998: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2005: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
        permute_2009: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2011: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_126: "f32[4, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_66, 1280);  rsqrt_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2014: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        permute_2018: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2020: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2027: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
        permute_2031: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2033: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2036: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
        permute_2040: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2042: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2045: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
        permute_2049: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2051: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_128: "f32[4, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_65, 1280);  rsqrt_65 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2054: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
        permute_2058: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2071: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
        permute_2075: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2077: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2080: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
        permute_2084: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2086: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2089: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
        permute_2093: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2095: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_132: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 1280);  rsqrt_57 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2098: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        permute_2102: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2104: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2111: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2118: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2125: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
        permute_2129: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2131: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_134: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 1280);  rsqrt_56 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2134: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
        permute_2138: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2140: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2147: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        permute_2151: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2153: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2156: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        permute_2160: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2162: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2165: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
        permute_2169: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2171: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_136: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 1280);  rsqrt_55 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2174: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        permute_2178: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2187: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
        permute_2191: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2193: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2196: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
        permute_2200: "f32[80, 5120]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2202: "f32[1280, 5120]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2205: "f32[10240, 80]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
        permute_2209: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2211: "f32[10240, 1280]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_138: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 1280);  rsqrt_51 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2214: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        permute_2218: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2220: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2227: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2234: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2241: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
        permute_2245: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2247: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_140: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 1280);  rsqrt_50 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2250: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        permute_2254: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2256: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2263: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_2267: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2269: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2272: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        permute_2276: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2278: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2281: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
        permute_2285: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2287: "f32[1280, 1280]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_142: "f32[4, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1280);  rsqrt_49 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2290: "f32[1280, 80]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        permute_2294: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2303: "f32[640, 80]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        permute_2307: "f32[80, 640]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2309: "f32[640, 640]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2312: "f32[640, 80]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        permute_2316: "f32[80, 2560]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2318: "f32[640, 2560]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2321: "f32[5120, 80]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_2325: "f32[80, 640]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2327: "f32[5120, 640]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_144: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 640);  rsqrt_45 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2330: "f32[640, 80]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        permute_2334: "f32[80, 640]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2336: "f32[640, 640]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2343: "f32[640, 80]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2350: "f32[640, 80]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2357: "f32[640, 80]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        permute_2361: "f32[80, 640]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2363: "f32[640, 640]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_146: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 640);  rsqrt_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2366: "f32[640, 80]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        permute_2370: "f32[80, 640]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2372: "f32[640, 640]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2379: "f32[640, 80]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_2383: "f32[80, 640]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2385: "f32[640, 640]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2388: "f32[640, 80]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        permute_2392: "f32[80, 640]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2394: "f32[640, 640]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2397: "f32[640, 80]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_2401: "f32[80, 640]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2403: "f32[640, 640]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_148: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 640);  rsqrt_43 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2406: "f32[640, 80]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_2410: "f32[80, 640]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2419: "f32[640, 80]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_2423: "f32[80, 640]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2425: "f32[640, 640]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2428: "f32[640, 80]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_2432: "f32[80, 2560]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2434: "f32[640, 2560]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2437: "f32[5120, 80]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_2441: "f32[80, 640]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2443: "f32[5120, 640]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_150: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 640);  rsqrt_39 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2446: "f32[640, 80]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        permute_2450: "f32[80, 640]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2452: "f32[640, 640]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2459: "f32[640, 80]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2466: "f32[640, 80]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2473: "f32[640, 80]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        permute_2477: "f32[80, 640]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2479: "f32[640, 640]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_152: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 640);  rsqrt_38 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2482: "f32[640, 80]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_2486: "f32[80, 640]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2488: "f32[640, 640]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2495: "f32[640, 80]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        permute_2499: "f32[80, 640]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2501: "f32[640, 640]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2504: "f32[640, 80]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        permute_2508: "f32[80, 640]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2510: "f32[640, 640]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2513: "f32[640, 80]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_2517: "f32[80, 640]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2519: "f32[640, 640]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_154: "f32[4, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 640);  rsqrt_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2522: "f32[640, 80]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        permute_2526: "f32[80, 640]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2535: "f32[320, 80]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_2539: "f32[80, 320]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2541: "f32[320, 320]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2544: "f32[320, 80]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        permute_2548: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2550: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2553: "f32[2560, 80]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_2557: "f32[80, 320]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2559: "f32[2560, 320]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_156: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 320);  rsqrt_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2562: "f32[320, 80]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_2566: "f32[80, 320]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2568: "f32[320, 320]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2575: "f32[320, 80]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2582: "f32[320, 80]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2589: "f32[320, 80]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        permute_2593: "f32[80, 320]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2595: "f32[320, 320]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_158: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 320);  rsqrt_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2598: "f32[320, 80]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        permute_2602: "f32[80, 320]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2604: "f32[320, 320]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2611: "f32[320, 80]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_2615: "f32[80, 320]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2617: "f32[320, 320]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2620: "f32[320, 80]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_2624: "f32[80, 320]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2626: "f32[320, 320]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2629: "f32[320, 80]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_2633: "f32[80, 320]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2635: "f32[320, 320]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_160: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 320);  rsqrt_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2638: "f32[320, 80]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        permute_2642: "f32[80, 320]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2651: "f32[320, 80]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_2655: "f32[80, 320]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2657: "f32[320, 320]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2660: "f32[320, 80]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        permute_2664: "f32[80, 1280]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2666: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2669: "f32[2560, 80]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_2673: "f32[80, 320]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2675: "f32[2560, 320]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:565 in forward, code: norm_hidden_states = self.norm3(hidden_states)
        div_162: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 320);  rsqrt_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2678: "f32[320, 80]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_2682: "f32[80, 320]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2684: "f32[320, 320]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2691: "f32[320, 80]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2698: "f32[320, 80]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2705: "f32[320, 80]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        permute_2709: "f32[80, 320]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2711: "f32[320, 320]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:539 in forward, code: norm_hidden_states = self.norm2(hidden_states)
        div_164: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 320);  rsqrt_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2714: "f32[320, 80]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_2718: "f32[80, 320]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2720: "f32[320, 320]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2727: "f32[320, 80]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_2731: "f32[80, 320]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2733: "f32[320, 320]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2736: "f32[320, 80]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_2740: "f32[80, 320]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2742: "f32[320, 320]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2745: "f32[320, 80]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_2749: "f32[80, 320]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2751: "f32[320, 320]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention.py:495 in forward, code: norm_hidden_states = self.norm1(hidden_states)
        div_166: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 320);  rsqrt_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2754: "f32[320, 80]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        permute_2758: "f32[80, 320]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2770: "f32[512, 4]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_2774: "f32[4, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_2776: "f32[512, 512]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2783: "f32[512, 4]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_2787: "f32[4, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2792: "f32[512, 4]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_2796: "f32[4, 512]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_2801: "f32[512, 4]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_2805: "f32[4, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        return (add_718, mul_98, add, add_15, add_31, add_47, primals_1, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_117, primals_118, primals_120, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_136, primals_140, primals_144, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_166, primals_168, primals_169, primals_170, primals_178, primals_180, primals_181, primals_182, primals_183, primals_184, primals_186, primals_187, primals_190, primals_191, primals_192, primals_194, primals_195, primals_196, primals_198, primals_202, primals_217, primals_232, primals_246, primals_247, primals_248, primals_250, primals_251, primals_254, primals_255, primals_256, primals_258, primals_259, primals_260, primals_262, primals_266, primals_281, primals_296, primals_310, primals_312, primals_313, primals_314, primals_315, primals_316, primals_318, primals_319, primals_322, primals_323, primals_324, primals_326, primals_327, primals_328, primals_330, primals_331, primals_332, primals_334, primals_338, primals_353, primals_368, primals_382, primals_383, primals_384, primals_386, primals_387, primals_390, primals_391, primals_392, primals_394, primals_395, primals_396, primals_398, primals_402, primals_417, primals_432, primals_446, primals_448, primals_449, primals_450, primals_451, primals_452, primals_454, primals_455, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_466, primals_467, primals_468, primals_470, primals_474, primals_489, primals_504, primals_518, primals_519, primals_520, primals_522, primals_523, primals_526, primals_527, primals_528, primals_530, primals_531, primals_532, primals_534, primals_538, primals_553, primals_568, primals_582, primals_584, primals_585, primals_586, primals_587, primals_588, primals_590, primals_591, primals_594, primals_595, primals_596, primals_598, primals_599, primals_600, primals_601, primals_602, primals_604, primals_605, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_616, primals_618, primals_619, primals_622, primals_623, primals_624, primals_626, primals_627, primals_628, primals_630, primals_634, primals_649, primals_664, primals_678, primals_679, primals_680, primals_682, primals_683, primals_686, primals_687, primals_688, primals_690, primals_691, primals_692, primals_693, primals_694, primals_696, primals_697, primals_700, primals_701, primals_702, primals_704, primals_705, primals_706, primals_708, primals_709, primals_710, primals_711, primals_712, primals_714, primals_715, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_726, primals_727, primals_728, primals_729, primals_730, primals_732, primals_733, primals_736, primals_737, primals_738, primals_740, primals_741, primals_742, primals_744, primals_745, primals_746, primals_748, primals_749, primals_750, primals_751, primals_752, primals_754, primals_755, primals_758, primals_759, primals_760, primals_762, primals_763, primals_764, primals_766, primals_767, primals_768, primals_770, primals_774, primals_789, primals_804, primals_818, primals_819, primals_820, primals_822, primals_823, primals_826, primals_827, primals_828, primals_830, primals_831, primals_832, primals_834, primals_835, primals_836, primals_838, primals_842, primals_857, primals_872, primals_886, primals_887, primals_888, primals_890, primals_891, primals_894, primals_895, primals_896, primals_898, primals_899, primals_900, primals_902, primals_903, primals_904, primals_906, primals_910, primals_925, primals_940, primals_954, primals_956, primals_957, primals_958, primals_959, primals_960, primals_962, primals_963, primals_966, primals_967, primals_968, primals_970, primals_971, primals_972, primals_974, primals_975, primals_976, primals_978, primals_982, primals_997, primals_1012, primals_1026, primals_1027, primals_1028, primals_1030, primals_1031, primals_1034, primals_1035, primals_1036, primals_1038, primals_1039, primals_1040, primals_1042, primals_1043, primals_1044, primals_1046, primals_1050, primals_1065, primals_1080, primals_1094, primals_1095, primals_1096, primals_1098, primals_1099, primals_1102, primals_1103, primals_1104, primals_1106, primals_1107, primals_1108, primals_1110, primals_1111, primals_1112, primals_1114, primals_1118, primals_1133, primals_1148, primals_1162, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1170, primals_1171, primals_1174, primals_1175, primals_1176, primals_1178, primals_1179, primals_1180, primals_1182, primals_1183, primals_1184, primals_1186, primals_1190, primals_1205, primals_1220, primals_1234, primals_1235, primals_1236, primals_1238, primals_1239, primals_1242, primals_1243, primals_1244, primals_1246, primals_1247, primals_1248, primals_1250, primals_1251, primals_1252, primals_1254, primals_1258, primals_1273, primals_1288, primals_1302, primals_1303, primals_1304, primals_1306, primals_1307, primals_1310, primals_1311, primals_1312, primals_1314, primals_1315, primals_1316, primals_1318, primals_1319, primals_1320, primals_1322, primals_1326, primals_1341, primals_1356, primals_1370, primals_1371, primals_1372, primals_1374, primals_1375, convolution_1, add, getitem_1, rsqrt, mul_3, convolution_4, add_3, getitem_3, rsqrt_1, mul_7, convolution_7, div, getitem_5, rsqrt_2, mul_11, convolution_10, add_10, getitem_7, rsqrt_3, mul_15, convolution_13, constant_pad_nd, convolution_16, add_15, getitem_9, rsqrt_4, mul_20, convolution_19, add_18, getitem_11, rsqrt_5, mul_24, convolution_22, convolution_25, div_2, getitem_13, rsqrt_6, mul_29, convolution_28, add_26, getitem_15, rsqrt_7, mul_33, convolution_31, constant_pad_nd_1, convolution_34, add_31, getitem_17, rsqrt_8, mul_38, convolution_37, add_34, getitem_19, rsqrt_9, mul_42, convolution_40, convolution_43, div_4, getitem_21, rsqrt_10, mul_47, convolution_46, add_42, getitem_23, rsqrt_11, mul_51, convolution_49, constant_pad_nd_2, convolution_52, add_47, getitem_25, rsqrt_12, mul_56, convolution_55, add_50, getitem_27, rsqrt_13, mul_60, convolution_58, div_6, getitem_29, rsqrt_14, mul_64, convolution_61, add_57, getitem_31, rsqrt_15, mul_68, convolution_64, div_7, getitem_33, rsqrt_16, mul_72, convolution_67, add_64, getitem_35, rsqrt_17, mul_76, convolution_70, view_36, squeeze_36, squeeze_37, view_42, mm, mm_2, mm_4, permute_15, permute_16, permute_17, getitem_38, getitem_39, getitem_40, getitem_41, mm_6, clone_13, getitem_43, rsqrt_19, mul_86, convolution_73, add_81, getitem_45, rsqrt_20, mul_90, convolution_76, clone_15, getitem_47, rsqrt_21, mul_94, convolution_79, add_88, getitem_49, inductor_random_default, mul_98, convolution_83, add_91, getitem_51, rsqrt_22, mul_107, convolution_86, add_95, getitem_53, rsqrt_23, mul_112, convolution_89, div_12, squeeze_48, squeeze_49, view_93, mm_8, mul_117, view_97, mm_11, mm_14, mm_17, permute_42, permute_43, permute_44, getitem_58, getitem_59, getitem_60, getitem_61, mm_19, mul_123, view_140, mm_22, view_148, mm_25, mm_28, permute_61, permute_62, permute_63, getitem_64, getitem_65, getitem_66, getitem_67, mm_30, mul_129, view_183, mm_32, getitem_73, getitem_74, view_194, mm_34, view_203, mm_36, add_125, getitem_77, rsqrt_28, mul_140, convolution_92, add_129, getitem_79, rsqrt_29, mul_145, convolution_95, div_15, squeeze_54, squeeze_55, view_225, mm_38, mul_150, view_229, mm_41, mm_44, mm_47, permute_96, permute_97, permute_98, getitem_84, getitem_85, getitem_86, getitem_87, mm_49, mul_156, view_272, mm_52, mm_55, mm_58, permute_115, permute_116, permute_117, getitem_90, getitem_91, getitem_92, getitem_93, mm_60, mul_162, view_315, mm_62, getitem_99, getitem_100, view_326, mm_64, view_335, mm_66, add_159, convolution_98, add_160, getitem_103, rsqrt_34, mul_174, convolution_101, add_164, getitem_105, rsqrt_35, mul_179, convolution_104, convolution_107, div_18, squeeze_60, squeeze_61, view_357, mm_68, mul_185, view_361, mm_71, mm_74, mm_77, permute_150, permute_151, permute_152, getitem_110, getitem_111, getitem_112, getitem_113, mm_79, mul_191, view_404, mm_82, mm_85, mm_88, permute_169, permute_170, permute_171, getitem_116, getitem_117, getitem_118, getitem_119, mm_90, mul_197, view_447, mm_92, getitem_125, getitem_126, view_458, mm_94, view_467, mm_96, add_195, getitem_129, rsqrt_40, mul_208, convolution_110, add_199, getitem_131, rsqrt_41, mul_213, convolution_113, div_21, squeeze_66, squeeze_67, view_489, mm_98, mul_218, view_493, mm_101, mm_104, mm_107, permute_204, permute_205, permute_206, getitem_136, getitem_137, getitem_138, getitem_139, mm_109, mul_224, view_536, mm_112, mm_115, mm_118, permute_223, permute_224, permute_225, getitem_142, getitem_143, getitem_144, getitem_145, mm_120, mul_230, view_579, mm_122, getitem_151, getitem_152, view_590, mm_124, view_599, mm_126, add_229, convolution_116, add_230, getitem_155, rsqrt_46, mul_242, convolution_119, add_234, getitem_157, rsqrt_47, mul_247, convolution_122, convolution_125, div_24, squeeze_72, squeeze_73, view_621, mm_128, mul_253, view_625, mm_131, mm_134, mm_137, permute_258, permute_259, permute_260, getitem_162, getitem_163, getitem_164, getitem_165, mm_139, mul_259, view_668, mm_142, mm_145, mm_148, permute_277, permute_278, permute_279, getitem_168, getitem_169, getitem_170, getitem_171, mm_150, mul_265, view_711, mm_152, getitem_177, getitem_178, view_722, mm_154, view_731, mm_156, add_265, getitem_181, rsqrt_52, mul_276, convolution_128, add_269, getitem_183, rsqrt_53, mul_281, convolution_131, div_27, squeeze_78, squeeze_79, view_753, mm_158, mul_286, view_757, mm_161, mm_164, mm_167, permute_312, permute_313, permute_314, getitem_188, getitem_189, getitem_190, getitem_191, mm_169, mul_292, view_800, mm_172, mm_175, mm_178, permute_331, permute_332, permute_333, getitem_194, getitem_195, getitem_196, getitem_197, mm_180, mul_298, view_843, mm_182, getitem_203, getitem_204, view_854, mm_184, view_863, mm_186, add_299, convolution_134, add_300, getitem_207, rsqrt_58, mul_310, convolution_137, add_304, getitem_209, rsqrt_59, mul_315, convolution_140, div_30, getitem_211, rsqrt_60, mul_319, convolution_143, add_312, getitem_213, rsqrt_61, mul_324, convolution_146, div_31, getitem_215, rsqrt_62, mul_328, convolution_149, add_320, getitem_217, rsqrt_63, mul_333, convolution_152, div_32, squeeze_92, squeeze_93, view_893, mm_188, mul_338, view_897, mm_191, mm_194, mm_197, permute_368, permute_369, permute_370, getitem_222, getitem_223, getitem_224, getitem_225, mm_199, mul_344, view_940, mm_202, mm_205, mm_208, permute_387, permute_388, permute_389, getitem_228, getitem_229, getitem_230, getitem_231, mm_210, mul_350, view_983, mm_212, getitem_237, getitem_238, view_994, mm_214, view_1003, mm_216, add_350, getitem_241, rsqrt_68, mul_361, convolution_155, add_354, getitem_243, rsqrt_69, mul_366, convolution_158, cat_2, getitem_245, rsqrt_70, mul_370, convolution_161, add_362, getitem_247, rsqrt_71, mul_375, convolution_164, convolution_167, cat_3, getitem_249, rsqrt_72, mul_380, convolution_170, add_371, getitem_251, rsqrt_73, mul_385, convolution_173, convolution_176, cat_4, getitem_253, rsqrt_74, mul_390, convolution_179, add_380, getitem_255, rsqrt_75, mul_395, convolution_182, convolution_185, convert_element_type_3, _unsafe_index, convolution_188, cat_5, getitem_257, rsqrt_76, mul_405, convolution_191, add_394, getitem_259, rsqrt_77, mul_410, convolution_194, convolution_197, div_39, squeeze_114, squeeze_115, view_1041, mm_218, mul_416, view_1045, mm_221, mm_224, mm_227, permute_426, permute_427, permute_428, getitem_264, getitem_265, getitem_266, getitem_267, mm_229, mul_422, view_1088, mm_232, mm_235, mm_238, permute_445, permute_446, permute_447, getitem_270, getitem_271, getitem_272, getitem_273, mm_240, mul_428, view_1131, mm_242, getitem_279, getitem_280, view_1142, mm_244, view_1151, mm_246, cat_6, getitem_283, rsqrt_82, mul_439, convolution_200, add_429, getitem_285, rsqrt_83, mul_444, convolution_203, convolution_206, div_42, squeeze_120, squeeze_121, view_1173, mm_248, mul_450, view_1177, mm_251, mm_254, mm_257, permute_480, permute_481, permute_482, getitem_290, getitem_291, getitem_292, getitem_293, mm_259, mul_456, view_1220, mm_262, mm_265, mm_268, permute_499, permute_500, permute_501, getitem_296, getitem_297, getitem_298, getitem_299, mm_270, mul_462, view_1263, mm_272, getitem_305, getitem_306, view_1274, mm_274, view_1283, mm_276, cat_7, getitem_309, rsqrt_88, mul_473, convolution_209, add_464, getitem_311, rsqrt_89, mul_478, convolution_212, convolution_215, div_45, squeeze_126, squeeze_127, view_1305, mm_278, mul_484, view_1309, mm_281, mm_284, mm_287, permute_534, permute_535, permute_536, getitem_316, getitem_317, getitem_318, getitem_319, mm_289, mul_490, view_1352, mm_292, mm_295, mm_298, permute_553, permute_554, permute_555, getitem_322, getitem_323, getitem_324, getitem_325, mm_300, mul_496, view_1395, mm_302, getitem_331, getitem_332, view_1406, mm_304, view_1415, mm_306, convert_element_type_7, _unsafe_index_1, convolution_218, cat_8, getitem_335, rsqrt_94, mul_512, convolution_221, add_504, getitem_337, rsqrt_95, mul_517, convolution_224, convolution_227, div_48, squeeze_132, squeeze_133, view_1437, mm_308, mul_523, view_1441, mm_311, mm_314, mm_317, permute_588, permute_589, permute_590, getitem_342, getitem_343, getitem_344, getitem_345, mm_319, mul_529, view_1484, mm_322, mm_325, mm_328, permute_607, permute_608, permute_609, getitem_348, getitem_349, getitem_350, getitem_351, mm_330, mul_535, view_1527, mm_332, getitem_357, getitem_358, view_1538, mm_334, view_1547, mm_336, cat_9, getitem_361, rsqrt_100, mul_546, convolution_230, add_539, getitem_363, rsqrt_101, mul_551, convolution_233, convolution_236, div_51, squeeze_138, squeeze_139, view_1569, mm_338, mul_557, view_1573, mm_341, mm_344, mm_347, permute_642, permute_643, permute_644, getitem_368, getitem_369, getitem_370, getitem_371, mm_349, mul_563, view_1616, mm_352, mm_355, mm_358, permute_661, permute_662, permute_663, getitem_374, getitem_375, getitem_376, getitem_377, mm_360, mul_569, view_1659, mm_362, getitem_383, getitem_384, view_1670, mm_364, view_1679, mm_366, cat_10, getitem_387, rsqrt_106, mul_580, convolution_239, add_574, getitem_389, rsqrt_107, mul_585, convolution_242, convolution_245, div_54, squeeze_144, squeeze_145, view_1701, mm_368, mul_591, view_1705, mm_371, mm_374, mm_377, permute_696, permute_697, permute_698, getitem_394, getitem_395, getitem_396, getitem_397, mm_379, mul_597, view_1748, mm_382, mm_385, mm_388, permute_715, permute_716, permute_717, getitem_400, getitem_401, getitem_402, getitem_403, mm_390, mul_603, view_1791, mm_392, getitem_409, getitem_410, view_1802, mm_394, view_1811, mm_396, convert_element_type_11, _unsafe_index_2, convolution_248, cat_11, getitem_413, rsqrt_112, mul_619, convolution_251, add_614, getitem_415, rsqrt_113, mul_624, convolution_254, convolution_257, div_57, squeeze_150, squeeze_151, view_1833, mm_398, mul_630, view_1837, mm_401, mm_404, mm_407, permute_750, permute_751, permute_752, getitem_420, getitem_421, getitem_422, getitem_423, mm_409, mul_636, view_1880, mm_412, mm_415, mm_418, permute_769, permute_770, permute_771, getitem_426, getitem_427, getitem_428, getitem_429, mm_420, mul_642, view_1923, mm_422, getitem_435, getitem_436, view_1934, mm_424, view_1943, mm_426, cat_12, getitem_439, rsqrt_118, mul_653, convolution_260, add_649, getitem_441, rsqrt_119, mul_658, convolution_263, convolution_266, div_60, squeeze_156, squeeze_157, view_1965, mm_428, mul_664, view_1969, mm_431, mm_434, mm_437, permute_804, permute_805, permute_806, getitem_446, getitem_447, getitem_448, getitem_449, mm_439, mul_670, view_2012, mm_442, mm_445, mm_448, permute_823, permute_824, permute_825, getitem_452, getitem_453, getitem_454, getitem_455, mm_450, mul_676, view_2055, mm_452, getitem_461, getitem_462, view_2066, mm_454, view_2075, mm_456, cat_13, getitem_465, rsqrt_124, mul_687, convolution_269, add_684, getitem_467, rsqrt_125, mul_692, convolution_272, convolution_275, div_63, squeeze_162, squeeze_163, view_2097, mm_458, mul_698, view_2101, mm_461, mm_464, mm_467, permute_858, permute_859, permute_860, getitem_472, getitem_473, getitem_474, getitem_475, mm_469, mul_704, view_2144, mm_472, mm_475, mm_478, permute_877, permute_878, permute_879, getitem_478, getitem_479, getitem_480, getitem_481, mm_480, mul_710, view_2187, mm_482, getitem_487, getitem_488, view_2198, mm_484, view_2207, mm_486, add_715, getitem_491, rsqrt_130, mul_721, convolution_278, permute_899, permute_903, permute_905, permute_908, permute_912, permute_914, permute_917, permute_921, permute_923, div_66, permute_926, permute_930, permute_932, permute_939, permute_946, permute_953, permute_957, permute_959, div_68, permute_962, permute_966, permute_968, permute_975, permute_979, permute_981, permute_984, permute_988, permute_990, permute_993, permute_997, permute_999, div_70, permute_1002, permute_1006, permute_1015, permute_1019, permute_1021, permute_1024, permute_1028, permute_1030, permute_1033, permute_1037, permute_1039, div_72, permute_1042, permute_1046, permute_1048, permute_1055, permute_1062, permute_1069, permute_1073, permute_1075, div_74, permute_1078, permute_1082, permute_1084, permute_1091, permute_1095, permute_1097, permute_1100, permute_1104, permute_1106, permute_1109, permute_1113, permute_1115, div_76, permute_1118, permute_1122, permute_1131, permute_1135, permute_1137, permute_1140, permute_1144, permute_1146, permute_1149, permute_1153, permute_1155, div_78, permute_1158, permute_1162, permute_1164, permute_1171, permute_1178, permute_1185, permute_1189, permute_1191, div_80, permute_1194, permute_1198, permute_1200, permute_1207, permute_1211, permute_1213, permute_1216, permute_1220, permute_1222, permute_1225, permute_1229, permute_1231, div_82, permute_1234, permute_1238, permute_1247, permute_1251, permute_1253, permute_1256, permute_1260, permute_1262, permute_1265, permute_1269, permute_1271, div_84, permute_1274, permute_1278, permute_1280, permute_1287, permute_1294, permute_1301, permute_1305, permute_1307, div_86, permute_1310, permute_1314, permute_1316, permute_1323, permute_1327, permute_1329, permute_1332, permute_1336, permute_1338, permute_1341, permute_1345, permute_1347, div_88, permute_1350, permute_1354, permute_1363, permute_1367, permute_1369, permute_1372, permute_1376, permute_1378, permute_1381, permute_1385, permute_1387, div_90, permute_1390, permute_1394, permute_1396, permute_1403, permute_1410, permute_1417, permute_1421, permute_1423, div_92, permute_1426, permute_1430, permute_1432, permute_1439, permute_1443, permute_1445, permute_1448, permute_1452, permute_1454, permute_1457, permute_1461, permute_1463, div_94, permute_1466, permute_1470, permute_1479, permute_1483, permute_1485, permute_1488, permute_1492, permute_1494, permute_1497, permute_1501, permute_1503, div_96, permute_1506, permute_1510, permute_1512, permute_1519, permute_1526, permute_1533, permute_1537, permute_1539, div_98, permute_1542, permute_1546, permute_1548, permute_1555, permute_1559, permute_1561, permute_1564, permute_1568, permute_1570, permute_1573, permute_1577, permute_1579, div_100, permute_1582, permute_1586, permute_1595, permute_1599, permute_1601, permute_1604, permute_1608, permute_1610, permute_1613, permute_1617, permute_1619, div_102, permute_1622, permute_1626, permute_1628, permute_1635, permute_1642, permute_1649, permute_1653, permute_1655, div_104, permute_1658, permute_1662, permute_1664, permute_1671, permute_1675, permute_1677, permute_1680, permute_1684, permute_1686, permute_1689, permute_1693, permute_1695, div_106, permute_1698, permute_1702, permute_1711, permute_1715, permute_1717, permute_1720, permute_1724, permute_1726, permute_1729, permute_1733, permute_1735, div_108, permute_1738, permute_1742, permute_1744, permute_1751, permute_1758, permute_1765, permute_1769, permute_1771, div_110, permute_1774, permute_1778, permute_1780, permute_1787, permute_1791, permute_1793, permute_1796, permute_1800, permute_1802, permute_1805, permute_1809, permute_1811, div_112, permute_1814, permute_1818, permute_1827, permute_1831, permute_1833, permute_1836, permute_1840, permute_1842, permute_1845, permute_1849, permute_1851, div_114, permute_1854, permute_1858, permute_1860, permute_1867, permute_1874, permute_1881, permute_1885, permute_1887, div_116, permute_1890, permute_1894, permute_1896, permute_1903, permute_1907, permute_1909, permute_1912, permute_1916, permute_1918, permute_1921, permute_1925, permute_1927, div_118, permute_1930, permute_1934, permute_1951, permute_1955, permute_1957, permute_1960, permute_1964, permute_1966, permute_1969, permute_1973, permute_1975, div_124, permute_1978, permute_1982, permute_1984, permute_1991, permute_1998, permute_2005, permute_2009, permute_2011, div_126, permute_2014, permute_2018, permute_2020, permute_2027, permute_2031, permute_2033, permute_2036, permute_2040, permute_2042, permute_2045, permute_2049, permute_2051, div_128, permute_2054, permute_2058, permute_2071, permute_2075, permute_2077, permute_2080, permute_2084, permute_2086, permute_2089, permute_2093, permute_2095, div_132, permute_2098, permute_2102, permute_2104, permute_2111, permute_2118, permute_2125, permute_2129, permute_2131, div_134, permute_2134, permute_2138, permute_2140, permute_2147, permute_2151, permute_2153, permute_2156, permute_2160, permute_2162, permute_2165, permute_2169, permute_2171, div_136, permute_2174, permute_2178, permute_2187, permute_2191, permute_2193, permute_2196, permute_2200, permute_2202, permute_2205, permute_2209, permute_2211, div_138, permute_2214, permute_2218, permute_2220, permute_2227, permute_2234, permute_2241, permute_2245, permute_2247, div_140, permute_2250, permute_2254, permute_2256, permute_2263, permute_2267, permute_2269, permute_2272, permute_2276, permute_2278, permute_2281, permute_2285, permute_2287, div_142, permute_2290, permute_2294, permute_2303, permute_2307, permute_2309, permute_2312, permute_2316, permute_2318, permute_2321, permute_2325, permute_2327, div_144, permute_2330, permute_2334, permute_2336, permute_2343, permute_2350, permute_2357, permute_2361, permute_2363, div_146, permute_2366, permute_2370, permute_2372, permute_2379, permute_2383, permute_2385, permute_2388, permute_2392, permute_2394, permute_2397, permute_2401, permute_2403, div_148, permute_2406, permute_2410, permute_2419, permute_2423, permute_2425, permute_2428, permute_2432, permute_2434, permute_2437, permute_2441, permute_2443, div_150, permute_2446, permute_2450, permute_2452, permute_2459, permute_2466, permute_2473, permute_2477, permute_2479, div_152, permute_2482, permute_2486, permute_2488, permute_2495, permute_2499, permute_2501, permute_2504, permute_2508, permute_2510, permute_2513, permute_2517, permute_2519, div_154, permute_2522, permute_2526, permute_2535, permute_2539, permute_2541, permute_2544, permute_2548, permute_2550, permute_2553, permute_2557, permute_2559, div_156, permute_2562, permute_2566, permute_2568, permute_2575, permute_2582, permute_2589, permute_2593, permute_2595, div_158, permute_2598, permute_2602, permute_2604, permute_2611, permute_2615, permute_2617, permute_2620, permute_2624, permute_2626, permute_2629, permute_2633, permute_2635, div_160, permute_2638, permute_2642, permute_2651, permute_2655, permute_2657, permute_2660, permute_2664, permute_2666, permute_2669, permute_2673, permute_2675, div_162, permute_2678, permute_2682, permute_2684, permute_2691, permute_2698, permute_2705, permute_2709, permute_2711, div_164, permute_2714, permute_2718, permute_2720, permute_2727, permute_2731, permute_2733, permute_2736, permute_2740, permute_2742, permute_2745, permute_2749, permute_2751, div_166, permute_2754, permute_2758, permute_2770, permute_2774, permute_2776, permute_2783, permute_2787, permute_2792, permute_2796, permute_2801, permute_2805)
        