"""Official image prompts and processors, with hardware-only routing adapters. No LLIE fitting."""
from pathlib import Path
import os,sys,ast,math,re,io,json,contextlib,datetime,inspect,textwrap,hashlib
from typing import Dict
import numpy as np,torch
from PIL import Image
O=None
PATHS=None
IDS=[1550,1661,6624,7852,3347]
WEIGHTS=np.array([1,.75,.5,.25,0],dtype=np.float64)
from .scoring import quality
def extracted(path,names,extra):
 tree=ast.parse(path.read_text('utf-8-sig'))
 body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert {n.name for n in body}==set(names),(names,[n.name for n in body])
 ns=dict(torch=torch,np=np,Image=Image,math=math,re=re,ast=ast,Dict=Dict);ns.update(extra);ns.setdefault('transformers',__import__('transformers'))
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns);return ns
def install_nf4_meta_compat():
 """Keep real NF4 metadata while Accelerate parks the packed parameter on meta.

 bitsandbytes 0.49 mutates QuantState.to in place. Accelerate 0.34 retains
 that same object in the CPU weight map, so moving metadata to meta destroys
 its data. Packed weights still follow the identical CPU/CUDA dispatch.
 """
 import bitsandbytes as bnb
 cls=bnb.nn.Params4bit
 if getattr(cls.to,'_preserve_quant_metadata',False):return
 original=cls.to
 def to_preserve_metadata(self,*args,**kwargs):
  device,dtype,non_blocking,_=torch._C._nn._parse_to(*args,**kwargs)
  if device is not None and device.type=='meta' and self.quant_state is not None:
   quant_state=self.quant_state
   # Metadata remains real and offloaded on CPU, never a data-less meta tensor.
   quant_state.to(torch.device('cpu'))
   self.quant_state=None
   try:result=original(self,*args,**kwargs)
   finally:self.quant_state=quant_state
   result.quant_state=quant_state
   return result
  return original(self,*args,**kwargs)
 to_preserve_metadata._preserve_quant_metadata=True
 cls.to=to_preserve_metadata

def load(modelname,precision,paths,work):
 global O,PATHS
 O=Path(work).resolve();PATHS=paths
 (O/modelname).mkdir(parents=True,exist_ok=True)
 # The official VQA model resolves slowfast.pth from the process directory.
 os.chdir(O/modelname)
 from transformers import AutoTokenizer,BitsAndBytesConfig
 # pytorchvideo's legacy import path was removed in torchvision 0.17; the equivalent public tensor implementation is retained.
 import torchvision.transforms._functional_tensor as ft
 sys.modules.setdefault('torchvision.transforms.functional_tensor',ft)
 dtype={'bf16_cpu':torch.bfloat16,'fp16':torch.float16,'nf4_fp32':torch.float32}[precision]
 input_device='cpu' if precision=='bf16_cpu' else 'cuda'
 cfgkwargs=dict(torch_dtype=dtype,low_cpu_mem_usage=True,local_files_only=True,attn_implementation='eager',output_loading_info=True)
 if precision=='bf16_cpu':cfgkwargs.update(device_map={'':'cpu'})
 elif precision=='fp16':
  cfgkwargs.update(device_map='auto',max_memory={0:'5GiB','cpu':'20GiB'},offload_folder=str(O/modelname/'offload_fp16'),offload_state_dict=True)
 else:
  install_nf4_meta_compat()
  skips=['vision_tower','vision_model','mm_projector','slowfast','slowfast_projector','slowfast_model','mlp1','motion_mlp','lm_head','norm','input_layernorm','post_attention_layernorm','embed_tokens']
  qconfig=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=False,llm_int8_skip_modules=skips,llm_int8_enable_fp32_cpu_offload=True)
  cfgkwargs.update(quantization_config=qconfig)
  if os.environ.get('LMM_DEVICE_MAP','auto')=='gpu':cfgkwargs.update(device_map={'':0})
  else:cfgkwargs.update(device_map='auto',max_memory={0:'5GiB','cpu':'25GiB'},offload_folder=str(O/modelname/'offload_candidate'),offload_state_dict=True,offload_buffers=True)
 if modelname=='vqa2':
  src=Path(PATHS['source'])/'quality_scoring';sys.path.insert(0,str(src))
  from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM,LlavaQwenConfig
  from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
  from llava.constants import IGNORE_INDEX,DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
  # Official llava_arch references an undefined NUM; batch-one image index is exactly zero.
  import llava.model.llava_arch as image_arch
  image_arch.NUM=0
  # Change only hardcoded precision/device casts in memory, keeping official computations.
  cls=image_arch.LlavaMetaForCausalLM
  patches=[]
  for method in ['encode_images','prepare_inputs_labels_for_multimodal']:
   original=inspect.getsource(getattr(cls,method));changed=textwrap.dedent(original)
   changed=changed.replace('self.get_model().slowfast(images[0]).half()','self.get_model().slowfast(images[0]).to(dtype=_runtime_dtype)')
   changed=changed.replace('encoded_image_features.cuda()','encoded_image_features.to(device=input_ids.device)')
   changed=changed.replace('encoded_slowfast_features.cuda()','encoded_slowfast_features.to(device=input_ids.device)')
   ns_patch=dict(vars(image_arch));ns_patch['_runtime_dtype']=dtype
   exec(compile(changed,str(O/modelname/(method+'_runtime.py')),'exec'),ns_patch)
   setattr(cls,method,ns_patch[method]);patches.append({'method':method,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(changed.encode()).hexdigest()})
   (O/modelname/(method+'_runtime.py')).write_text(changed,encoding='utf-8')

  path=Path(PATHS['checkpoint'])
  slow=Path(PATHS['slowfast'])
  if not slow.exists():
   found=list(slow.parent.rglob('*.pth'));assert len(found)==1,found;slow=found[0]
  local=O/'vqa2/slowfast.pth'
  if not local.exists():
   import shutil
   shutil.copyfile(slow,local)
  tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True)
  config=LlavaQwenConfig.from_pretrained(path,local_files_only=True);config._attn_implementation='eager'
  # Keep residual blocks and the complete vision/motion towers intact for automatic CPU offload.
  LlavaQwenForCausalLM._no_split_modules=list(set((LlavaQwenForCausalLM._no_split_modules or [])+['SigLipVisionTower','slowfast','slowfast_projector']))
  original_load=torch.load
  def cpu_slowfast(*args,**kw):
   if args and str(args[0]).endswith('slowfast.pth'):
    kw['map_location']='cpu';obj=original_load(*args,**kw)
    return obj.to(dtype=dtype) if hasattr(obj,'to') else obj
   return original_load(*args,**kw)
  with contextlib.ExitStack() as stack:
   from unittest.mock import patch
   stack.enter_context(patch.object(torch,'load',cpu_slowfast))
   model,loading=LlavaQwenForCausalLM.from_pretrained(path,config=config,**cfgkwargs)
  def tower_device(self):
   p=next(self.vision_tower.parameters())
   if p.device.type!='meta':return p.device
   for m in self.vision_tower.modules():
    hook=getattr(m,'_hf_hook',None);device=getattr(hook,'execution_device',None)
    if device is not None:return torch.device(device)
   raise RuntimeError('No execution device for offloaded vision tower')
  SigLipVisionTower.device=property(tower_device)
  ns=extracted(src/'llava/eval/model_score_image.py',['select_best_resolution','resize_and_pad_image','divide_to_patches','get_anyres_image_grid_shape','process_anyres_image','preprocess_qwen'],dict(transformers=__import__('transformers'),IGNORE_INDEX=IGNORE_INDEX,DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX))
  input_ids=ns['preprocess_qwen'](['<image>',{'from':'gpt','value':'The overall quality of the image is'}],tokenizer,has_image=True).to(input_device)
  processor=model.get_vision_tower().image_processor
  def score(rgb):
   # The released processor takes a file-like object. PNG is lossless and preserves every area256 RGB byte.
   b=io.BytesIO();Image.fromarray(rgb).save(b,format='PNG');b.seek(0)
   image,size=ns['process_anyres_image'](b,processor)
   images=[[image.repeat(4,1,1,1).to(input_device,dtype=dtype)],[image.to(input_device,dtype=dtype)]]
   logits=model(input_ids,images=images,image_sizes=[size],modalities=['image'],use_cache=False)['logits'][:,-3].mean(0)[IDS].float().cpu().numpy()
   # Also retain the exact released endpoint-key defect as a diagnostic, never silently use it as five-level scoring.
   z=np.array([0.,float(logits[1]),float(logits[2]),float(logits[3]),0.]);pp=np.exp(z-z.max());pp/=pp.sum()
   raw=float(pp@np.array([1,.8,.6,.4,.2]))
   return quality(logits),logits.tolist(),raw
 elif modelname=='vital':
  src=Path(PATHS['source'])/'VITAL-LMM/test';sys.path.insert(0,str(src))
  from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
  from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
  from internvl.conversation import get_conv_template
  from internvl.train.constants import IMG_START_TOKEN,IMG_END_TOKEN,IMG_CONTEXT_TOKEN
  path=Path(PATHS['checkpoint'])
  tokenizer=AutoTokenizer.from_pretrained(path,use_fast=False,local_files_only=True,trust_remote_code=False)
  config=InternVLChatConfig.from_pretrained(path,local_files_only=True)
  config.llm_config._attn_implementation='eager';config.llm_config.attn_implementation='eager';config.vision_config.use_flash_attn=False
  InternVLChatModel._no_split_modules=list(set((InternVLChatModel._no_split_modules or [])+['InternVisionModel','slowfast']))
  # The full checkpoint supplies all motion-tower weights; skip redundant pretrained initialization.
  import pytorchvideo.models.hub as pvhub
  original_slowfast=pvhub.slowfast_r50
  import internvl.model.internvl_chat.modeling_internvl_chat as im
  with contextlib.ExitStack() as stack:
   from unittest.mock import patch
   stack.enter_context(patch.object(im,'slowfast_r50',lambda *a,**kw:original_slowfast(pretrained=False)))
   model,loading=InternVLChatModel.from_pretrained(path,config=config,use_flash_attn=False,**cfgkwargs)
  model.img_context_token_id=tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
  ds=src/'internvl/train/dataset.py'
  from torchvision.transforms import InterpolationMode
  import torchvision.transforms as T
  from copy import deepcopy
  extra=dict(get_conv_template=get_conv_template,IMG_START_TOKEN=IMG_START_TOKEN,IMG_END_TOKEN=IMG_END_TOKEN,IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,IGNORE_TOKEN_ID=-100,F=torch.nn.functional,deepcopy=deepcopy,T=T,InterpolationMode=InterpolationMode,IMAGENET_MEAN=(.485,.456,.406),IMAGENET_STD=(.229,.224,.225))
  ns=extracted(ds,['preprocess_internvl2_5','find_closest_aspect_ratio','dynamic_preprocess','build_transform'],extra)
  tokenizer.model_max_length=16384
  def score(rgb):
   image=Image.fromarray(rgb)
   patches=ns['dynamic_preprocess'](image,min_num=1,max_num=12,image_size=448,use_thumbnail=True)
   transform=ns['build_transform'](is_train=False,input_size=448,pad2square=False,normalize_type='imagenet')
   pixels=torch.stack([transform(p) for p in patches]).to(input_device,dtype=dtype)
   convs=[[{'from':'human','value':'<image>\n'},{'from':'gpt','value':'The quality of this image is'}]]
   ret=ns['preprocess_internvl2_5']('internvl2_5',convs,tokenizer,[model.num_image_token*len(patches)],group_by_length=True,ds_name='quality_evaluation',num_image=1)
   ret={k:v.to(input_device) for k,v in ret.items() if k in ['input_ids','attention_mask']}
   logits=model(pixel_values=pixels,**ret,modality=torch.tensor([0],device=input_device),image_flags=torch.ones((len(patches),1),dtype=torch.long,device=input_device),use_cache=False)['logits'][:,-3].mean(0)[IDS].float().cpu().numpy()
   return quality(logits),logits.tolist(),None
 else:raise ValueError(modelname)
 assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys'),loading
 model.eval()
 if precision=='nf4_fp32':
  import bitsandbytes as bnb
  quantized=[n for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)]
  assert quantized,'No quantized layers'
  assert all(m.compute_dtype==torch.float32 for m in model.modules() if isinstance(m,bnb.nn.Linear4bit))
  wrong=[(n,str(p.dtype),str(p.device)) for n,p in model.named_parameters() if p.is_floating_point() and p.dtype!=torch.float32]
 else:
  wrong=[(n,str(p.dtype),str(p.device)) for n,p in model.named_parameters() if p.is_floating_point() and p.dtype!=dtype];quantized=[]
 assert not wrong,('Unexpected floating parameter dtype',wrong[:50])
 assert not any(str(v)=='disk' for v in getattr(model,'hf_device_map',{}).values()),'Disk offloading is outside this protocol'
 if precision=='bf16_cpu':assert all(p.device.type=='cpu' for p in model.parameters())
 if precision=='nf4_fp32' and os.environ.get('LMM_DEVICE_MAP','auto')=='gpu':
  # Windows WDDM can silently back overcommitted CUDA allocations with host memory.
  # A full-GPU trial must fit the physical device before its first score.
  total=torch.cuda.get_device_properties(0).total_memory;allocated=torch.cuda.memory_allocated(0)
  fit={'dedicated_gpu_bytes':total,'allocated_after_load_bytes':allocated,'reserved_after_load_bytes':torch.cuda.memory_reserved(0),'fits_dedicated_gpu':allocated<=total,'phase':precision,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  (O/modelname/'candidate_gpu_memory_fit.json').write_text(json.dumps(fit,indent=2)+'\n',encoding='utf-8')
  if allocated>total:raise torch.cuda.OutOfMemoryError('Physical GPU out of memory guard before scoring: CUDA allocations exceed dedicated VRAM; use the prespecified auto-map fallback. '+json.dumps(fit))
 info={'device_map':{k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()},'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),'quality_token_ids':IDS,'quality_token_text':[tokenizer.decode([i]) for i in IDS],'weights':WEIGHTS.tolist(),'loading_info':loading,'quantized_output_head':False,'quantized_module_names':quantized,'compute_dtype':str(dtype),'input_device':input_device,'nf4_metadata_meta_compat':precision=='nf4_fp32','offload_buffers':cfgkwargs.get('offload_buffers',False),'runtime_cast_patches':patches if modelname=='vqa2' else []}
 return model,score,info
