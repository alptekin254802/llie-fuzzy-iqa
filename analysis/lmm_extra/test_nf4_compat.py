from pathlib import Path
import os,json,hashlib,datetime
import argparse
p=argparse.ArgumentParser(description='Verify packed NF4 weights and outputs through meta/CPU/CUDA round trips')
p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch,bitsandbytes as bnb
from accelerate.utils.modeling import set_module_tensor_to_device
from .adapters import install_nf4_meta_compat
torch.set_num_threads(1);torch.manual_seed(20261018)
x=torch.randn(3,64,device='cuda',dtype=torch.float32)
w=torch.randn(32,64,dtype=torch.float32)
def layer(device):
 m=bnb.nn.Linear4bit(64,32,bias=False,compute_dtype=torch.float32,compress_statistics=False,quant_type='nf4').eval()
 m.weight=bnb.nn.Params4bit(w.clone(),requires_grad=False,compress_statistics=False,quant_type='nf4',module=m).to(device)
 return m
def metadata(q):
 return {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else str(v) for k,v in q.as_dict().items()}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'torch':torch.__version__,'bnb':bnb.__version__,'adapter_sha256':hashlib.sha256((Path(__file__).resolve().parent/'adapters.py').read_bytes()).hexdigest(),'configuration':'NF4, float32 compute, no double quantization','tests':[]}
m=layer('cuda');payload=m.weight.data.cpu().clone()
try:
 set_module_tensor_to_device(m,'weight','meta')
 set_module_tensor_to_device(m,'weight','cuda:0',value=payload)
 result['unpatched_error']=None
except Exception as e:result['unpatched_error']=type(e).__name__+': '+str(e)
assert result['unpatched_error'] and 'meta tensor' in result['unpatched_error']
del m;torch.cuda.empty_cache()
install_nf4_meta_compat()
for initial in ['cuda','cpu']:
 m=layer(initial);payload=m.weight.data.cpu().clone();qs_before=metadata(m.weight.quant_state)
 if initial=='cpu':set_module_tensor_to_device(m,'weight','cuda:0',value=payload)
 before=m(x).detach().cpu();diffs=[]
 for i in range(3):
  set_module_tensor_to_device(m,'weight','meta')
  assert m.weight.quant_state.code.device.type=='cpu' and m.weight.quant_state.absmax.device.type=='cpu'
  set_module_tensor_to_device(m,'weight','cuda:0',value=payload)
  assert torch.equal(m.weight.data.cpu(),payload)
  afterqs=metadata(m.weight.quant_state)
  assert qs_before.keys()==afterqs.keys()
  assert all(torch.equal(v,afterqs[k]) if isinstance(v,torch.Tensor) else v==afterqs[k] for k,v in qs_before.items())
  after=m(x).detach().cpu();assert bool(torch.isfinite(after).all())
  diff=float((before-after).abs().max());diffs.append(diff);assert torch.equal(before,after)
 result['tests'].append({'initial_quantization_device':initial,'round_trips':3,'packed_weights_bit_equal':True,'metadata_bit_equal':True,'forward_outputs_bit_equal':True,'forward_max_abs_differences':diffs})
 del m;torch.cuda.empty_cache()
result['passed']=True
(O/'nf4_meta_compat_test.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
