#!/usr/bin/env python3
import argparse,json,os,platform,shutil,subprocess
from pathlib import Path
QUANTS={'Q2_K':0.34,'Q3_K_M':0.43,'Q4_K_M':0.55,'Q5_K_M':0.67,'Q6_K':0.80,'Q8_0':1.05}
def memory_info():
 try:
  if os.name=='nt':
   ps="$c=Get-CimInstance Win32_ComputerSystem;$o=Get-CimInstance Win32_OperatingSystem;@{total=$c.TotalPhysicalMemory;available=$o.FreePhysicalMemory*1KB}|ConvertTo-Json -Compress";d=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',ps],text=True,timeout=4));return float(d['total'])/1024**3,float(d['available'])/1024**3
  if Path('/proc/meminfo').exists():
   vals={};
   for line in Path('/proc/meminfo').read_text().splitlines():
    if ':' in line: k,v=line.split(':',1);vals[k]=int(v.strip().split()[0])
   return vals.get('MemTotal',0)/1024**2,vals.get('MemAvailable',0)/1024**2
 except:pass
 return 0,0
def gpus():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines();rows=[]
  for line in out:
   name,mb=[x.strip() for x in line.rsplit(',',1)];rows.append({'name':name,'vram_gb':round(float(mb)/1024,1)})
  return rows
 except:return []
def cpu_name():
 x=platform.processor().strip()
 if x:return x
 try:
  if Path('/proc/cpuinfo').exists():
   for l in Path('/proc/cpuinfo').read_text(errors='ignore').splitlines():
    if 'model name' in l:return l.split(':',1)[1].strip()
 except:pass
 return 'Unknown CPU'
def runtimes():
 candidates={'Ollama':['ollama'],'llama.cpp':['llama-cli','llama-server'],'LM Studio':['lms'],'vLLM':['vllm']};return {k:next((shutil.which(x) for x in names if shutil.which(x)),None) for k,names in candidates.items()}
def inspect():
 total,avail=memory_info();_,_,free=shutil.disk_usage(Path.home());gs=gpus();return {'os':f'{platform.system()} {platform.release()}','cpu':cpu_name(),'ram_gb':round(total,1),'ram_available_gb':round(avail,1),'gpus':gs,'vram_gb':round(sum(g['vram_gb'] for g in gs),1),'disk_free_gb':round(free/1024**3,1),'runtimes':runtimes()}
def estimate_gb(params_b,quant='Q4_K_M'):return round(params_b*QUANTS[quant]*1.20,2)
def fit(hw,params_b,quant,disk_gb=None):
 need=estimate_gb(params_b,quant);v=hw['vram_gb'];ram=hw['ram_gb'];disk=hw['disk_free_gb'];disk_need=disk_gb or need*.9
 if disk<disk_need*1.2:return 'NO',need,'Not enough free disk'
 if v>0:
  util=need/v
  if util<=.85:return 'GOOD',need,f'GPU fit at {util:.0%} of detected VRAM'
  if util<=.98:return 'TIGHT',need,f'Near VRAM limit at {util:.0%}; leave room for context/cache'
 ram_budget=max(0,ram-4)
 if need<=ram_budget:return 'SLOW',need,'Fits system RAM with CPU/GPU offload; expect lower speed'
 return 'NO',need,'Estimated model memory exceeds safe RAM/VRAM budget'
def override(hw,a):
 h=json.loads(json.dumps(hw))
 if a.ram is not None:h['ram_gb']=a.ram
 if a.vram is not None:h['vram_gb']=a.vram;h['gpus']=[{'name':'override','vram_gb':a.vram}] if a.vram else []
 if a.disk is not None:h['disk_free_gb']=a.disk
 return h
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--quant',choices=QUANTS,default='Q4_K_M');ap.add_argument('--ram',type=float);ap.add_argument('--vram',type=float);ap.add_argument('--disk',type=float);ap.add_argument('--model-size',type=float,help='Evaluate a custom model parameter count in billions');ap.add_argument('--json');a=ap.parse_args();hw=override(inspect(),a);models=json.loads(Path(__file__).with_name('models.json').read_text());
 if a.model_size:models=[{'name':f'Custom {a.model_size:g}B','params_b':a.model_size,'use_case':'custom','ollama':''}]
 print(f"OS: {hw['os']}\nCPU: {hw['cpu']}\nRAM: {hw['ram_gb']} GB (currently available {hw['ram_available_gb']} GB)\nGPU(s): "+(', '.join(f"{g['name']} {g['vram_gb']}GB" for g in hw['gpus']) or 'Not detected')+f"\nTotal VRAM: {hw['vram_gb']} GB\nFree disk: {hw['disk_free_gb']} GB\nQuantization: {a.quant}\n")
 installed=[k for k,v in hw['runtimes'].items() if v];print('Local runtimes: '+(', '.join(installed) if installed else 'none detected')+'\n');rows=[]
 for m in models:
  status,need,note=fit(hw,float(m['params_b']),a.quant,m.get('disk_gb'));r={**m,'quant':a.quant,'estimated_memory_gb':need,'status':status,'note':note};rows.append(r);pull=(f" | ollama pull {m['ollama']}" if m.get('ollama') and status!='NO' else '');print(f"{status:5} {m['name']:<20} ~{need:>5.1f} GB | {note}{pull}")
 report={'hardware':hw,'quantization':a.quant,'models':rows,'estimation_note':'Approximate weight+runtime overhead only; context length, KV cache, backend and architecture can change real usage.'}
 if a.json:Path(a.json).write_text(json.dumps(report,indent=2),encoding='utf8')
if __name__=='__main__':main()
