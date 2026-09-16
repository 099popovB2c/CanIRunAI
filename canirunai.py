#!/usr/bin/env python3
import argparse,json,os,platform,shutil,subprocess
from pathlib import Path

def mem_gb():
 try:
  if os.name=='nt':
   out=subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],text=True,timeout=3);return int(out.strip())/1024**3
  if Path('/proc/meminfo').exists():
   kb=int(Path('/proc/meminfo').read_text().split('MemTotal:')[1].split()[0]);return kb/1024**2
  import resource; return 0
 except:return 0

def gpu():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0]
  name,mb=[x.strip() for x in out.rsplit(',',1)];return {'name':name,'vram_gb':round(float(mb)/1024,1)}
 except:return {'name':'Not detected','vram_gb':0}

def cpu_name():
 x=platform.processor().strip()
 if x:return x
 try:
  if Path('/proc/cpuinfo').exists():
   for l in Path('/proc/cpuinfo').read_text(errors='ignore').splitlines():
    if 'model name' in l:return l.split(':',1)[1].strip()
 except:pass
 return 'Unknown CPU'

def inspect():
 total,used,free=shutil.disk_usage(Path.home())
 return {'os':f'{platform.system()} {platform.release()}','cpu':cpu_name(),'ram_gb':round(mem_gb(),1),'gpu':gpu(),'disk_free_gb':round(free/1024**3,1)}

def rate(hw,m):
 ram=hw['ram_gb'];v=hw['gpu']['vram_gb'];disk=hw['disk_free_gb']
 if disk<m['disk_gb']*1.3:return 'NO','Not enough free disk'
 if v>=m['vram_gb'] and ram>=m['ram_gb']:return 'GOOD','Fits primarily in GPU/normal RAM budget'
 if ram>=m['ram_gb']*1.15:return 'SLOW','Can fit in system RAM; CPU/offload may be slow'
 return 'NO','Memory budget is too small'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--json');a=ap.parse_args();hw=inspect();models=json.loads(Path(__file__).with_name('models.json').read_text());rows=[]
 for m in models:
  status,note=rate(hw,m);rows.append({**m,'status':status,'note':note})
 print(f"OS: {hw['os']}\nCPU: {hw['cpu']}\nRAM: {hw['ram_gb']} GB\nGPU: {hw['gpu']['name']} ({hw['gpu']['vram_gb']} GB VRAM)\nFree disk: {hw['disk_free_gb']} GB\n")
 for r in rows: print(f"{r['status']:4}  {r['name']:<16} {r['note']}"+(f" | ollama pull {r['ollama']}" if r['status']!='NO' else ''))
 report={'hardware':hw,'models':rows}
 if a.json:Path(a.json).write_text(json.dumps(report,indent=2),encoding='utf8')
if __name__=='__main__':main()
