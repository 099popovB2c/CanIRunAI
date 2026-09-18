#!/usr/bin/env python3
import argparse,json,math,os,platform,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path

VERSION="0.4.0"
QUANTS={"Q2_K":0.34,"Q3_K_M":0.43,"Q4_K_M":0.55,"Q5_K_M":0.67,"Q6_K":0.80,"Q8_0":1.05}
KV_BYTES={"fp16":2.0,"q8":1.0,"q4":0.5}

def memory_info():
    try:
        if os.name=="nt":
            ps="$c=Get-CimInstance Win32_ComputerSystem;$o=Get-CimInstance Win32_OperatingSystem;@{total=$c.TotalPhysicalMemory;available=$o.FreePhysicalMemory*1KB}|ConvertTo-Json -Compress"
            d=json.loads(subprocess.check_output(["powershell","-NoProfile","-Command",ps],text=True,timeout=4));return float(d["total"])/1024**3,float(d["available"])/1024**3
        if Path("/proc/meminfo").exists():
            vals={}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if ":" in line:
                    k,v=line.split(":",1);vals[k]=int(v.strip().split()[0])
            return vals.get("MemTotal",0)/1024**2,vals.get("MemAvailable",0)/1024**2
    except Exception:pass
    return 0,0

def gpus():
    try:
        out=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],text=True,timeout=4).strip().splitlines();rows=[]
        for line in out:
            name,mb=[x.strip() for x in line.rsplit(",",1)];rows.append({"name":name,"vram_gb":round(float(mb)/1024,1)})
        return rows
    except Exception:return []

def cpu_name():
    x=platform.processor().strip()
    if x:return x
    try:
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if "model name" in line:return line.split(":",1)[1].strip()
    except Exception:pass
    return "Unknown CPU"

def runtimes():
    candidates={"Ollama":["ollama"],"llama.cpp":["llama-cli","llama-server"],"LM Studio":["lms"],"vLLM":["vllm"]};return {k:next((shutil.which(x) for x in names if shutil.which(x)),None) for k,names in candidates.items()}

def inspect():
    total,avail=memory_info();_,_,free=shutil.disk_usage(Path.home());gs=gpus();return {"os":f"{platform.system()} {platform.release()}","cpu":cpu_name(),"ram_gb":round(total,1),"ram_available_gb":round(avail,1),"gpus":gs,"vram_gb":round(sum(g["vram_gb"] for g in gs),1),"disk_free_gb":round(free/1024**3,1),"runtimes":runtimes()}

def weight_gb(params_b,quant="Q4_K_M"):return round(float(params_b)*QUANTS[quant]*1.12,2)
def kv_cache_gb(model,context=8192,kv_quant="fp16"):
    factor=KV_BYTES[kv_quant];layers=model.get("layers");hidden=model.get("hidden_size");heads=model.get("attention_heads");kv_heads=model.get("kv_heads")
    if all(isinstance(x,(int,float)) and x>0 for x in (layers,hidden,heads,kv_heads)):
        head_dim=hidden/heads;bytes_total=2*layers*context*kv_heads*head_dim*factor;return round(bytes_total/1024**3,2)
    params=float(model.get("params_b",1));gb_at_8k=0.20*math.sqrt(max(params,1));return round(gb_at_8k*(context/8192)*(factor/2.0),2)
def runtime_overhead_gb(params_b):return round(0.35+min(1.5,float(params_b)*0.015),2)
def memory_breakdown(model,quant="Q4_K_M",context=8192,kv_quant="fp16"):
    weights=weight_gb(model["params_b"],quant);kv=kv_cache_gb(model,context,kv_quant);runtime=runtime_overhead_gb(model["params_b"]);return {"weights_gb":weights,"kv_cache_gb":kv,"runtime_overhead_gb":runtime,"total_gb":round(weights+kv+runtime,2)}
def estimate_gb(params_b,quant="Q4_K_M"):return round(weight_gb(params_b,quant)+runtime_overhead_gb(params_b),2)
def fit_model(hw,model,quant="Q4_K_M",context=8192,kv_quant="fp16"):
    mem=memory_breakdown(model,quant,context,kv_quant);need=mem["total_gb"];v=float(hw.get("vram_gb",0));ram=float(hw.get("ram_gb",0));avail=float(hw.get("ram_available_gb",ram) or ram);disk=float(hw.get("disk_free_gb",0));disk_need=float(model.get("disk_gb") or mem["weights_gb"])
    if disk and disk<disk_need*1.15:return "NO",mem,"Not enough free disk"
    if v>0:
        util=need/v
        if util<=.82:return "GOOD",mem,f"Model + {context//1024}K context fits GPU with headroom"
        if util<=.96:return "TIGHT",mem,f"Near VRAM limit ({util:.0%}); reduce context or KV precision"
    ram_budget=max(0,min(ram-3,avail+max(0,ram-avail)*0.35))
    if need<=max(ram_budget,ram-5):return "SLOW",mem,"Needs CPU/RAM offload; usable but slower"
    return "NO",mem,"Estimated weights + KV cache + runtime overhead exceed safe memory budget"
def fit(hw,params_b,quant,disk_gb=None):
    m={"name":"custom","params_b":params_b}
    if disk_gb is not None:m["disk_gb"]=disk_gb
    status,mem,note=fit_model(hw,m,quant,context=0,kv_quant="fp16");return status,mem["total_gb"],note
def max_context_estimate(hw,model,quant="Q4_K_M",kv_quant="fp16",cap=262144):
    base=memory_breakdown(model,quant,0,kv_quant);capacity=float(hw.get("vram_gb",0))*0.90
    if capacity<=base["total_gb"]:capacity=max(0,float(hw.get("ram_gb",0))-5)
    per_1k=kv_cache_gb(model,1024,kv_quant)
    if per_1k<=0:return min(cap,int(model.get("context",cap)))
    remaining=max(0,capacity-base["total_gb"]);ctx=int((remaining/per_1k)*1024);return max(0,min(cap,int(model.get("context",cap)),(ctx//1024)*1024))
def heuristic_mid(model,status):
    p=max(0.5,float(model.get("active_params_b") or model.get("params_b",1)))
    if status=="GOOD":return max(3.0,70/(p**0.72))
    if status=="TIGHT":return max(2.0,52/(p**0.75))
    if status=="SLOW":return max(0.5,12/(p**0.82))
    return None
def load_benchmarks(path):
    try:
        d=json.loads(Path(path).read_text(encoding="utf8"));return d if isinstance(d,dict) and isinstance(d.get("entries"),list) else {"version":1,"entries":[]}
    except Exception:return {"version":1,"entries":[]}
def save_benchmarks(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2),encoding="utf8")
def hw_distance(hw,bhw):
    v=max(.1,float(hw.get("vram_gb",0)));r=max(1,float(hw.get("ram_gb",0)));return abs(float(bhw.get("vram_gb",0))-float(hw.get("vram_gb",0)))/v + .25*abs(float(bhw.get("ram_gb",0))-float(hw.get("ram_gb",0)))/r
def benchmark_factor(hw,benchmarks,quant=None):
    entries=[e for e in benchmarks.get("entries",[]) if e.get("heuristic_mid") and e.get("tps")]
    if quant:
        same=[e for e in entries if e.get("quant")==quant]
        if same:entries=same
    if not entries:return None
    ranked=sorted(entries,key=lambda e:hw_distance(hw,e.get("hardware",{})))[:5];num=den=0
    for e in ranked:
        ratio=float(e["tps"])/max(.01,float(e["heuristic_mid"]));w=1/(.15+hw_distance(hw,e.get("hardware",{})));num+=max(.25,min(4.0,ratio))*w;den+=w
    return max(.25,min(4.0,num/den)) if den else None
def speed_estimate(hw,model,status,calibration=None):
    mid=heuristic_mid(model,status)
    if mid is None:return None
    kind="heuristic";factor=1.0
    if calibration:factor=float(calibration);mid*=factor;kind="calibrated"
    return {"low":round(mid*.65,1),"high":round(mid*1.35,1),"mid":round(mid,1),"unit":"tok/s","kind":kind,"calibration_factor":round(factor,3)}
def recommend(rows):
    usable=[r for r in rows if r["status"]!="NO"];rank={"GOOD":3,"TIGHT":2,"SLOW":1};return sorted(usable,key=lambda r:(rank[r["status"]],float(r.get("active_params_b") or r["params_b"])),reverse=True)
def override(hw,a):
    h=json.loads(json.dumps(hw))
    if a.ram is not None:h["ram_gb"]=a.ram;h["ram_available_gb"]=a.ram
    if a.vram is not None:h["vram_gb"]=a.vram;h["gpus"]=[{"name":"override","vram_gb":a.vram}] if a.vram else []
    if a.disk is not None:h["disk_free_gb"]=a.disk
    return h
def match_models(models,terms):
    if not terms:return models
    qs=[x.strip().lower() for x in terms.split(",") if x.strip()];return [m for m in models if any(q in (m.get("name","")+" "+m.get("ollama","")).lower() for q in qs)]
def record_benchmark(path,hw,model,quant,context,kv_quant,tps,status):
    data=load_benchmarks(path);mid=heuristic_mid(model,status);entry={"recorded_at":datetime.now(timezone.utc).isoformat(),"model":model.get("name"),"params_b":model.get("params_b"),"active_params_b":model.get("active_params_b"),"quant":quant,"context":context,"kv_quant":kv_quant,"tps":float(tps),"status":status,"heuristic_mid":round(mid,3) if mid else None,"hardware":{"ram_gb":hw.get("ram_gb"),"vram_gb":hw.get("vram_gb"),"cpu":hw.get("cpu"),"gpus":hw.get("gpus",[])}};data.setdefault("entries",[]).append(entry);save_benchmarks(path,data);return entry

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--quant",choices=QUANTS,default="Q4_K_M");ap.add_argument("--context",type=int,default=8192);ap.add_argument("--kv-quant",choices=KV_BYTES,default="fp16");ap.add_argument("--ram",type=float);ap.add_argument("--vram",type=float);ap.add_argument("--disk",type=float);ap.add_argument("--model-size",type=float);ap.add_argument("--json");ap.add_argument("--top",type=int,default=5);ap.add_argument("--compare",help="comma-separated model name/tag filters");ap.add_argument("--benchmark-file",default="benchmarks.json");ap.add_argument("--record-benchmark",nargs=2,metavar=("MODEL","TOK_PER_SEC"));ap.add_argument("--list-benchmarks",action="store_true");a=ap.parse_args();hw=override(inspect(),a);bench=load_benchmarks(a.benchmark_file)
    if a.list_benchmarks:print(json.dumps(bench,indent=2));return
    models_path=Path(__file__).with_name("models.json")\n    if not models_path.exists():\n        from importlib.resources import files as resource_files\n        models_path=resource_files("canirunai_data").joinpath("models.json")\n    models=json.loads(models_path.read_text(encoding="utf8"))
    if a.model_size:models=[{"name":f"Custom {a.model_size:g}B","params_b":a.model_size,"use_case":"custom","ollama":"","context":a.context}]
    models=match_models(models,a.compare)
    if not models:raise SystemExit("No models matched --compare.")
    factor=benchmark_factor(hw,bench,a.quant)
    if a.record_benchmark:
        term,tps=a.record_benchmark;matches=match_models(models,term)
        if not matches:raise SystemExit(f"Model not found: {term}")
        m=matches[0];status,_,_=fit_model(hw,m,a.quant,a.context,a.kv_quant);entry=record_benchmark(a.benchmark_file,hw,m,a.quant,a.context,a.kv_quant,float(tps),status);print(json.dumps(entry,indent=2));return
    print(f"CanIRunAI {VERSION}\nRAM: {hw['ram_gb']} GB | VRAM: {hw['vram_gb']} GB | Free disk: {hw['disk_free_gb']} GB\nQuant: {a.quant} | Context: {a.context:,} | KV: {a.kv_quant}");print(f"Speed calibration: x{factor:.2f} from {len(bench.get('entries',[]))} local benchmark(s)\n" if factor else "Speed calibration: none (heuristic ranges)\n")
    rows=[]
    for m in models:
        status,mem,note=fit_model(hw,m,a.quant,a.context,a.kv_quant);max_ctx=max_context_estimate(hw,m,a.quant,a.kv_quant);speed=speed_estimate(hw,m,status,factor);r={**m,"quant":a.quant,"context_requested":a.context,"kv_quant":a.kv_quant,"memory":mem,"status":status,"note":note,"estimated_max_context":max_ctx,"speed_estimate":speed};rows.append(r);pull=(f" | ollama run {m['ollama']}" if m.get("ollama") and status!="NO" else "");speed_text=(f" | ~{speed['low']}-{speed['high']} tok/s {speed['kind']}" if speed else "");print(f"{status:5} {m['name']:<20} {mem['total_gb']:>5.1f}GB | max ctx ~{max_ctx//1024}K{speed_text}{pull}")
    best=recommend(rows)[:max(0,a.top)];print("\nBest fits:");[print(f"{i}. {r['name']} [{r['status']}] — {r['use_case']}") for i,r in enumerate(best,1)];report={"version":VERSION,"hardware":hw,"quantization":a.quant,"context":a.context,"kv_quant":a.kv_quant,"benchmark_file":a.benchmark_file,"calibration_factor":factor,"models":rows,"recommended":[r["name"] for r in best],"estimation_note":"Memory values are planning estimates. Speed ranges are calibrated only when local benchmark samples exist."}
    if a.json:Path(a.json).write_text(json.dumps(report,indent=2),encoding="utf8")
if __name__=="__main__":main()
