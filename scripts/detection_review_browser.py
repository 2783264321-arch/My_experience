#!/usr/bin/env python3
"""Detection-review browser: localization misses of the coarse+fine baseline.

Loads the stage-5 val predictions, marks every GT instance as found / missed
with the evaluator's class-agnostic rule (greedy one-to-one, IoU>=0.5,
top-100 predictions by score), and builds ONE self-contained HTML page:

  - found GT: faint family color; missed GT: red, drawn on top
  - sidebar instance list with check / cross per instance
  - prev-miss / next-miss jump between samples that contain a miss
  - optional prediction overlay (all classes, thin gray)
  - global miss breakdown by family

Usage:
  PYTHONPATH=. python scripts/detection_review_browser.py \
      --out reports/detection_review_browser.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.class_group_metrics import load_predictions
from src.signals.metrics import _match_single

NAMES = [
    "W20-QPSK", "W20-16QAM", "W20-64QAM", "W40-QPSK", "W40-16QAM", "W40-64QAM",
    "BLE LE1M", "BLE LE2M", "Zigbee", "LoRa 250k", "SRRC QPSK", "SRRC 16QAM",
    "AM", "FM",
]
FAM_COLOR = (
    ["#1f77b4"] * 3 + ["#17becf"] * 3 + ["#2ca02c"] * 3
    + ["#ff7f0e"] + ["#d62728"] * 2 + ["#9467bd"] * 2
)
FAM_NAME = ["WiFi 20MHz", "WiFi 40MHz", "BLE/Zigbee", "LoRa", "SRRC", "AM/FM"]


def fam_of(c: int) -> str:
    if c < 3:
        return "WiFi20 0-2"
    if c < 6:
        return "WiFi40 3-5"
    if c < 9:
        return "BLE/Zig 6-8"
    if c == 9:
        return "LoRa 9"
    if c < 12:
        return "SRRC 10-11"
    return "AM/FM 12-13"


def build(val_dir: Path):
    recs, gts, preds = load_predictions(val_dir)
    data = []
    tot = Counter()
    miss = Counter()
    miss_samples = 0
    by_rel = Counter()
    by_rel_miss = Counter()
    by_dur = Counter()
    by_dur_miss = Counter()
    for r, g, p in zip(recs, gts, preds):
        B = float(r.B_mhz)
        T = float(r.T_ms)
        f_lo = float(r.f_lo)
        matches, _ = _match_single(g, p, 0.5, class_aware=False, max_dets=100)
        matched = {gi for _, gi in matches}
        gg = []
        for i in range(len(g.boxes)):
            cls = int(g.labels[i])
            x1, y1, x2, y2 = (float(v) for v in g.boxes[i])
            found = 1 if i in matched else 0
            gg.append([cls, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4), found])
            fam = fam_of(cls)
            tot[fam] += 1
            bw = (y2 - y1) * B
            dur = (x2 - x1) * T
            rel = bw / B * 100 if B else 0.0
            rb = ("<=0.5%" if rel <= 0.5 else "0.5-2%" if rel <= 2 else
                  "2-5%" if rel <= 5 else "5-20%" if rel <= 20 else ">20%")
            db = ("<=5ms" if dur <= 5 else "5-15ms" if dur <= 15 else
                  "15-30ms" if dur <= 30 else "30-60ms" if dur <= 60 else ">60ms")
            by_rel[rb] += 1
            by_dur[db] += 1
            if not found:
                miss[fam] += 1
                by_rel_miss[rb] += 1
                by_dur_miss[db] += 1
        if any(not f for *_, f in gg):
            miss_samples += 1
        pp = [
            [int(p.labels[i]), round(float(p.boxes[i, 0]), 4),
             round(float(p.boxes[i, 1]), 4), round(float(p.boxes[i, 2]), 4),
             round(float(p.boxes[i, 3]), 4), round(float(p.scores[i]), 3)]
            for i in range(len(p.boxes))
        ]
        data.append([r.sample_id, B, T, f_lo, gg, pp])

    n_gt = sum(tot.values())
    n_miss = sum(miss.values())
    print(f"samples {len(recs)}  gt {n_gt}  pred {sum(len(p.boxes) for p in preds)}")
    print(f"missed {n_miss}/{n_gt} = {n_miss/n_gt*100:.2f}%   "
          f"samples with >=1 miss: {miss_samples}/{len(recs)}")
    print("\nby family:")
    for k in tot:
        print(f"  {k:<15} {miss[k]:>4} / {tot[k]:>5}  ({miss[k]/tot[k]*100:5.1f}%)")
    print("\nby rel_bw:")
    for k in ["<=0.5%", "0.5-2%", "2-5%", "5-20%", ">20%"]:
        if by_rel[k]:
            print(f"  {k:<8} {by_rel_miss[k]:>4} / {by_rel[k]:>5}  ({by_rel_miss[k]/by_rel[k]*100:5.1f}%)")
    print("\nby duration:")
    for k in ["<=5ms", "5-15ms", "15-30ms", "30-60ms", ">60ms"]:
        if by_dur[k]:
            print(f"  {k:<8} {by_dur_miss[k]:>4} / {by_dur[k]:>5}  ({by_dur_miss[k]/by_dur[k]*100:5.1f}%)")
    return data


HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Detection review — localization misses</title>
<style>
:root { --line:#ddd; }
* { box-sizing:border-box; }
body { margin:0; font:13px/1.45 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; display:flex; }
#side { width:270px; min-width:270px; border-right:1px solid var(--line); padding:14px; height:100vh; overflow-y:auto; }
#main { flex:1; padding:14px 20px; height:100vh; overflow:auto; }
h1 { font-size:14px; margin:0 0 4px; }
h1 span { font-weight:400; color:#888; font-size:11px; }
.note { font-size:11px; color:#888; margin-bottom:10px; line-height:1.4; }
label { display:block; font-size:11px; color:#666; margin:10px 0 3px; }
select, input[type=text] { width:100%; padding:4px 6px; font-size:12px; }
button { padding:4px 9px; font-size:12px; margin:2px 4px 2px 0; cursor:pointer; }
button.miss { border:1px solid #d62728; color:#a11; }
#legend { margin-top:12px; font-size:11px; }
#legend div, #ilist div { margin:3px 0; white-space:nowrap; }
#legend i, #ilist i { display:inline-block; width:12px; height:12px; border:1.5px solid; margin-right:6px; vertical-align:-2px; }
#ilist { margin-top:12px; font-size:12px; }
#ilist div.miss { color:#b00; font-weight:600; background:#fff2f2; border-radius:3px; }
#ilist div:hover { background:#f2f2f2; }
#ilist div.miss:hover { background:#ffe6e6; }
#famfilter { margin-top:12px; font-size:11px; }
#famfilter div { margin:3px 0; white-space:nowrap; }
#famfilter i { display:inline-block; width:10px; height:10px; border:1.5px solid; margin:0 5px 0 3px; vertical-align:-1px; }
#famfilter .cnt { color:#999; margin-left:4px; }
#famfilter input { vertical-align:-1px; margin:0; }
#stats { margin-top:10px; font-size:11px; color:#555; }
#head .id { font-size:15px; font-weight:600; }
#head .meta { color:#666; font-size:12px; margin:2px 0 2px; }
#head .miss { color:#c0392b; font-weight:600; font-size:12px; margin-bottom:8px; }
#ovl { font-size:11px; margin-bottom:8px; color:#777; max-height:96px; overflow-y:auto; line-height:1.55; }
#ovl .t { font-weight:600; color:#999; }
svg { background:#fff; max-height:calc(100vh - 190px); }
#tip { position:fixed; display:none; background:#fff; border:1px solid #bbb; box-shadow:0 2px 8px rgba(0,0,0,.15);
       padding:7px 10px; font-size:12px; pointer-events:none; z-index:10; white-space:nowrap; }
#tip .t1 { font-weight:600; } #tip .t2 { color:#555; }
text.num { font-size:11px; text-anchor:middle; paint-order:stroke; stroke:#fff; stroke-width:3px; fill:#111; pointer-events:none; }
</style></head>
<body>
<div id="side">
  <h1>Detection review <span>· localization misses</span></h1>
  <div class="note">class-agnostic match, IoU≥0.5, greedy by score
  (same rule as the evaluator's localization recall).
  <b style="color:#b00">red = missed</b>, faint = found.</div>
  <label>sample id (e.g. train/2)</label>
  <input id="goto" type="text" placeholder="train/2">
  <label>observation bandwidth</label>
  <select id="fb"><option value="">all</option></select>
  <label>filters</label>
  <div style="font-size:12px"><input type="checkbox" id="fmiss"> samples with misses only</div>
  <div style="font-size:12px;margin-top:2px"><input type="checkbox" id="fpred"> show predictions (all classes)</div>
  <div style="margin-top:10px">
    <button id="prev">&#8592; prev</button>
    <button id="next">next &#8594;</button>
    <button id="rand">random</button>
  </div>
  <div style="margin-top:4px">
    <button id="prevmiss" class="miss">&#8592; prev miss</button>
    <button id="nextmiss" class="miss">next miss &#8594;</button>
  </div>
  <div id="pos" style="margin-top:8px;font-size:12px;color:#666"></div>
  <div id="legend"></div>
  <div id="famfilter"></div>
  <div id="ilist"></div>
  <div id="stats"></div>
  <div style="margin-top:14px;font-size:11px;color:#999">
    hover a box or a list row for details<br>
    ← / → prev-next sample
  </div>
</div>
<div id="main">
  <div id="head"></div>
  <div id="ovl"></div>
  <svg id="plot" viewBox="0 0 520 1040" width="480"></svg>
</div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const NAMES = __NAMES__;
const FAMC = __FAMC__, FAMN = __FAMN__, REPR = [0,3,6,9,10,12];
const famOf = c => c<3?0 : c<6?1 : c<9?2 : c===9?3 : c<12?4 : 5;
const W=520,H=1040,PL=6,PR=6,PT=12,PB=12,PW=W-PL-PR,PH=H-PT-PB;
const X = t => PL + t*PW, Y = f => PT + (1-f)*PH;

let list = DATA.slice(), idx = 0, boxEls = [];
const $ = id => document.getElementById(id);
const svg = $('plot'), tip = $('tip');

// B filter options
[...new Set(DATA.map(s=>s[1]))].sort((a,b)=>a-b).forEach(v=>{
  const o=document.createElement('option'); o.textContent=v; o.value=v; $('fb').append(o);
});

function selectedFams(){
  return new Set([...document.querySelectorAll('#famfilter input:checked')].map(e=>+e.dataset.fam));
}
function applyFilter(){
  const b = $('fb').value, only = $('fmiss').checked;
  const sel = selectedFams();
  list = DATA.filter(s => {
    const gs = s[4];
    if(!gs.some(g=>sel.has(famOf(g[0])))) return false;
    if(b && String(s[1])!==b) return false;
    if(only) return gs.some(g=>sel.has(famOf(g[0])) && !g[5]);
    return true;
  });
  idx = 0; render();
}
$('fb').onchange = applyFilter;
$('fmiss').onchange = applyFilter;
$('fpred').onchange = render;
$('prev').onclick = () => { if(list.length){ idx=(idx-1+list.length)%list.length; render(); } };
$('next').onclick = () => { if(list.length){ idx=(idx+1)%list.length; render(); } };
$('rand').onclick = () => { if(list.length){ idx=Math.floor(Math.random()*list.length); render(); } };
function jumpMiss(dir){
  if(!list.length) return;
  const sel = selectedFams();
  for(let step=1; step<=list.length; step++){
    const k=(idx+dir*step+list.length*2)%list.length;
    if(list[k][4].some(g=>sel.has(famOf(g[0])) && !g[5])){ idx=k; render(); return; }
  }
  alert('no sample with selected-family misses in current filter');
}
$('nextmiss').onclick = () => jumpMiss(1);
$('prevmiss').onclick = () => jumpMiss(-1);
$('goto').onchange = () => {
  const p = list.findIndex(s=>s[0]===$('goto').value || s[0].endsWith('/'+$('goto').value));
  if(p>=0){ idx=p; render(); } else { alert('not in current filter'); }
};
document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='ArrowLeft') $('prev').click();
  if(e.key==='ArrowRight') $('next').click();
});

$('legend').innerHTML =
  `<div><i style="border-color:#d62728;background:#d6272833"></i>missed GT</div>`
+ `<div><i style="border-color:#999;background:#99999922"></i>found GT (family fill)</div>`
+ `<div><i style="border-color:#000;background:none;border-style:dashed"></i>prediction (toggle)</div>`;

// family stats + filter checkboxes
const FAMSTATS = FAMN.map(()=>({m:0,t:0}));
DATA.forEach(s=>s[4].forEach(g=>{
  const f=famOf(g[0]); FAMSTATS[f].t++; if(!g[5]) FAMSTATS[f].m++;
}));
$('famfilter').innerHTML = '<label>families (missed / total)</label>' + FAMN.map((n,f)=>
  `<div><input type="checkbox" data-fam="${f}" checked>`
+ `<i style="border-color:${FAMC[REPR[f]]};background:${FAMC[REPR[f]]}33"></i>${n}`
+ `<span class="cnt">${FAMSTATS[f].m}/${FAMSTATS[f].t}</span></div>`
).join('');
$('famfilter').addEventListener('change', applyFilter);
(() => {
  const tm = FAMSTATS.reduce((a,v)=>a+v.m,0);
  const tt = FAMSTATS.reduce((a,v)=>a+v.t,0);
  $('stats').innerHTML = `<b>${tm} / ${tt} missed (${(tm/tt*100).toFixed(2)}%)</b> · ${DATA.length} samples`;
})();

function mk(tag,attrs){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const k in attrs) el.setAttribute(k,attrs[k]); svg.append(el); return el;}

function showGT(ev,s,k,g){
  const [id,B,T,flo]=s, [cls,x1,y1,x2,y2,found]=g;
  const fc=flo+(y1+y2)/2*B, rel=(y2-y1)*100;
  tip.innerHTML =
    `<div class="t1">#${k+1} · ${NAMES[cls]} — ${found?'found':'MISSED'}</div>`
  + `<div class="t2">${FAMN[famOf(cls)]} · bw ${((y2-y1)*B).toPrecision(3)} MHz · dur ${((x2-x1)*T).toPrecision(3)} ms</div>`
  + `<div class="t2">rel_bw ${rel.toPrecision(3)}% · t ${x1.toFixed(2)}-${x2.toFixed(2)} · f ${y1.toFixed(3)}-${y2.toFixed(3)}</div>`
  + `<div class="t2">center ${fc.toFixed(2)} MHz · abs t ${(x1*T).toFixed(1)}-${(x2*T).toFixed(1)} ms</div>`;
  tip.style.display='block'; moveTip(ev);
}
function showPred(ev,s,i){
  const [id,B,T,flo]=s, [cls,x1,y1,x2,y2,score]=s[5][i];
  tip.innerHTML =
    `<div class="t1">pred · ${NAMES[cls]}</div>`
  + `<div class="t2">score ${score.toFixed(3)} · bw ${((y2-y1)*B).toPrecision(3)} MHz · dur ${((x2-x1)*T).toPrecision(3)} ms</div>`
  + `<div class="t2">t ${x1.toFixed(2)}-${x2.toFixed(2)} · f ${y1.toFixed(3)}-${y2.toFixed(3)}</div>`;
  tip.style.display='block'; moveTip(ev);
}
function moveTip(ev){
  const w=tip.offsetWidth,hh=tip.offsetHeight;
  let x=ev.clientX+14,y=ev.clientY+14;
  if(x+w>innerWidth-8)x=ev.clientX-w-10; if(y+hh>innerHeight-8)y=ev.clientY-hh-10;
  tip.style.left=x+'px'; tip.style.top=y+'px';
}
function hide(){ tip.style.display='none'; }

function render(){
  if(!list.length){ svg.innerHTML=''; $('head').innerHTML='<div class="id">no sample in filter</div>'; $('ilist').innerHTML=''; $('ovl').innerHTML=''; return; }
  const s = list[idx];
  const [id,B,T,flo,gts,preds] = s;
  const sel = selectedFams();
  const shown = gts.map((g,k)=>({g,k})).filter(o=>sel.has(famOf(o.g[0])));
  const nm = shown.filter(o=>!o.g[5]).length;
  $('pos').textContent = `${idx+1} / ${list.length}`;
  svg.innerHTML = ''; boxEls = [];

  mk('rect',{x:PL,y:PT,width:PW,height:PH,fill:'#fff',stroke:'#999','stroke-width':1});

  // GT overlap pairs (selected families only)
  const pairs=[];
  for(let i=0;i<shown.length;i++) for(let j=i+1;j<shown.length;j++){
    const a=shown[i].g, b=shown[j].g;
    const ix=Math.min(a[3],b[3])-Math.max(a[1],b[1]), iy=Math.min(a[4],b[4])-Math.max(a[2],b[2]);
    if(ix>0 && iy>0){
      const inter=ix*iy, aa=(a[3]-a[1])*(a[4]-a[2]), bb=(b[3]-b[1])*(b[4]-b[2]);
      pairs.push([shown[i].k+1,shown[j].k+1,inter/aa,inter/bb]);
    }
  }
  $('head').innerHTML = `<div class="id">${id}</div>`
    + `<div class="meta">${B} MHz × ${T} ms · GT ${shown.length} signal${shown.length>1?'s':''}${shown.length!==gts.length?` (of ${gts.length})`:''}</div>`
    + `<div class="miss">${nm ? `${nm} missed (localized ${shown.length-nm}/${shown.length} = ${((shown.length-nm)/shown.length*100).toFixed(1)}%)` : 'all shown GT localized'}</div>`;
  $('ovl').innerHTML = pairs.length
    ? `<div class="t">${pairs.length} overlapping pair${pairs.length>1?'s':''}</div>`
      + pairs.map(p=>`<div>#${p[0]} ∩ #${p[1]} · ${(p[2]*100).toFixed(1)}% of #${p[0]} · ${(p[3]*100).toFixed(1)}% of #${p[1]}</div>`).join('')
    : '';

  // sidebar instance list (selected families only)
  $('ilist').innerHTML = '<label>instances (✓ found · ✗ missed)</label>' + shown.map(o=>
    `<div class="${o.g[5]?'':'miss'}"><i style="border-color:${o.g[5]?FAMC[o.g[0]]:'#d62728'};background:${o.g[5]?FAMC[o.g[0]]+'33':'#d6272833'}"></i>`
    + `#${o.k+1} ${NAMES[o.g[0]]} <b style="color:${o.g[5]?'#2a8a2a':'#b00'}">${o.g[5]?'✓':'✗'}</b></div>`
  ).join('');
  [...$('ilist').querySelectorAll('div')].forEach((row,j)=>{
    const o = shown[j], k = o.k;
    row.addEventListener('mouseenter',ev=>{
      showGT(ev,s,k,o.g);
      const [r,t]=boxEls[k]||[];
      if(r){ r.setAttribute('stroke-width', r.getAttribute('data-bw')==='miss'?'4':'3'); svg.append(r); }
      if(t){ svg.append(t); t.style.fontSize='12px'; }
    });
    row.addEventListener('mousemove',moveTip);
    row.addEventListener('mouseleave',()=>{
      hide();
      const [r,t]=boxEls[k]||[];
      if(r) r.setAttribute('stroke-width', r.getAttribute('data-bw')==='miss'?'2.6':'1.1');
      if(t) t.style.fontSize='';
    });
  });

  // predictions under GT
  if($('fpred').checked && preds.length){
    preds.forEach((g,i)=>{
      const [cls,x1,y1,x2,y2,score]=g;
      const op = Math.min(0.9, 0.45 + 0.45*score);
      const r = mk('rect',{x:X(x1),y:Y(y2),width:Math.max((x2-x1)*PW,1.5),height:Math.max((y2-y1)*PH,1.5),
        fill:'none',stroke:'#000', 'stroke-width':0.9, 'stroke-dasharray':'3 2', opacity:op});
      r.addEventListener('mouseenter',ev=>showPred(ev,s,i));
      r.addEventListener('mousemove',moveTip);
      r.addEventListener('mouseleave',hide);
    });
  }

  // GT boxes (selected families): found first (faint), missed last (red, on top)
  const order = shown.filter(o=>o.g[5]).concat(shown.filter(o=>!o.g[5]));
  order.forEach(o=>{
    const k=o.k, g=o.g, [cls,x1,y1,x2,y2,found]=g;
    const col = found ? FAMC[cls] : '#d62728';
    const fillOp = found ? 0.10 : 0.24;
    const lw = found ? 1.1 : 2.6;
    let hPx=(y2-y1)*PH, yPx=Y(y2);
    if(hPx<2.6){ yPx-=(2.6-hPx)/2; hPx=2.6; }
    const r=mk('rect',{x:X(x1),y:yPx,width:Math.max((x2-x1)*PW,2.6),height:hPx,
      fill:col,'fill-opacity':fillOp,stroke:col,'stroke-width':lw});
    r.setAttribute('data-bw', found?'found':'miss');
    const t=mk('text',{x:(X(x1)+X(x2))/2,y:yPx+hPx/2+4,class:'num'});
    t.textContent=k+1;
    if(!found) t.style.fill='#b00';
    t.style.pointerEvents='none';
    r.addEventListener('mouseenter',ev=>showGT(ev,s,k,g));
    r.addEventListener('mousemove',moveTip);
    r.addEventListener('mouseleave',hide);
    boxEls[k]=[r,t];
  });
}
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="runs/yolo11s_c1_directreg_2048x512/stage5_val")
    ap.add_argument("--out", default="reports/detection_review_browser.html")
    args = ap.parse_args()

    data = build(ROOT / args.val_dir)
    payload = json.dumps(data, separators=(",", ":"))
    html = (
        HTML.replace("__DATA__", payload)
        .replace("__NAMES__", json.dumps(NAMES))
        .replace("__FAMC__", json.dumps(FAM_COLOR))
        .replace("__FAMN__", json.dumps(FAM_NAME, ensure_ascii=False))
    )
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out}  {out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
