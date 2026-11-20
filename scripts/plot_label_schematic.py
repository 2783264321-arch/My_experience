#!/usr/bin/env python3
"""Label-only time-frequency schematics for SpaceNet samples.

Draws each sample's annotated signal instances as boxes in a normalized
0-1 x 0-1 time-frequency frame (time right, frequency up), aspect 2:1
(freq:time), numbered boxes, family colors, no legend / no side list.
Boxes use translucent fills so overlaps stay visible.

Modes:
  static   render PNG schematics into a directory
  browser  build ONE self-contained interactive HTML (no server, no deps):
           sample navigation, hover an instance for its info, per-sample
           overlap pair list. Default splits: fit+val; --include-test adds
           the sealed test split (labels only, still no model involvement).

Examples:
  python scripts/plot_label_schematic.py --mode static --stratified \
      --out reports/label_schematics
  python scripts/plot_label_schematic.py --mode static \
      --samples train/2,train/25 --out reports/label_schematics
  python scripts/plot_label_schematic.py --mode browser \
      --out reports/label_schematic_browser.html
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

DATA_ROOT = Path("/root/autodl-fs/SpaceNet_Dataset/advanced")
MANIFEST = Path(
    "/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv"
)

NAMES = [
    "W20-QPSK", "W20-16QAM", "W20-64QAM", "W40-QPSK", "W40-16QAM", "W40-64QAM",
    "BLE LE1M", "BLE LE2M", "Zigbee", "LoRa 250k", "SRRC QPSK", "SRRC 16QAM",
    "AM", "FM",
]
# family color per class (matches the family analysis convention)
FAM_COLOR = (
    ["#1f77b4"] * 3 + ["#17becf"] * 3 + ["#2ca02c"] * 3
    + ["#ff7f0e"] + ["#d62728"] * 2 + ["#9467bd"] * 2
)
FAM_NAME = ["WiFi 20MHz", "WiFi 40MHz", "BLE/Zigbee", "LoRa", "SRRC", "AM/FM"]

ASPECT = 2.0  # freq : time


def load_samples(splits: list[str]) -> list[dict]:
    import csv

    rows = list(csv.DictReader(open(MANIFEST)))
    out = []
    for r in rows:
        if r["split_new"] not in splits:
            continue
        d = json.loads(
            (DATA_ROOT / r["split"] / f"{r['stem']}.json").read_text()
        )
        B = float(r["bandwidth_mhz"])
        T = float(r["duration_ms"])
        f_lo = float(r["f_lo_mhz"])
        sigs = []
        for s in d["signals"]:
            x0 = max(0.0, s["start_time"] / T)
            x1 = min(1.0, s["end_time"] / T)
            y0 = (s["start_frequency"] - f_lo) / B
            y1 = (s["end_frequency"] - f_lo) / B
            bw = s["end_frequency"] - s["start_frequency"]
            dur = s["end_time"] - s["start_time"]
            sigs.append(
                [int(s["class"]), round(x0, 5), round(x1, 5),
                 round(y0, 5), round(y1, 5), round(bw, 5), round(dur, 3)]
            )
        out.append(
            {"id": r["sample_id"], "split": r["split_new"], "B": B, "T": T,
             "f_lo": f_lo, "sigs": sigs}
        )
    return out


# ---------------------------------------------------------------- static ---

def render_static(sample: dict, path: Path, dpi: int = 120) -> None:
    fig = plt.figure(figsize=(5, 5 * ASPECT))
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.93])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.set_title(
        f"{sample['id']}  ·  {sample['B']:g} MHz x {sample['T']:g} ms"
        f"  ·  {len(sample['sigs'])} signal{'' if len(sample['sigs']) == 1 else 's'}"
        f"  ·  {sample['split']}",
        fontsize=9,
    )
    for i, s in enumerate(sample["sigs"], 1):
        cls, x0, x1, y0, y1 = s[0], s[1], s[2], s[3], s[4]
        col = FAM_COLOR[cls]
        h = max(y1 - y0, 0.002)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=col,
                               alpha=0.22, edgecolor=col, lw=1.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(i), fontsize=10,
                ha="center", va="center", color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def pick_stratified(samples: list[dict], limit: int) -> list[dict]:
    """Densest sample per observation bandwidth + global densest + sparsest."""
    picked: dict[str, dict] = {}
    for B in sorted({s["B"] for s in samples}):
        c = max((s for s in samples if s["B"] == B),
                key=lambda s: len(s["sigs"]))
        picked[c["id"]] = c
    dense = max(samples, key=lambda s: len(s["sigs"]))
    sparse = min(samples, key=lambda s: len(s["sigs"]))
    picked[dense["id"]] = dense
    picked[sparse["id"]] = sparse
    return list(picked.values())[:limit]


# --------------------------------------------------------------- browser ---

HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>SpaceNet label schematics</title>
<style>
:root { --line:#ddd; }
* { box-sizing:border-box; }
body { margin:0; font:13px/1.45 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; display:flex; }
#side { width:250px; min-width:250px; border-right:1px solid var(--line); padding:14px; height:100vh; overflow-y:auto; }
#main { flex:1; padding:14px 20px; height:100vh; overflow:auto; }
h1 { font-size:15px; margin:0 0 10px; }
label { display:block; font-size:11px; color:#666; margin:10px 0 3px; }
select, input[type=text] { width:100%; padding:4px 6px; font-size:12px; }
button { padding:4px 10px; font-size:12px; margin:2px 4px 2px 0; cursor:pointer; }
#legend { margin-top:14px; font-size:11px; }
#legend div, #ilist div { margin:3px 0; white-space:nowrap; }
#ilist div:hover { background:#f2f2f2; border-radius:3px; }
#legend i, #ilist i { display:inline-block; width:12px; height:12px; border:1.5px solid; margin-right:6px; vertical-align:-2px; }
#ilist { margin-top:14px; font-size:12px; }
#ilist label { margin:0 0 4px; }
#stats { margin-top:16px; font-size:11px; color:#888; }
#head .id { font-size:15px; font-weight:600; }
#head .meta { color:#666; font-size:12px; margin:2px 0 8px; }
#ovl { font-size:12px; margin-bottom:8px; color:#444; max-height:132px; overflow-y:auto; line-height:1.65; }
#ovl .t { font-weight:600; color:#c0392b; margin-bottom:1px; }
svg { background:#fff; max-height:calc(100vh - 120px); }
#tip { position:fixed; display:none; background:#fff; border:1px solid #bbb; box-shadow:0 2px 8px rgba(0,0,0,.15);
       padding:7px 10px; font-size:12px; pointer-events:none; z-index:10; white-space:nowrap; }
#tip .t1 { font-weight:600; } #tip .t2 { color:#555; }
text.num { font-size:11px; text-anchor:middle; paint-order:stroke; stroke:#fff; stroke-width:3px; fill:#111; pointer-events:none; }
</style></head>
<body>
<div id="side">
  <h1>SpaceNet label schematics</h1>
  <label>sample id (e.g. train/2)</label>
  <input id="goto" type="text" placeholder="train/2">
  <label>split</label>
  <select id="fsplit"><option value="">all</option></select>
  <label>observation bandwidth</label>
  <select id="fb"><option value="">all</option></select>
  <div style="margin-top:12px">
    <button id="prev">&#8592; prev</button>
    <button id="next">next &#8594;</button>
    <button id="rand">random</button>
  </div>
  <div id="pos" style="margin-top:8px;font-size:12px;color:#666"></div>
  <div id="legend"></div>
  <div id="ilist"></div>
  <div id="stats"></div>
  <div style="margin-top:14px;font-size:11px;color:#999">
    time &#8594; 0-1 &nbsp;·&nbsp; freq &#8593; 0-1 &nbsp;·&nbsp; aspect 2:1<br>
    hover a box for instance info
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

// controls
const splits = [...new Set(DATA.map(s=>s[1]))], bs = [...new Set(DATA.map(s=>s[2]))];
splits.forEach(v=>{const o=document.createElement('option');o.textContent=v;$('fsplit').append(o);});
bs.sort((a,b)=>a-b).forEach(v=>{const o=document.createElement('option');o.textContent=v;$('fb').append(o);});
$('fsplit').onchange = $('fb').onchange = () => {
  const sp=$('fsplit').value, b=$('fb').value;
  list = DATA.filter(s => (!sp||s[1]===sp) && (!b||String(s[2])===b));
  idx = 0; render();
};
$('prev').onclick = () => { idx=(idx-1+list.length)%list.length; render(); };
$('next').onclick = () => { idx=(idx+1)%list.length; render(); };
$('rand').onclick = () => { idx=Math.floor(Math.random()*list.length); render(); };
$('goto').onchange = () => {
  const p = list.findIndex(s=>s[0]===$('goto').value || s[0].endsWith('/'+$('goto').value));
  if(p>=0){ idx=p; render(); } else { alert('not in current filter'); }
};
document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='ArrowLeft') $('prev').click();
  if(e.key==='ArrowRight') $('next').click();
});
// sidebar legend
$('legend').innerHTML = FAMN.map((n,i)=>`<div><i style="border-color:${FAMC[REPR[i]]};background:${FAMC[REPR[i]]}33"></i>${n}</div>`).join('');
$('stats').innerHTML = `${DATA.length} samples · ${DATA.reduce((a,s)=>a+s[5].length,0)} instances in this file`;

function render(){
  const s = list[idx];
  $('pos').textContent = `${idx+1} / ${list.length}`;
  const [id, split, B, T, flo, sigs] = s;
  svg.innerHTML = '';
  boxEls = [];
  const mk = (tag,attrs)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const k in attrs) el.setAttribute(k,attrs[k]); svg.append(el); return el;};
  // frame = the 0-1 observation window
  mk('rect',{x:PL,y:PT,width:PW,height:PH,fill:'#fff',stroke:'#999','stroke-width':1});

  // overlap pairs
  const pairs=[];
  for(let i=0;i<sigs.length;i++) for(let j=i+1;j<sigs.length;j++){
    const a=sigs[i], b=sigs[j];
    const ix=Math.min(a[2],b[2])-Math.max(a[1],b[1]), iy=Math.min(a[4],b[4])-Math.max(a[3],b[3]);
    if(ix>0 && iy>0){
      const inter=ix*iy, aa=(a[2]-a[1])*(a[4]-a[3]), bb=(b[2]-b[1])*(b[4]-b[3]);
      pairs.push([i+1,j+1,inter/aa,inter/bb]);
    }
  }
  $('head').innerHTML = `<div class="id">${id}</div>`
    + `<div class="meta">${B} MHz × ${T} ms · ${sigs.length} signal${sigs.length>1?'s':''} · split ${split}</div>`;
  $('ovl').innerHTML = pairs.length
    ? `<div class="t">${pairs.length} overlapping pair${pairs.length>1?'s':''}</div>`
      + pairs.map(p=>`<div>#${p[0]} ∩ #${p[1]} · ${(p[2]*100).toFixed(1)}% of #${p[0]} · ${(p[3]*100).toFixed(1)}% of #${p[1]}</div>`).join('')
    : '<div>no overlapping instances</div>';
  // per-sample instance list (box id + class) in the sidebar
  $('ilist').innerHTML = '<label>instances in this sample</label>'
    + sigs.map((g,k)=>`<div><i style="border-color:${FAMC[g[0]]};background:${FAMC[g[0]]}33"></i>#${k+1} &nbsp;${NAMES[g[0]]}</div>`).join('');
  [...$('ilist').querySelectorAll('div')].forEach((row,k)=>{
    row.addEventListener('mouseenter',ev=>{
      show(ev,s,k+1,sigs[k]);
      const [r,t]=boxEls[k]||[];
      if(r){ r.setAttribute('stroke-width',3); svg.append(r); }
      if(t){ svg.append(t); t.style.fontSize='12px'; }
    });
    row.addEventListener('mousemove',moveTip);
    row.addEventListener('mouseleave',()=>{
      hide();
      const [r,t]=boxEls[k]||[];
      if(r) r.setAttribute('stroke-width',1.5);
      if(t) t.style.fontSize='';
    });
  });

  // boxes
  sigs.forEach((g,k)=>{
    const [cls,x0,x1,y0,y1,bw,dur] = g, col=FAMC[cls], n=k+1;
    let hPx=(y1-y0)*PH; let yPx=Y(y1);
    if(hPx<2.4){ yPx-=(2.4-hPx)/2; hPx=2.4; }
    const r=mk('rect',{x:X(x0),y:yPx,width:Math.max((x1-x0)*PW,2.4),height:hPx,
      fill:col,'fill-opacity':0.22,stroke:col,'stroke-width':1.5});
    const t=mk('text',{x:(X(x0)+X(x1))/2,y:yPx+hPx/2+4,class:'num'});
    t.textContent=n;
    r.addEventListener('mouseenter',ev=>show(ev,s,n,g));
    r.addEventListener('mousemove',ev=>moveTip(ev));
    r.addEventListener('mouseleave',hide);
    t.style.pointerEvents='none';
    boxEls.push([r,t]);
  });
}

function show(ev,s,n,g){
  const [id,split,B,T,flo]=s, [cls,x0,x1,y0,y1,bw,dur]=g;
  const fc=flo+(y0+y1)/2*B, rel=bw/B*100;
  tip.innerHTML =
    `<div class="t1">#${n} · ${NAMES[cls]}</div>`+
    `<div class="t2">${FAMN[famOf(cls)]} · bw ${bw<1?(bw*1000).toPrecision(3)+' kHz':bw.toPrecision(3)+' MHz'} · dur ${dur.toPrecision(3)} ms</div>`+
    `<div class="t2">rel_bw ${rel<1?rel.toPrecision(2):rel.toFixed(1)}% · t ${x0.toFixed(2)}-${x1.toFixed(2)} · f ${y0.toFixed(3)}-${y1.toFixed(3)}</div>`+
    `<div class="t2">center ${fc.toFixed(2)} MHz · abs t ${(x0*T).toFixed(1)}-${(x1*T).toFixed(1)} ms</div>`;
  tip.style.display='block'; moveTip(ev);
}
function moveTip(ev){
  const w=tip.offsetWidth,hh=tip.offsetHeight;
  let x=ev.clientX+14,y=ev.clientY+14;
  if(x+w>innerWidth-8)x=ev.clientX-w-10; if(y+hh>innerHeight-8)y=ev.clientY-hh-10;
  tip.style.left=x+'px'; tip.style.top=y+'px';
}
function hide(){ tip.style.display='none'; }
render();
</script></body></html>
"""


def build_browser(samples: list[dict], out: Path) -> None:
    data = [
        [s["id"], s["split"], s["B"], s["T"], s["f_lo"], s["sigs"]]
        for s in samples
    ]
    payload = json.dumps(data, separators=(",", ":"))
    html = (
        HTML.replace("__DATA__", payload)
        .replace("__NAMES__", json.dumps(NAMES))
        .replace("__FAMC__", json.dumps(FAM_COLOR))
        .replace("__FAMN__", json.dumps(FAM_NAME, ensure_ascii=False))
    )
    out.write_text(html, encoding="utf-8")
    n_inst = sum(len(s["sigs"]) for s in samples)
    print(
        f"browser: {out}  {out.stat().st_size/1e6:.2f} MB  "
        f"{len(samples)} samples  {n_inst} instances"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["static", "browser"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="fit,val",
                    help="comma list of split_new values (default fit,val)")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--samples", default="",
                    help="static mode: comma list of sample ids")
    ap.add_argument("--stratified", type=int, default=0, metavar="N",
                    help="static mode: densest-per-B + extremes, cap N")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s]
    if args.include_test and "test" not in splits:
        splits.append("test")
    t0 = time.perf_counter()
    samples = load_samples(splits)
    print(f"loaded {len(samples)} samples "
          f"({sum(len(s['sigs']) for s in samples)} instances) "
          f"in {time.perf_counter()-t0:.1f}s from splits {splits}")

    out = Path(args.out)
    if args.mode == "browser":
        build_browser(samples, out)
        return

    out.mkdir(parents=True, exist_ok=True)
    if args.samples:
        want = {x.strip() for x in args.samples.split(",")}
        picked = [s for s in samples if s["id"] in want]
        missing = want - {s["id"] for s in picked}
        if missing:
            print(f"!! not found / wrong split: {sorted(missing)}")
    else:
        picked = pick_stratified(samples, args.stratified or 10)
    for s in picked:
        p = out / f"{s['id'].replace('/', '_')}.png"
        render_static(s, p, dpi=args.dpi)
        print(f"{p}  {p.stat().st_size/1024:.1f} KB  ({len(s['sigs'])} sig)")


if __name__ == "__main__":
    main()
