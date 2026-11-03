#!/usr/bin/env python3
"""Render package tables from a compact published summary and the toy result."""
import argparse
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads(path.read_text())
def table(headers, rows):
    def cell(value): return str(value).replace('|', '&#124;')
    return ('\n| '+' | '.join(map(cell,headers))+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(cell,row))+' |\n' for row in rows))
def render(root, pages=None, check=False):
    outputs={}
    allowed=set(pages) if pages else None
    def fragment(page,key,value,prefix='generated'):
        if allowed is not None and page not in allowed: return
        path=root/page
        text=outputs.get(path,path.read_text())
        start,end=f'<!-- {prefix}:{key}:start -->',f'<!-- {prefix}:{key}:end -->'
        assert text.count(start)==text.count(end)==1, page+':'+key
        outputs[path]=re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _:start+'\n'+value.rstrip()+'\n'+end,text,flags=re.S)
    for block in read(root/'docs/assets/benchmark/documentation_summary.json')['blocks']:
        fragment(block['page'],block['key'],block['text'],block['prefix'])
    toy = root / "docs/assets/toy_result.json"
    if toy.exists():
        result = read(toy)
        pip_error = result["comparison"]["errors"]["pip"]["max_abs_error"]
        text = (
            f"The independent 500-variant toy example converged in {result['niter']} iterations "
            f"and returned {len(result['cs'])} credible sets. Its maximum absolute PIP difference "
            f"from the separately generated susieR 0.16.6 reference was {pip_error:.3g}.\n"
        )
        for page in ("docs/example.md", "docs/numerical_accuracy.md"):
            fragment(page, "toy", text)
        cs = table(
            ["Original component", "Synthetic variants", "Returned posterior mass", "Minimum |r|"],
            [
                [
                    c["component_id_0based"],
                    ", ".join(f"syn{j + 1:04d}" for j in c["members_0based"]),
                    f"{c['coverage']:.6f}",
                    f"{c['purity']['min_abs_corr']:.6f}",
                ]
                for c in result["cs"]
            ],
        )
        fragment("docs/example.md", "toy-cs", cs)

    stale = [str(p.relative_to(root)) for p, text in outputs.items() if p.read_text() != text]
    if check:
        assert not stale, "Stale generated blocks: " + ", ".join(stale)
    else:
        for path, text in outputs.items():
            path.write_text(text)
    return stale

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--pages')
    args=parser.parse_args()
    changed=render(ROOT,pages=args.pages.split(',') if args.pages else None,check=args.check)
    print('Current blocks:',len(changed),'stale' if args.check else 'updated')
if __name__=='__main__': main()
