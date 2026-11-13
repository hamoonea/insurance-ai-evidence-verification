import argparse
import json
from pathlib import Path
from .core import validate_dataset
from .run import execute
from .analyze import analyze, export_review


def main():
    p=argparse.ArgumentParser(description='Research-only insurance policy evaluation')
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate')
    run=sub.add_parser('run')
    run.add_argument('--backend',choices=['fixture','mlx'],default='mlx')
    run.add_argument('--model-path')
    run.add_argument('--out',required=True)
    run.add_argument('--limit',type=int)
    run.add_argument('--repeats',type=int,default=1)
    run.add_argument('--seed',type=int,default=20261113)
    run.add_argument('--temperature',type=float,default=0.0)
    run.add_argument('--max-tokens',type=int,default=2000)
    for cmd in ('analyze','review'):
        x=sub.add_parser(cmd);x.add_argument('--out',required=True)
    a=p.parse_args()
    try:
        if a.command=='validate':result=validate_dataset(a.root)
        elif a.command=='run':result={'output':str(execute(a))}
        elif a.command=='analyze':result=analyze(a.root,a.out)
        else:result=export_review(a.root,a.out)
        print(json.dumps(result,indent=2))
    except (ValueError,RuntimeError,OSError) as e:
        p.exit(1,str(e)+'\n')


if __name__=='__main__':
    main()
