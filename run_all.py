"""Run the complete deterministic research workflow and verify the outputs."""
import os,sys,argparse,subprocess
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=min(12,os.cpu_count() or 1));ap.add_argument('--output',default='reproduced');args=ap.parse_args()
 if args.workers<1:ap.error('--workers must be positive')
 root=Path(__file__).resolve().parent;os.chdir(root);out=Path(args.output).resolve()
 if out==root/'results':ap.error('Choose a reproduction directory distinct from the reference results directory.')
 out.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
 for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
 env['PHYSICS_RESULTS_DIR']=str(out);env['PHYSICS_WORKERS']=str(args.workers)
 commands=[['-m','unittest','discover','-s','tests','-v'],['src/run_benchmark.py','--workers',str(args.workers),'--output',str(out/'main.jsonl')],['src/extra_experiments.py'],['src/export_examples.py'],['src/analyze.py','--root',str(out)],['verify_results.py','--generated',str(out)]]
 for command in commands:subprocess.run([sys.executable,*command],check=True,env=env)
 print('Complete reproduction and numerical verification passed.')

if __name__=='__main__':main()
