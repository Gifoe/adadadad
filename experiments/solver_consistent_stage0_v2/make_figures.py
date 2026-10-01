import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from runtime import ROOT,NAMES

def figures():
    out=ROOT/'artifacts'; dest=out/'figures'; dest.mkdir(exist_ok=True)
    rows=list(csv.DictReader((out/'stage0_results.csv').open()))
    colors=['#3b6fb6','#d77632','#29916e']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for number,metric,label in [(1,'accuracy','Accuracy'),(2,'harmful_flip_vs_8','Harmful flips vs budget 8')]:
        fig,axes=plt.subplots(1,4,figsize=(16,4),sharey=True)
        for ax,split in zip(axes,['iid','ood_6','ood_8','ood_12']):
            for name,color in zip(NAMES,colors):
                rs=sorted([r for r in rows if r['model']==name and r['split']==split and r['reasoning_depth']=='all' and r['solver'] in ['euler','loop'] and r['schedule']=='uniform'],key=lambda r:int(r['budget']))
                xs=[int(r['budget']) for r in rs]; ys=[float(r[metric]) for r in rs]
                ax.plot(xs,ys,label=name,color=color,marker='o',markersize=4)
                for x,y in zip(xs,ys):
                    if x in [4,8]: ax.scatter([x],[y],color=color,marker='s',s=50,zorder=4)
            ax.axvline(4,color='gray',alpha=.3,ls='--'); ax.axvline(8,color='gray',alpha=.3,ls='--')
            ax.set_title(split); ax.set_xlabel('NFE / loop count'); ax.set_xticks([3,4,5,7,8,12,16]); ax.grid(alpha=.2)
        axes[0].set_ylabel(label); axes[-1].legend(fontsize=8)
        fig.suptitle('Seed 0 exploratory; squares mark trained budgets 4 and 8'); fig.tight_layout()
        fig.savefig(dest/f'figure{number}.png',dpi=180); fig.savefig(dest/f'figure{number}.pdf'); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for i,split in enumerate(['iid','ood_6','ood_8','ood_12']):
        for sol,color in zip(['euler','heun','rk4'],colors):
            rs=sorted([r for r in rows if r['model']=='vector_field' and r['split']==split and r['reasoning_depth']=='all' and r['solver']==sol and r['schedule']=='uniform' and int(r['budget']) in [4,8,12,16]],key=lambda r:int(r['budget']))
            for ax,metric in [(axes[0,i],'accuracy'),(axes[1,i],'js_vs_8')]: ax.plot([int(r['budget']) for r in rs],[float(r[metric]) for r in rs],label=sol,color=color,marker='o'); ax.grid(alpha=.2)
        axes[0,i].set_title(split); axes[1,i].set_xlabel('Matched NFE'); axes[0,i].legend()
    axes[0,0].set_ylabel('Accuracy'); axes[1,0].set_ylabel('JS vs Euler NFE 8 (diagnostic)')
    fig.tight_layout(); fig.savefig(dest/'figure3.png',dpi=180); fig.savefig(dest/'figure3.pdf'); plt.close(fig)
    conv=json.loads((out/'vector_field/convergence.json').read_text()); fig,ax=plt.subplots(figsize=(6,4))
    for sol,color in zip(['euler','heun','rk4'],colors):
        rs=[r for r in conv['rows'] if r['solver']==sol and r['steps']<={'euler':64,'heun':32,'rk4':16}[sol] and r['relative_endpoint_error']>0]
        ax.plot(np.log([r['dt'] for r in rs]),np.log([r['relative_endpoint_error'] for r in rs]),marker='o',color=color,label=f'{sol}, fitted slope {conv["observed_slopes"][sol]:.2f}')
    ax.set_xlabel('log(dt)'); ax.set_ylabel('log(relative endpoint error)'); ax.set_title('FP32 numerical diagnostic; does not establish reasoning'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(dest/'figure4.png',dpi=180); fig.savefig(dest/'figure4.pdf'); plt.close(fig)

if __name__=='__main__': figures()
