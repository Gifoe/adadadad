"""Render the seven required static diagnostic figure groups from saved CSVs."""
import csv, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';FIG=OUT/'figures';FIG.mkdir(exist_ok=True,parents=True)
COLORS={1:'#1769aa',3:'#e38924',6:'#8a3bb7'}

def load(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def filt(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
def seq(rows,x,y):
    rows=sorted(rows,key=lambda r:float(r[x]));return [float(r[x]) for r in rows],[float(r[y]) for r in rows]
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close(fig)

def main():
    s=load('stage1_epoch_metrics.csv');g=load('latent_geometry.csv')
    a=load('latent_geometry_alignment.csv');c=load('decoder_training_curves.csv')
    r=load('decoder_training_results.csv');p=load('paired_time_blocks.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['NP','PJM']):
        for domain,label,style in [('validation_chronological','Validation','-o'),
                                   ('train_probe_chronological','Train probe','-s')]:
            x,y=seq(filt(s,dataset=ds,domain=domain),'stage1_epoch','loss');ax.plot(x,y,style,label=label)
        ax.set(title=ds,xlabel='Stage-1 epoch',ylabel='Latent objective');ax.legend();ax.grid(alpha=.25)
    save(fig,'01_stage1_train_validation_latent.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['NP','PJM']):
        rows=filt(s,dataset=ds,domain='validation_chronological')
        for key,scale,label in [('mse_loss',10,'10 × latent L1'),('cosine_loss',15,'15 × cosine'),
                                ('sigreg_loss',.1,'0.1 × regularizer')]:
            x,y=seq(rows,'stage1_epoch',key);ax.plot(x,[scale*z for z in y],'-o',label=label)
        ax.set(title=ds,xlabel='Stage-1 epoch',ylabel='Weighted objective component');ax.legend();ax.grid(alpha=.25)
    save(fig,'02_latent_loss_components.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,ds in enumerate(['NP','PJM']):
        for state in ['h_target','h_pred']:
            rows=filt(g,dataset=ds,scope='val',state=state)
            x,y=seq(rows,'stage1_epoch','mean_token_l2');axes[0,col].plot(x,y,'-o',label=state)
            x,y=seq(rows,'stage1_epoch','effective_rank_singular_entropy');axes[1,col].plot(x,y,'-o',label=state)
        axes[0,col].set(title=ds,ylabel='Mean token L2',xlabel='Stage-1 epoch')
        axes[1,col].set(ylabel='Effective rank',xlabel='Stage-1 epoch')
        for row in axes[:,col]:row.legend();row.grid(alpha=.25)
    save(fig,'03_latent_norm_effective_rank.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,ds in enumerate(['NP','PJM']):
        for state in ['h_hist','h_target','h_pred']:
            rows=filt(a,dataset=ds,state=state)
            x,y=seq(rows,'stage1_epoch','centered_linear_cka_val');axes[0,col].plot(x,y,'-o',label=state)
            x,y=seq(rows,'stage1_epoch','registered_val_mse_over_epoch1_val_variance');axes[1,col].plot(x,y,'-o',label=state)
        axes[0,col].set(title=ds,ylabel='Centered linear CKA vs epoch 1',xlabel='Stage-1 epoch')
        axes[1,col].set(ylabel='Train-fit Procrustes Val MSE / epoch-1 variance',xlabel='Stage-1 epoch')
        for row in axes[:,col]:row.legend();row.grid(alpha=.25)
    save(fig,'04_cka_procrustes.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,ds in enumerate(['NP','PJM']):
        for ep in [1,3,6]:
            rows=filt(c,dataset=ds,stage1_epoch=ep)
            x,y=seq(rows,'decoder_epoch','val_select_selection_mae_batch_mean');axes[0,col].plot(x,y,color=COLORS[ep],label=f'Stage1 E{ep}')
            rows=[x for x in rows if x['decoder_epoch']!='0']
            x,y=seq(rows,'decoder_epoch','train_pred_mae');axes[1,col].plot(x,y,color=COLORS[ep],label=f'Stage1 E{ep}')
        axes[0,col].set(title=ds,ylabel='V_select forecast MAE',xlabel='Decoder epoch')
        axes[1,col].set(ylabel='Online Train forecast MAE',xlabel='Decoder epoch')
        for row in axes[:,col]:row.legend();row.grid(alpha=.25)
    save(fig,'05_decoder_training_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['NP','PJM']):
        for ep in [1,3,6]:
            latent=filt(s,dataset=ds,domain='validation_chronological',stage1_epoch=ep)[0]
            result=filt(r,dataset=ds,stage1_epoch=ep,decoder_checkpoint='best')[0]
            x=float(latent['loss']);y=float(result['val_diagnostic_forecast_mae'])
            ax.scatter([x],[y],color=COLORS[ep],s=75)
            ax.annotate(f'E{ep}',(x,y),xytext=(4,5),textcoords='offset points')
        ax.set(title=ds,xlabel='Fixed full Validation latent objective',
               ylabel='V_diagnostic forecast MAE');ax.grid(alpha=.25)
    save(fig,'06_latent_loss_vs_downstream_mae.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['NP','PJM']):
        for ep in [3,6]:
            rows=filt(p,dataset=ds,stage1_epoch=ep)
            x=[(int(z['origin_first'])+int(z['origin_last']))/2 for z in rows]
            y=[float(z['delta_candidate_minus_epoch1']) for z in rows]
            ax.plot(x,y,'-o',color=COLORS[ep],label=f'E{ep} − E1')
        ax.axhline(0,color='black',lw=.8)
        ax.set(title=ds,xlabel='V_diagnostic origin center (raw hour index)',
               ylabel='Forecast MAE difference by 360-origin block');ax.legend();ax.grid(alpha=.25)
    save(fig,'07_paired_time_blocks.png')
    assert len(list(FIG.glob('*.png')))==7
    print('FIGURES_COMPLETE',7,flush=True)

if __name__=='__main__':main()
