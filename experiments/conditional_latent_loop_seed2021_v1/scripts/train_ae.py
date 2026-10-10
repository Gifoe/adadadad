from common import *
import time
from torch.utils.data import DataLoader,TensorDataset

def train(name):
    out=ROOT/'ae'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    seed_all();df,y,c,tr,tv,sy,sc=load_data(name);ae=official_ae().cuda()
    opt=torch.optim.Adam(ae.parameters(),lr=.0005)
    loader=DataLoader(TensorDataset(y[:tr]),batch_size=512,shuffle=True)
    best=float('inf');wait=0;history=[];start=time.perf_counter()
    for epoch in range(500):
        ae.train();total=0
        for (x,) in loader:
            x=x.cuda();opt.zero_grad();loss=(ae(x)-x).abs().mean();loss.backward();opt.step();total+=float(loss)*len(x)
        ae.eval()
        with torch.no_grad():val=float((ae(y[tr:tv].cuda())-y[tr:tv].cuda()).abs().mean())
        history.append({'epoch':epoch+1,'train_mae':total/tr,'val_mae':val})
        if val<best:
            best=val;wait=0;best_epoch=epoch+1;torch.save(ae.state_dict(),out/'checkpoint.pt')
        else:wait+=1
        if epoch%10==0:print('AE',name,epoch+1,history[-1],flush=True)
        if wait>=20:break
    ae.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True));ae.eval();metrics={}
    with torch.no_grad():
        for split,x in [('train',y[:tr]),('validation',y[tr:tv])]:
            x=x.cuda();err=ae(x)-x;metrics[split]={'mae':float(err.abs().mean()),'mse':float(err.square().mean())}
    json_write(out/'result.json',{'dataset':name,'seed':2021,'input_dim':1,'latent_dim':32,'hidden_dim':64,'loss':'MAE',
      'learning_rate':.0005,'batch_size':512,'maximum_epochs':500,'patience':20,'epochs':len(history),'best_epoch':best_epoch,
      'train_seconds':time.perf_counter()-start,'metrics':metrics,'history':history,'official_class':'LatentTSF/my_AE.py::AutoEncoder',
      'protocol_adjustment':'Pointwise AE training on unique training timestamps, batch512; avoids overweighting timestamps through overlapping windows.'})
    print('AE_FINISHED',name,metrics,flush=True)

if __name__=='__main__':
    audit_data()
    for name in ('NP','PJM'):train(name)
    rows=[]
    for name in ('NP','PJM'):
        r=json.loads((ROOT/'ae'/name/'result.json').read_text())
        for split,m in r['metrics'].items():rows.append({'dataset':name,'split':split,**m,'best_epoch':r['best_epoch'],'epochs':r['epochs']})
    pd.DataFrame(rows).to_csv(ROOT/'ae_reconstruction.csv',index=False)
