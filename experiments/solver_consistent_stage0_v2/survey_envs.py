import pathlib,re,json,os
root=pathlib.Path('E:/Anaconda/envs'); results=[]
for env in root.iterdir():
    packages={}
    for name in ['torch','numpy','mkl','mkl_service']:
        for f in (env/'Lib/site-packages').glob(name+'-*.dist-info/METADATA'):
            text=f.read_text(errors='replace'); packages[name]=re.search(r'^Version: (.+)$',text,re.MULTILINE).group(1)
    results.append({'env':env.name,'packages':packages})
print(json.dumps(results,indent=2)); print('PATH',os.environ['PATH'])
