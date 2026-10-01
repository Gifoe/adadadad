import os,sys,contextlib,io,time,pathlib
from official_adapter import official_generator
g,concepts=official_generator()
start=time.time()
for d in [6,8,12]:
    n=0; shortcut=0
    for i in range(1000):
        with contextlib.redirect_stdout(io.StringIO()): q=g.generate_question(d+1,concepts,formula_ordering='random',distractors='relevant')
        n+=q[0] is not None
        if q[0] is not None: shortcut+=int((q[0]+' '+q[1]).lower().split().count('not')%2==0)==int(q[4]=='True')
    print(d,n,'/1000','parity acc',shortcut/max(n,1),'sec',round(time.time()-start),flush=True)
