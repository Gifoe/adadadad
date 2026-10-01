"""Predicate-renaming-invariant signed unary-rule graph fingerprint."""
import hashlib
from symmetric_adapter import literal,rule

def invariant_hash(formulas,query,selected_entity,fol,rounds=12):
    nodes=set();edges=[];fact_roles=[]
    for f in formulas:
        r=rule(f,fol)
        if r:
            a,b=r;nodes.update([a[0],b[0]]);edges.append((a[0],b[0],a[1],b[1]))
        else:
            l=literal(f,fol)
            if l:
                atom=f.operand if l[1] else f; ent=getattr(atom.args[0],'constant',None);nodes.add(l[0]);fact_roles.append((l[0],l[1],ent==selected_entity))
    q=literal(query,fol);nodes.add(q[0])
    incoming={n:[] for n in nodes};outgoing={n:[] for n in nodes}
    for a,b,an,bn in edges:outgoing[a].append((b,an,bn));incoming[b].append((a,an,bn))
    facts={n:[] for n in nodes}
    for n,neg,selected in fact_roles:facts[n].append((neg,selected))
    colors={n:hashlib.sha256(repr((sorted(facts[n]),n==q[0],q[1] if n==q[0] else None)).encode()).hexdigest() for n in nodes}
    for _ in range(rounds):
        colors={n:hashlib.sha256(repr((colors[n],sorted((colors[v],an,bn) for v,an,bn in incoming[n]),sorted((colors[v],an,bn) for v,an,bn in outgoing[n]))).encode()).hexdigest() for n in nodes}
    signature=(sorted(colors.values()),sorted((colors[a],colors[b],an,bn) for a,b,an,bn in edges),sorted((colors[n],neg,sel) for n,neg,sel in fact_roles),colors[q[0]],q[1])
    return hashlib.sha256(repr(signature).encode()).hexdigest()
