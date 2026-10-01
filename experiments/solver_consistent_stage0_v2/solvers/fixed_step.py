import math

def schedule(steps, kind='uniform'):
    if steps < 1: raise ValueError('positive step count required')
    if kind == 'uniform': weights=[1.]*steps
    elif kind == 'front-loaded': weights=[1.+i/max(1,steps-1) for i in range(steps)]
    elif kind == 'back-loaded': weights=[2.-i/max(1,steps-1) for i in range(steps)]
    else: raise ValueError(kind)
    result=[x/sum(weights) for x in weights]
    assert min(result)>0 and math.isclose(sum(result),1.,abs_tol=1e-12)
    return result

def integrate(field, state, dts, solver='euler'):
    if solver not in ('euler','heun','rk4'): raise ValueError(solver)
    if not dts or min(dts)<=0 or not math.isclose(sum(dts),1.,abs_tol=1e-8): raise ValueError('schedule must partition [0,1]')
    t=0.; nfe=0
    for dt in dts:
        k1=field(state,t); nfe+=1
        if solver == 'euler': state=state+dt*k1
        elif solver == 'heun':
            k2=field(state+dt*k1,t+dt); nfe+=1
            state=state+dt*(k1+k2)/2
        else:
            k2=field(state+dt*k1/2,t+dt/2)
            k3=field(state+dt*k2/2,t+dt/2)
            k4=field(state+dt*k3,t+dt); nfe+=3
            state=state+dt*(k1+2*k2+2*k3+k4)/6
        t+=dt
    return state,nfe
