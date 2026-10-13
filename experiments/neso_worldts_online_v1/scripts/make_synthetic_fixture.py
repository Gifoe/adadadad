"""Create SYNTHETIC data for unit tests only. Never use these as NESO results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def settlement_fields(end: pd.Timestamp):
    # Period is defined by the local settlement DAY of the 30 minutes just ended.
    one_ns = pd.Timedelta(nanoseconds=1)
    local=(end-one_ns).tz_convert('Europe/London')
    midnight=pd.Timestamp(local.date()).tz_localize('Europe/London').tz_convert('UTC')
    period=int((end-midnight)/pd.Timedelta(minutes=30))
    return local.date().isoformat(),period


def make(out:Path, start='2025-11-01', end='2026-04-17', hour_step=6):
    out.mkdir(parents=True,exist_ok=True)
    times=pd.date_range(start,end,freq='30min',inclusive='left',tz='UTC')
    phase=2*np.pi*np.arange(len(times))/48
    rng=np.random.default_rng(11)
    wind=4000+1000*np.sin(phase/5)+rng.normal(0,100,len(times))
    solar=3000*np.maximum(0,np.sin(phase-np.pi/2))
    nd=26000+3000*np.cos(phase)+0.1*wind-0.15*solar+rng.normal(0,70,len(times))
    data=[]
    # Actual ND rows include local date/SP; across DST there may be 46/50.
    for t,y in zip(times,nd):
        day,per=settlement_fields(t)
        data.append({'SETTLEMENT_DATE':day,'SETTLEMENT_PERIOD':per,'ND':round(float(y),3),
                     'FORECAST_ACTUAL_INDICATOR':'A'})
    demand=pd.DataFrame(data)
    demand.loc[::500,'FORECAST_ACTUAL_INDICATOR']='F'  # intentionally excluded
    demand.to_csv(out/'demand_synthetic.csv',index=False)
    base=pd.DataFrame({'valid_end_utc':times,'wind_mw':wind,'solar_mw':solar}).set_index('valid_end_utc')
    issues=pd.date_range(times[336],times[-145],freq=f'{hour_step}h',tz='UTC') + pd.Timedelta(minutes=12)
    forecasts=[]
    for j,issue in enumerate(issues):
        targets=pd.date_range(issue.ceil('30min'),periods=144,freq='30min',tz='UTC')
        vals=base.reindex(targets)
        for k,(valid,row) in enumerate(zip(targets,vals.itertuples())):
            day,per=settlement_fields(valid)
            forecasts.append({'DATE_GMT':valid.strftime('%Y-%m-%d'),
                              'TIME_GMT':valid.strftime('%H:%M:%S'),
                              'SETTLEMENT_DATE':day,'SETTLEMENT_PERIOD':per,
                              'EMBEDDED_WIND_FORECAST':round(row.wind_mw+(8*np.sin(j/3+k/21)),3),
                              'EMBEDDED_SOLAR_FORECAST':round(row.solar_mw+(4*np.cos(j/4+k/21)),3),
                              'Forecast_Datetime':issue.strftime('%Y-%m-%dT%H:%M:%S')})
    pd.DataFrame(forecasts).to_csv(out/'forecast_synthetic.csv',index=False)
    print({'synthetic_only':True,'actual_rows':len(data),'forecast_rows':len(forecasts),
           'issue_times':len(issues),'location':str(out)})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',default='fixtures/synthetic')
    a.add_argument('--start',default='2025-11-01');a.add_argument('--end',default='2026-04-17')
    z=a.parse_args();make(Path(z.out),z.start,z.end)
