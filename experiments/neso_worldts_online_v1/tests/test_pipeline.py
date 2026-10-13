import numpy as np
import pandas as pd
import pytest
from neso_online.prepare import (
    _utc_end_from_settlement,_normalize_forecast_chunk,load_forecasts,load_demand,
    build_pairs,split_pairs,
)


def test_uk_daylight_saving_period_counts():
    for day,expected in [('2026-03-29',46),('2026-10-25',50),('2026-01-01',48)]:
        a=_utc_end_from_settlement(pd.Series([day,day]),pd.Series([1,expected]))
        midnight=pd.Timestamp(day).tz_localize('Europe/London').tz_convert('UTC')
        nextmid=(pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize('Europe/London').tz_convert('UTC')
        assert a.iloc[0]==midnight+pd.Timedelta(minutes=30)
        assert a.iloc[1]==nextmid


def test_period_end_forecast_matches_settlement():
    df=pd.DataFrame({'DATE_GMT':['2025-01-01'], 'TIME_GMT':['00:30:00'],
                     'SETTLEMENT_DATE':['2025-01-01'],'SETTLEMENT_PERIOD':[1],
                     'EMBEDDED_WIND_FORECAST':[5452],'EMBEDDED_SOLAR_FORECAST':[0],
                     'Forecast_Datetime':['2025-01-01T00:12:00']})
    out=_normalize_forecast_chunk(df)
    assert str(out.iloc[0]['valid_end_utc'])=='2025-01-01 00:30:00+00:00'
    df.loc[0,'SETTLEMENT_PERIOD']=2
    with pytest.raises(ValueError,match='does not match'):
        _normalize_forecast_chunk(df)


def test_streamed_synthetic_data(tmp_path):
    from scripts.make_synthetic_fixture import make
    make(tmp_path,start='2025-12-01',end='2026-04-17')
    f,fm=load_forecasts([tmp_path/'forecast_synthetic.csv'])
    d,dm=load_demand([tmp_path/'demand_synthetic.csv'])
    assert fm['settlement_time_mismatches']==0
    assert dm['excluded_non_actual_rows']>0
    arr,stats=build_pairs(f,d,lookback=96,horizon=48,revision_hours=6,
                          issue_every_hours=6,min_eval_slots=12,max_samples=125)
    assert len(arr['history'])>5
    assert arr['history'].shape[1:]==(96,3)
    assert arr['cov_a'].shape[1:]==(48,2)
    assert arr['update_mask'].shape[1:]==(48,)
    assert np.any(arr['cov_a']!=arr['cov_b'])
    # The update at B MUST NOT change already-ended target periods.
    assert np.array_equal(arr['cov_a'][~arr['update_mask']],arr['cov_b'][~arr['update_mask']])
    assert np.all(pd.to_datetime(arr['issue_b'],utc=True)>pd.to_datetime(arr['issue_a'],utc=True))
    for arr_a,arr_b in zip(arr['issue_a'],arr['issue_b']):
        assert pd.Timestamp(str(arr_b))>pd.Timestamp(str(arr_a))


def test_revision_module_zero_initialized():
    import torch
    from neso_online.native_worldts import LatentRevisionUpdater
    a=LatentRevisionUpdater(8,2,16)
    z=torch.randn(2,20,8)
    ca=torch.randn(2,20,2)
    cb=ca+0.1
    mask=torch.ones(2,20)
    result=a(z,ca,cb,mask)
    assert torch.equal(result,z)


def test_uk_date_is_day_first():
    # 03/04/2026 is 3 April, NOT March 4.
    x=_utc_end_from_settlement(pd.Series(['03/04/2026']),pd.Series([1]))
    assert x.iloc[0]==pd.Timestamp('2026-04-02T23:30:00Z')


def test_splits_do_not_share_target_labels():
    arr={'history':np.zeros((4,4,3),np.float32),
         'first_target_end':np.array(['2025-12-31 12:00Z','2025-12-31 23:30Z',
                                      '2026-01-01 01:00Z','2026-04-01 00:30Z']),
         'last_target_end':np.array(['2025-12-31 12:30Z','2026-01-01 00:30Z',
                                    '2026-01-01 01:30Z','2026-04-01 01:00Z'])}
    groups=split_pairs(arr)
    assert [len(groups[k]['history']) for k in ('train','val','test')]==[1,1,1]
