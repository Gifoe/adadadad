import numpy as np
import pandas as pd
import pytest
import torch
from neso_online.prepare import _normalize_forecast_chunk,_utc_end_from_settlement
from neso_online.native_worldts import replay,evaluate_stage,as_loader
from test_model_driver import DummyCore,_arrays

def test_real_2026_hhmm_is_validated_against_settlement():
    # Real API sample: UK summer settlement period48 ends at23:00 UTC.
    record=pd.DataFrame({'DATE_GMT':['2026-08-07'],'TIME_GMT':['23:00'],
        'SETTLEMENT_DATE':['2026-08-07'],'SETTLEMENT_PERIOD':[48],
        'EMBEDDED_WIND_FORECAST':[838],'EMBEDDED_SOLAR_FORECAST':[0],
        'Forecast_Datetime':['2026-07-27T16:52:39']})
    assert _normalize_forecast_chunk(record).iloc[0].valid_end_utc==pd.Timestamp('2026-08-07T23:00:00Z')
    record['SETTLEMENT_PERIOD']=47
    with pytest.raises(ValueError,match='does not match'):_normalize_forecast_chunk(record)

def test_settlement_period_bounds_are_real_dst_day_bounds():
    with pytest.raises(ValueError,match='Invalid settlement'):
        _utc_end_from_settlement(pd.Series(['2026-03-29']),pd.Series([47]))
    with pytest.raises(ValueError,match='Invalid settlement'):
        _utc_end_from_settlement(pd.Series(['2026-01-01']),pd.Series([49]))
    assert _utc_end_from_settlement(pd.Series(['2026-10-25']),pd.Series([50])).iloc[0]==pd.Timestamp('2026-10-26T00:00Z')

def test_replay_mw_units_and_saved_absolute_predictions(tmp_path):
    model=DummyCore()
    for parameter in model.parameters():parameter.data.zero_()
    arrays=_arrays(2);arrays['target'][:]=1
    scaler={'mean':[10000.,100.,200.],'std':[2000.,50.,70.]}
    result=replay(model,arrays,scaler,torch.device('cpu'),batch=2,save_path=tmp_path/'p.npz')
    old=result['methods']['old']
    assert old['mae']==2000. and old['mse']==4000000.
    assert old['normalized_mae']==1. and old['normalized_mse']==1.
    with np.load(tmp_path/'p.npz') as z:
        assert np.all(z['pred_old_mw']==10000.) and np.all(z['target_mw']==12000.)

def test_stage2_validation_is_label_free():
    model=DummyCore();original=model.forward
    def forward(input,future_target=None,**kwargs):
        assert future_target is None
        return original(input,future_target=future_target,**kwargs)
    model.forward=forward
    score=evaluate_stage(model,as_loader(_arrays(3),2,False,2021),torch.device('cpu'),2)
    assert np.isfinite(score)
def test_iso_z_settlement_is_civil_date():
    from neso_online.prepare import _utc_end_from_settlement
    import pandas as pd
    got=_utc_end_from_settlement(pd.Series(['2025-07-01T00:00:00Z']),pd.Series([1]))
    assert got.iloc[0]==pd.Timestamp('2025-06-30T23:30:00Z')

def test_ambiguous_source_rows_explicitly_excluded(tmp_path):
    from neso_online.prepare import load_forecasts
    import pandas as pd
    rows=pd.DataFrame({'DATE_GMT':['2026-04-22','2026-04-22'],'TIME_GMT':['00:30:00','00:00:00'],
        'SETTLEMENT_DATE':['2026-04-22','2026-04-22'],'SETTLEMENT_PERIOD':[1,2],
        'EMBEDDED_WIND_FORECAST':[100,100],'EMBEDDED_SOLAR_FORECAST':[0,0],'Forecast_Datetime':['2026-04-20T00:00:00Z']*2})
    path=tmp_path/'archive.csv';rows.to_csv(path,index=False)
    data,stats=load_forecasts([path],exclude_settlement_mismatch=True)
    assert len(data)==1 and stats['settlement_time_mismatches']==1
    assert stats['file_coverage'][0]['excluded_future_timestamp_conflicts']==1
