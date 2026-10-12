# Failures and negative results

All metric rows, including inferior methods, are retained. No seed or difficult window was deleted.

| dataset   |   seed | nonrecurrent         | shared_loop          |   relative_mse_gain |   relative_mae_gain |   mse_difference_loop_minus_control |
|:----------|-------:|:---------------------|:---------------------|--------------------:|--------------------:|------------------------------------:|
| NP        |   2021 | NP_H360_C4_K1_s2021  | NP_H360_C5_K2_s2021  |         0.0214042   |           0.0178789 |                        -0.0105632   |
| PJM       |   2021 | PJM_H360_B2_K1_s2021 | PJM_H360_C5_K4_s2021 |        -0.000827757 |          -0.0112703 |                         0.000100471 |

Every seed2021 configuration versus its equal-budget B0 and the original checkpoint:

NP_H24_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=1.87%. Validation MAE=0.106693; best Stage1 epoch=1/6, Stage2=6/11.

NP_H24_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.13%, MAE gain=0.79%; versus verified original checkpoint MSE gain=2.97%. Validation MAE=0.104927; best Stage1 epoch=1/6, Stage2=6/11.

NP_H24_C5_K4_s2021: relative test MSE gain versus equally continued B0=-1.00%, MAE gain=-0.12%; versus verified original checkpoint MSE gain=0.89%. Validation MAE=0.107212; best Stage1 epoch=1/6, Stage2=6/11.

NP_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-14.50%. Validation MAE=0.222573; best Stage1 epoch=1/6, Stage2=12/17.

NP_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.83%, MAE gain=1.19%; versus verified original checkpoint MSE gain=-12.40%. Validation MAE=0.215863; best Stage1 epoch=1/6, Stage2=8/13.

NP_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=2.21%, MAE gain=-0.74%; versus verified original checkpoint MSE gain=-11.98%. Validation MAE=0.213409; best Stage1 epoch=1/6, Stage2=9/14.

NP_H360_C1_K4_s2021: relative test MSE gain versus equally continued B0=-0.57%, MAE gain=0.04%; versus verified original checkpoint MSE gain=-15.15%. Validation MAE=0.226455; best Stage1 epoch=1/6, Stage2=6/11.

NP_H360_C3_K4_s2021: relative test MSE gain versus equally continued B0=3.81%, MAE gain=1.43%; versus verified original checkpoint MSE gain=-10.14%. Validation MAE=0.212747; best Stage1 epoch=1/6, Stage2=4/9.

NP_H360_C4_K1_s2021: relative test MSE gain versus equally continued B0=-0.14%, MAE gain=-0.73%; versus verified original checkpoint MSE gain=-14.66%. Validation MAE=0.208630; best Stage1 epoch=1/6, Stage2=16/20.

NP_H360_C5_K2_s2021: relative test MSE gain versus equally continued B0=2.00%, MAE gain=1.08%; versus verified original checkpoint MSE gain=-12.21%. Validation MAE=0.216265; best Stage1 epoch=1/6, Stage2=13/18.

NP_H360_C5_K4_s2021: relative test MSE gain versus equally continued B0=2.10%, MAE gain=1.11%; versus verified original checkpoint MSE gain=-12.09%. Validation MAE=0.218404; best Stage1 epoch=1/6, Stage2=8/13.

NP_H360_FULL_K4_s2021: relative test MSE gain versus equally continued B0=3.77%, MAE gain=1.77%; versus verified original checkpoint MSE gain=-10.18%. Validation MAE=0.213199; best Stage1 epoch=1/6, Stage2=13/18.

NP_H360_NOANCHOR_K4_s2021: relative test MSE gain versus equally continued B0=6.30%, MAE gain=3.08%; versus verified original checkpoint MSE gain=-7.29%. Validation MAE=0.215158; best Stage1 epoch=1/6, Stage2=14/19.

PJM_H24_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=1.56%. Validation MAE=0.109202; best Stage1 epoch=1/6, Stage2=3/8.

PJM_H24_B1_K1_s2021: relative test MSE gain versus equally continued B0=1.03%, MAE gain=0.57%; versus verified original checkpoint MSE gain=2.57%. Validation MAE=0.108863; best Stage1 epoch=1/6, Stage2=1/6.

PJM_H24_C5_K4_s2021: relative test MSE gain versus equally continued B0=0.42%, MAE gain=0.27%; versus verified original checkpoint MSE gain=1.97%. Validation MAE=0.109262; best Stage1 epoch=1/6, Stage2=1/6.

PJM_H360_B0_K1_s2021: relative test MSE gain versus equally continued B0=0.00%, MAE gain=0.00%; versus verified original checkpoint MSE gain=-10.79%. Validation MAE=0.163790; best Stage1 epoch=1/6, Stage2=8/13.

PJM_H360_B1_K1_s2021: relative test MSE gain versus equally continued B0=5.52%, MAE gain=2.44%; versus verified original checkpoint MSE gain=-4.68%. Validation MAE=0.167901; best Stage1 epoch=1/6, Stage2=2/7.

PJM_H360_B2_K1_s2021: relative test MSE gain versus equally continued B0=4.80%, MAE gain=4.60%; versus verified original checkpoint MSE gain=-5.47%. Validation MAE=0.160664; best Stage1 epoch=1/6, Stage2=4/9.

PJM_H360_C1_K4_s2021: relative test MSE gain versus equally continued B0=-4.24%, MAE gain=-1.21%; versus verified original checkpoint MSE gain=-15.50%. Validation MAE=0.162155; best Stage1 epoch=1/6, Stage2=16/20.

PJM_H360_C3_K4_s2021: relative test MSE gain versus equally continued B0=9.43%, MAE gain=4.90%; versus verified original checkpoint MSE gain=-0.34%. Validation MAE=0.168792; best Stage1 epoch=1/6, Stage2=5/10.

PJM_H360_C4_K1_s2021: relative test MSE gain versus equally continued B0=7.25%, MAE gain=2.80%; versus verified original checkpoint MSE gain=-2.76%. Validation MAE=0.163477; best Stage1 epoch=1/6, Stage2=4/9.

PJM_H360_C5_K2_s2021: relative test MSE gain versus equally continued B0=6.48%, MAE gain=3.31%; versus verified original checkpoint MSE gain=-3.61%. Validation MAE=0.167089; best Stage1 epoch=1/6, Stage2=2/7.

PJM_H360_C5_K4_s2021: relative test MSE gain versus equally continued B0=4.73%, MAE gain=3.53%; versus verified original checkpoint MSE gain=-5.56%. Validation MAE=0.164605; best Stage1 epoch=1/6, Stage2=4/9.

PJM_H360_FULL_K4_s2021: relative test MSE gain versus equally continued B0=11.09%, MAE gain=7.31%; versus verified original checkpoint MSE gain=1.50%. Validation MAE=0.161695; best Stage1 epoch=1/6, Stage2=5/10.

PJM_H360_NOANCHOR_K4_s2021: relative test MSE gain versus equally continued B0=13.44%, MAE gain=5.24%; versus verified original checkpoint MSE gain=4.09%. Validation MAE=0.162514; best Stage1 epoch=1/6, Stage2=3/8.

Original failures:

[
  {
    "path": "logs\\NP_H360_B1_K1_s2021_attempt1.log",
    "content": "2 02:54:32 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):\n  File \"D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\ts_benchmark\\evaluation\\strategy\\forecasting.py\", line 51, in execute\n    single_series_results = self._execute(\n  File \"D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\ts_benchmark\\evaluation\\strategy\\rolling_forecast.py\", line 245, in _execute\n    return self._eval_batch(series, meta_info, model, series_name)\n  File \"D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\ts_benchmark\\evaluation\\strategy\\rolling_forecast.py\", line 445, in _eval_batch\n    fit_method(\n  File \"D:\\worldts_future_response_loop\\scripts\\train.py\", line 110, in fit_observed\n    torch.cuda.reset_peak_memory_stats();begin=time.time();original_fit(self,*a,**kw);torch.cuda.synchronize()\n  File \"D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\ts_benchmark\\baselines\\worldts\\worldts.py\", line 726, in forecast_fit\n    self._run_stage_train(\n  File \"D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\ts_benchmark\\baselines\\worldts\\worldts.py\", line 112, in _run_stage_train\n    # Preserve the historical path unless an experiment opts into VISReg.\n  File \"D:\\worldts_future_response_loop\\scripts\\train.py\", line 85, in epoch_record\n    write_json(out/'status.json',{'run_id':out.name,'status':'RUNNING','stage':stage,'epoch':epoch,'validation_loss':row['validation_loss'],'pid':os.getpid()})\n  File \"D:\\worldts_future_response_loop\\scripts\\common.py\", line 14, in write_json\n    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8');tmp.replace(p)\n  File \"E:\\Anaconda\\envs\\persist_stable_251\\lib\\pathlib.py\", line 1247, in replace\n    self._accessor.replace(self, target)\nPermissionError: [WinError 5] \ufffd\u073e\ufffd\ufffd\ufffd\ufffd\u02a1\ufffd: 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json.tmp' -> 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json'\n\n[WinError 5] \ufffd\u073e\ufffd\ufffd\ufffd\ufffd\u02a1\ufffd: 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json.tmp' -> 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json'\n2026-10-12 02:54:32 [INFO] ts_benchmark.recording(103): loading log file D:\\worldts_future_response_loop\\upstream\\WorldTS\\DAG-bench\\result\\worldts_response_loop/NP_H360_B1_K1_s2021\\WorldTS.1791744872.DESKTOP-C4FOPGI.25408.csv.tar.gz\n2026-10-12 02:54:32 [INFO] ts_benchmark.report.utils.leaderboard(164): There are 7 NaN values in the leaderboard due to a higher-than-threshold NaN ratio in the corresponding model+algorithm pairs.\nTraceback (most recent call last):\n  File \"D:\\worldts_future_response_loop\\scripts\\train.py\", line 128, in execute\n    sys.argv=argv;runpy.run_path(str(src/'scripts/run_benchmark.py'),run_name='__main__');assert 'metrics' in state,'Official runner failed; inspect log'\nAssertionError: Official runner failed; inspect log\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File \"D:\\worldts_future_response_loop\\scripts\\train.py\", line 148, in <module>\n    execute(SPEC['dataset'],SPEC['horizon'])\n  File \"D:\\worldts_future_response_loop\\scripts\\train.py\", line 142, in execute\n    write_json(out/'status.json',{'status':'FAILED','exit_code':1,'traceback':traceback.format_exc(),'run_id':out.name});raise\n  File \"D:\\worldts_future_response_loop\\scripts\\common.py\", line 14, in write_json\n    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8');tmp.replace(p)\n  File \"E:\\Anaconda\\envs\\persist_stable_251\\lib\\pathlib.py\", line 1247, in replace\n    self._accessor.replace(self, target)\nPermissionError: [WinError 5] \ufffd\u073e\ufffd\ufffd\ufffd\ufffd\u02a1\ufffd: 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json.tmp' -> 'D:\\\\worldts_future_response_loop\\\\runs\\\\NP_H360_B1_K1_s2021\\\\status.json'\n"
  },
  {
    "path": "logs\\orchestrate_attempt1.log",
    "content": "COMMAND_EXIT data_link 0 D:\\worldts_future_response_loop\\logs\\data_link_attempt1.log\nCOMMAND_EXIT NP_H360_B1_K1_s2021 1 D:\\worldts_future_response_loop\\logs\\NP_H360_B1_K1_s2021_attempt1.log\nTraceback (most recent call last):\n  File \"D:\\worldts_future_response_loop\\scripts\\orchestrate.py\", line 40, in <module>\n    if __name__=='__main__':main()\n  File \"D:\\worldts_future_response_loop\\scripts\\orchestrate.py\", line 17, in main\n    write_json(ROOT/'STATUS.json',{'stage':stage,'status':'RUNNING','active_run':row['run_id'],'overall_complete':False});run_command('train.py','--run-id',row['run_id'],prefix=row['run_id'])\n  File \"D:\\worldts_future_response_loop\\scripts\\orchestrate.py\", line 9, in run_command\n    if r.returncode:raise RuntimeError(str(log))\nRuntimeError: D:\\worldts_future_response_loop\\logs\\NP_H360_B1_K1_s2021_attempt1.log\n"
  },
  {
    "path": "logs\\synthetic_attempt1.log",
    "content": "Traceback (most recent call last):\n  File \"D:\\worldts_future_response_loop\\scripts\\synthetic.py\", line 60, in <module>\n    if __name__=='__main__':main()\n  File \"D:\\worldts_future_response_loop\\scripts\\synthetic.py\", line 58, in main\n    rows=[run_one(a,noise,v) for a in [.2,.95] for noise in [.05,.5] for v in ['B0','B1','C5']];write_csv(ROOT/'results/stage1_synthetic.csv',rows)\n  File \"D:\\worldts_future_response_loop\\scripts\\synthetic.py\", line 58, in <listcomp>\n    rows=[run_one(a,noise,v) for a in [.2,.95] for noise in [.05,.5] for v in ['B0','B1','C5']];write_csv(ROOT/'results/stage1_synthetic.csv',rows)\n  File \"D:\\worldts_future_response_loop\\scripts\\synthetic.py\", line 55, in run_one\n    with torch.no_grad():assert torch.equal(restored.forecast(audit_x,x_future=audit_f),model.forecast(audit_x,x_future=audit_f))\nAssertionError\n"
  }
]

The synthetic reload failure was an integrity-check freezing-state mismatch with equal parameter hashes and a measured7.15e-7 kernel difference; it was corrected before candidate training. No NaN/OOM result is assumed absent unless the saved logs and integrity audits establish it. Inferior forecasting scores are statistical/mechanistic evidence, not automatically implementation failures.
