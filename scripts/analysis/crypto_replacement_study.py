"""Fixed hourly signals on the existing eight OKX crypto swaps; research only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.strategy_replacement_study import (
    START, END, PERIODS, STEP, add_features, execute, metrics,
)


def signals_for(inst,frame):
    h=frame.resample('1h').agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    counts=frame.close.resample('1h').count()
    h=h[counts==4].copy()
    previous=h.close.shift()
    tr=pd.concat([h.high-h.low,(h.high-previous).abs(),(h.low-previous).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean()
    ema8=h.close.ewm(span=8,adjust=False,min_periods=72).mean()
    ema24=h.close.ewm(span=24,adjust=False,min_periods=72).mean()
    ema72=h.close.ewm(span=72,adjust=False,min_periods=72).mean()
    high=h.high.shift().rolling(24,min_periods=24).max()
    low=h.low.shift().rolling(24,min_periods=24).min()
    rows=[]
    for i in range(72,len(h)):
        stamp=h.index[i]+pd.Timedelta(hours=1)
        if stamp not in frame.index or stamp<START or stamp>=END:
            continue
        px=float(h.close.iloc[i]);width=max(.01,1.5*float(atr.iloc[i])/px)
        if not np.isfinite(width) or width>.05:
            continue
        trend=1 if ema24.iloc[i]>ema72.iloc[i] else -1
        breakout=1 if px>high.iloc[i] and trend==1 else -1 if px<low.iloc[i] and trend==-1 else 0
        pullback=1 if (trend==1 and px>ema8.iloc[i] and h.close.iloc[i-1]<=ema8.iloc[i-1]) else (
                -1 if trend==-1 and px<ema8.iloc[i] and h.close.iloc[i-1]>=ema8.iloc[i-1] else 0)
        for strategy,side,hours in [('hourly_breakout',breakout,12),('hourly_pullback',pullback,6)]:
            if side:
                rows.append(dict(strategy=strategy,inst_id=inst,entry_ts=stamp,signal_ts=stamp,
                    side=side,stop_width=width,take_width=2*width,
                    deadline=stamp+pd.Timedelta(hours=hours),
                    strength=abs(px/h.close.iloc[i-1]-1)))
    return rows


def main(root):
    u=pd.read_csv(root/'universe.csv');frames={};funding={};signals=[];coverage=[]
    for inst in u.instId:
        f=pd.read_csv(root/'market'/f'{inst}.csv');f.ts=pd.to_datetime(f.ts,utc=True)
        f=add_features(f.set_index('ts').sort_index());frames[inst]=f
        signals.extend(signals_for(inst,f))
        raw=pd.DataFrame(json.loads((root/'market'/f'{inst}_funding.json').read_text()))
        raw['ts']=pd.to_datetime(pd.to_numeric(raw.fundingTime),unit='ms',utc=True)
        raw['rate']=pd.to_numeric(raw.realizedRate.where(raw.realizedRate.ne(''),raw.fundingRate))
        rates=raw.drop_duplicates('ts').set_index('ts').rate.sort_index();funding[inst]=rates
        gap=rates.index.to_series().diff().dt.total_seconds().max()/3600
        assert rates.index.min()<=START and rates.index.max()>=END and gap<=8
        coverage.append(dict(inst=inst,first=str(rates.index.min()),last=str(rates.index.max()),max_gap_h=gap))
    sig=pd.DataFrame(signals).sort_values(['entry_ts','strength','inst_id'],ascending=[True,False,True])
    sig.to_csv(root/'signals.csv',index=False)
    summary=[];contributions=[];monthly=[]
    for strategy in ['cash','hourly_breakout','hourly_pullback']:
        for cost in [10,20]:
            for period,(start,end) in PERIODS.items():
                tr,rejected=execute(sig[sig.strategy==strategy],frames,u,funding,cost,start,end,
                                    max_positions=5,max_daily=4)
                stats,eq=metrics(tr,frames,start,end)
                summary.append(dict(stats,strategy=strategy,cost_bps=cost,period=period,rejected=len(rejected)))
                tr.to_csv(root/f'trades_{strategy}_{period}_{cost}.csv',index=False)
                rejected.to_csv(root/f'rejected_{strategy}_{period}_{cost}.csv',index=False)
                eq.rename('equity').to_csv(root/f'equity_{strategy}_{period}_{cost}.csv')
                if len(tr):
                    for inst,g in tr.groupby('inst_id'):
                        contributions.append(dict(strategy=strategy,period=period,cost_bps=cost,
                                                  inst=inst,trades=len(g),net_usdt=g.pnl.sum()))
                    for month,g in tr.groupby(tr.entry_ts.dt.strftime('%Y-%m')):
                        monthly.append(dict(strategy=strategy,period=period,cost_bps=cost,
                                            month=month,trades=len(g),net_usdt=g.pnl.sum()))
    summary=pd.DataFrame(summary);summary.to_csv(root/'summary.csv',index=False)
    pd.DataFrame(contributions).to_csv(root/'contributions.csv',index=False)
    pd.DataFrame(monthly).to_csv(root/'monthly.csv',index=False)
    (root/'funding_coverage.json').write_text(json.dumps(coverage,indent=2))
    gates=[]
    for strategy in ['hourly_breakout','hourly_pullback']:
        v=summary[(summary.strategy==strategy)&(summary.period=='validation')&(summary.cost_bps==20)].iloc[0]
        r=summary[(summary.strategy==strategy)&(summary.period=='recent')&(summary.cost_bps==20)].iloc[0]
        gates.append(dict(strategy=strategy,passed=bool(v.trades>=30 and v.net_usdt>0 and
            v.profitable_symbols>=3 and v.biggest_positive_share<=.5 and r.net_usdt>0)))
    (root/'gates.json').write_text(json.dumps(gates,indent=2))
    print(summary.to_string(index=False));print(gates)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('results/crypto_replacement_20260911'))
    main(ap.parse_args().root)
