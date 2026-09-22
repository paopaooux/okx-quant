"""Fixed-rule stock replacement screen on an isolated OKX archive. No trading API."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stocks.config import Config
from strategies.stocks.market import sessions
from strategies.stocks.research import events

STEP = pd.Timedelta(minutes=15)
START = pd.Timestamp('2026-06-15T00:00Z')
END = pd.Timestamp('2026-09-11T07:35Z').floor('15min')
PERIODS = {
    'development': (START, pd.Timestamp('2026-08-01T00:00Z')),
    'validation': (pd.Timestamp('2026-08-01T00:00Z'), pd.Timestamp('2026-09-01T00:00Z')),
    'recent': (pd.Timestamp('2026-09-01T00:00Z'), END),
    'live_days': (pd.Timestamp('2026-09-04T10:36:57.464Z'), END),
}


def add_features(frame):
    f = frame.copy()
    previous = f.close.shift()
    tr = pd.concat([f.high-f.low, (f.high-previous).abs(), (f.low-previous).abs()],axis=1).max(axis=1)
    f['atr'] = tr.rolling(14,min_periods=14).mean()
    f['ema8'] = f.close.ewm(span=8,adjust=False,min_periods=32).mean()
    f['ema32'] = f.close.ewm(span=32,adjust=False,min_periods=32).mean()
    f['ema_slope'] = f.ema32.diff(4)
    f['trailing_volume'] = f.volume_quote.rolling(96,min_periods=96).sum()
    return f


def candidate_signals(frames):
    rows = []
    cash_sessions = sessions.sessions_between(START,END)
    for inst,f in frames.items():
        for sess in cash_sessions:
            g=f[(f.index>=sess.open_ts)&(f.index<sess.close_ts)].copy()
            # Opening range must include both opening bars, never a partial session.
            if len(g)<3 or not g.index[:2].equals(pd.date_range(sess.open_ts,periods=2,freq='15min')):
                continue
            g['vwap']=(g.close*g.volume).cumsum()/g.volume.cumsum().replace(0,np.nan)
            hi,lo=float(g.high.iloc[:2].max()),float(g.low.iloc[:2].min())
            found=set()
            for i in range(2,len(g)):
                bar=g.iloc[i];prev=g.iloc[i-1];signal_ts=g.index[i]+STEP
                if signal_ts<START or signal_ts>=sess.close_ts or signal_ts not in f.index:
                    continue
                if not np.isfinite(bar.atr) or bar.atr<=0:
                    continue
                width=max(.005,float(bar.atr/bar.close))
                if width>.03:
                    continue
                orb=1 if bar.close>hi and prev.close<=hi else -1 if bar.close<lo and prev.close>=lo else 0
                trend=1 if (bar.ema8>bar.ema32 and bar.ema_slope>0 and
                            bar.close>bar.vwap and prev.close<=prev.vwap) else (
                      -1 if (bar.ema8<bar.ema32 and bar.ema_slope<0 and
                             bar.close<bar.vwap and prev.close>=prev.vwap) else 0)
                for strategy,side,hours in [('opening_range',orb,3),('trend_reclaim',trend,2)]:
                    if not side or strategy in found:
                        continue
                    found.add(strategy)
                    rows.append(dict(strategy=strategy,inst_id=inst,signal_ts=signal_ts,
                        entry_ts=signal_ts,side=side,stop_width=width,take_width=2*width,
                        deadline=min(signal_ts+pd.Timedelta(hours=hours),sess.close_ts),
                        signal_price=bar.close,strength=abs(float(bar.close/prev.close-1))))
    # Original off-hours mechanism on the SAME eight instruments and 15m clock.
    end_frames={k:f.set_index(f.index+STEP) for k,f in frames.items()}
    ev=events.off_hours_dislocation(end_frames,Config(dislocation_bps=600),
        windows=sessions.closed_windows(START-pd.Timedelta(days=3),END+pd.Timedelta(days=8)))
    if len(ev):
        for e in ev.itertuples():
            if not START<=e.event_ts<END:
                continue
            f=frames[e.inst_id]
            if e.event_ts not in f.index:
                continue
            rows.append(dict(strategy='original_reversion',inst_id=e.inst_id,signal_ts=e.event_ts,
                entry_ts=e.event_ts,side=int(e.side),stop_width=.03,take_width=0.,
                deadline=min(e.resolve_ts+pd.Timedelta(hours=1),e.event_ts+pd.Timedelta(hours=30)),
                signal_price=float(f.loc[e.event_ts-STEP,'close']),strength=abs(float(e.deviation))))
    return pd.DataFrame(rows).sort_values(['entry_ts','strategy','strength','inst_id'],ascending=[True,True,False,True])


def path_exit(frame,entry_ts,side,stop_width,take_width,deadline):
    """Next-open entry; scan only bars after entry and before timeout. Stop first.

    A stop through the opening gap fills at the adverse open, not the barrier.
    Target is capped at the target price. Timeout executes at its bar open.
    """
    entry=float(frame.loc[entry_ts,'open'])
    if deadline not in frame.index:
        return None
    path=frame[(frame.index>=entry_ts)&(frame.index<deadline)]
    expected=pd.date_range(entry_ts,deadline,freq='15min',inclusive='left')
    if not path.index.equals(expected):
        return None
    stop=entry*(1-side*stop_width);target=entry*(1+side*take_width)
    for ts,b in path.iterrows():
        stopped=b.low<=stop if side==1 else b.high>=stop
        targeted=take_width>0 and (b.high>=target if side==1 else b.low<=target)
        if stopped:
            px=min(stop,b.open) if side==1 else max(stop,b.open)
            return entry,float(px),ts+STEP,'stop'
        if targeted:
            return entry,float(target),ts+STEP,'target'
    return entry,float(frame.loc[deadline,'open']),deadline,'deadline'


def execute(signals,frames,universe,funding,cost_bps,start,end,*,max_positions=3,max_daily=2):
    # Every chronological slice starts flat, with no train positions crossing into validation.
    selected=signals[(signals.entry_ts>=start)&(signals.deadline<end)]
    positions=[];days={};trades=[];rejected=[]
    specs=universe.set_index('instId')
    for s in selected.itertuples():
        f=frames[s.inst_id]
        reason=''
        positions=[(t,inst) for t,inst in positions if t>s.entry_ts]
        bar=f.loc[s.entry_ts-STEP]
        if len(positions)>=max_positions:reason='capacity'
        elif s.inst_id in {inst for _,inst in positions}:reason='same_instrument'
        elif days.get(s.entry_ts.date(),0)>=max_daily:reason='daily_limit'
        elif not np.isfinite(bar.trailing_volume) or bar.trailing_volume<150000:reason='liquidity_24h'
        elif bar.volume_quote<1400:reason='signal_bar_liquidity'
        spec=specs.loc[s.inst_id];px=float(f.loc[s.entry_ts,'open'])
        lot=float(spec.lotSz);contract=float(spec.ctVal)
        qty=math.floor((14/(px*contract))/lot+1e-10)*lot
        if qty<float(spec.minSz):reason=reason or 'minimum_lot'
        result=path_exit(f,s.entry_ts,s.side,s.stop_width,s.take_width,s.deadline)
        if result is None:reason=reason or 'missing_exit_path'
        rates=funding[s.inst_id]
        if rates.index.min()>s.entry_ts or rates.index.max()<s.deadline:
            reason=reason or 'funding_coverage'
        if reason:
            rejected.append(dict(inst_id=s.inst_id,entry_ts=s.entry_ts,reason=reason))
            continue
        ep,xp,xt,why=result
        gross=s.side*(xp/ep-1)
        # Settlement notionals use last completed candle close (historical mark unavailable).
        settlements=rates[(rates.index>s.entry_ts)&(rates.index<=xt)]
        fund=0.
        for stamp,rate in settlements.items():
            known=f[f.index+STEP<=stamp]
            fund+=-s.side*float(rate)*float(known.close.iloc[-1])/ep
        notional=qty*contract*ep
        cost=cost_bps/1e4
        net=gross+fund-cost
        trades.append(dict(inst_id=s.inst_id,entry_ts=s.entry_ts,exit_ts=xt,side=s.side,
            entry_px=ep,exit_px=xp,qty=qty,notional=notional,stop_width=s.stop_width,
            gross=gross,funding=fund,cost=cost,net=net,pnl=notional*net,
            reason=why,hold_h=(xt-s.entry_ts).total_seconds()/3600))
        positions.append((xt,s.inst_id));days[s.entry_ts.date()]=days.get(s.entry_ts.date(),0)+1
    cols=['inst_id','entry_ts','exit_ts','side','entry_px','exit_px','qty','notional',
          'stop_width','gross','funding','cost','net','pnl','reason','hold_h']
    return pd.DataFrame(trades,columns=cols),pd.DataFrame(rejected)


def metrics(trades,frames,start,end):
    # Marked at 15m closes, costs split at entry/exit. Funding booked at exit;
    # therefore intratrade equity omits unrealized funding until settlement.
    clock=pd.date_range(start.floor('15min'),end.floor('15min'),freq='15min')
    equity=pd.Series(70.,index=clock)
    for t in trades.itertuples():
        equity.loc[equity.index>=t.exit_ts]+=t.pnl
        inside=(clock>=t.entry_ts)&(clock<t.exit_ts)
        if inside.any():
            closes=pd.Series(frames[t.inst_id].close.to_numpy(),index=frames[t.inst_id].index+STEP)
            mark=closes.reindex(clock[inside],method='ffill')
            equity.loc[inside]+=t.notional*(t.side*(mark.to_numpy()/t.entry_px-1)-t.cost/2)
    equity=pd.concat([pd.Series([70.],index=[clock[0]-STEP]),equity])
    drawdown=equity/equity.cummax()-1
    n=len(trades);pnl=trades.pnl.astype(float)
    contrib=trades.groupby('inst_id').pnl.sum() if n else pd.Series(dtype=float)
    positive=contrib[contrib>0]
    win=pnl[pnl>0];loss=pnl[pnl<0]
    return dict(trades=n,wins=len(win),win_pct=len(win)/n*100 if n else np.nan,
        net_usdt=float(pnl.sum()),return_pct=float(pnl.sum()/70*100),
        marked_max_dd_pct=float(drawdown.min()*100),mean_hold_h=float(trades.hold_h.mean()) if n else np.nan,
        avg_win_bps=float(trades.loc[pnl>0,'net'].mean()*1e4) if len(win) else np.nan,
        avg_loss_bps=float(trades.loc[pnl<0,'net'].mean()*1e4) if len(loss) else np.nan,
        profit_factor=float(win.sum()/-loss.sum()) if len(loss) else np.nan,
        profitable_symbols=len(positive),biggest_positive_share=float(positive.max()/positive.sum()) if len(positive) else np.nan,
        traded_days=trades.entry_ts.dt.date.nunique() if n else 0),equity


def main(root):
    universe=pd.read_csv(root/'universe.csv')
    frames={};funding={};coverage=[]
    for inst in universe.instId:
        f=pd.read_csv(root/'market'/f'{inst}.csv');f.ts=pd.to_datetime(f.ts,utc=True)
        f=f.set_index('ts').sort_index();frames[inst]=add_features(f)
        raw=pd.DataFrame(json.loads((root/'market'/f'{inst}_funding.json').read_text()))
        raw['ts']=pd.to_datetime(pd.to_numeric(raw.fundingTime),unit='ms',utc=True)
        raw['rate']=pd.to_numeric(raw.realizedRate.where(raw.realizedRate.ne(''),raw.fundingRate))
        r=raw.drop_duplicates('ts').set_index('ts').rate.sort_index();funding[inst]=r
        coverage.append(dict(inst=inst,funding_first=str(r.index.min()),funding_last=str(r.index.max()),
                             max_funding_gap_h=r.index.to_series().diff().dt.total_seconds().max()/3600))
    signals=candidate_signals(frames);signals.to_csv(root/'signals.csv',index=False)
    summary=[];contrib=[];monthly=[]
    for strategy in ['cash','original_reversion','opening_range','trend_reclaim']:
        sig=signals[signals.strategy==strategy]
        for cost in [44,70]:
            for period,(start,end) in PERIODS.items():
                tr,rejected=execute(sig,frames,universe,funding,cost,start,end)
                stats,eq=metrics(tr,frames,start,end)
                stats.update(strategy=strategy,cost_bps=cost,period=period,rejected=len(rejected))
                summary.append(stats)
                tr.to_csv(root/f'trades_{strategy}_{period}_{cost}.csv',index=False)
                rejected.to_csv(root/f'rejected_{strategy}_{period}_{cost}.csv',index=False)
                eq.rename('equity').to_csv(root/f'equity_{strategy}_{period}_{cost}.csv')
                if len(tr):
                    for inst,g in tr.groupby('inst_id'):
                        contrib.append(dict(strategy=strategy,cost_bps=cost,period=period,inst=inst,
                                            trades=len(g),net_usdt=g.pnl.sum()))
                    for month,g in tr.groupby(tr.entry_ts.dt.strftime('%Y-%m')):
                        monthly.append(dict(strategy=strategy,cost_bps=cost,period=period,month=month,
                                            trades=len(g),net_usdt=g.pnl.sum(),win_pct=(g.net>0).mean()*100))
    summary=pd.DataFrame(summary);summary.to_csv(root/'summary.csv',index=False)
    pd.DataFrame(contrib).to_csv(root/'contributions.csv',index=False)
    pd.DataFrame(monthly).to_csv(root/'monthly.csv',index=False)
    (root/'funding_coverage.json').write_text(json.dumps(coverage,indent=2))
    gates=[]
    for strategy in ['original_reversion','opening_range','trend_reclaim']:
        val=summary[(summary.strategy==strategy)&(summary.period=='validation')&(summary.cost_bps==70)].iloc[0]
        recent=summary[(summary.strategy==strategy)&(summary.period=='recent')&(summary.cost_bps==70)].iloc[0]
        passed=bool(val.trades>=30 and val.net_usdt>0 and val.profitable_symbols>=3 and
                    val.biggest_positive_share<=.5 and recent.net_usdt>0)
        gates.append(dict(strategy=strategy,passes_prespecified_gate=passed))
    (root/'gates.json').write_text(json.dumps(gates,indent=2))
    print(summary.to_string(index=False));print(gates)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('results/strategy_replacement_20260911'))
    main(ap.parse_args().root)
