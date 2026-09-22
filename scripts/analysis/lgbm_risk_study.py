"""Frozen LightGBM threshold/sizing audit. All tools and files are research-only."""
from pathlib import Path
import heapq
import json
import sqlite3

import numpy as np
import pandas as pd

ROOT=Path('results/lgbm_risk_study_20260911')
BAR=900000
SYMBOLS=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','LINKUSDT']


def raw_exit(ts,opening,high,low,signal_ts,width,side,slip_bps=10):
    """Next-open entry, 48 held bars, timeout BEFORE timeout bar extremes."""
    entry_ts=signal_ts+BAR
    i=np.searchsorted(ts,entry_ts);j=i+48
    if j>=len(ts) or ts[i]!=entry_ts or not np.all(np.diff(ts[i:j+1])==BAR):
        return None
    entry=opening[i];stop=entry*(1-side*width);target=entry*(1+side*width)
    for k in range(i,j):
        stopped=low[k]<=stop if side==1 else high[k]>=stop
        targeted=high[k]>=target if side==1 else low[k]<=target
        if stopped:
            px=(min(opening[k],stop) if side==1 else max(opening[k],stop))*(1-side*slip_bps/1e4)
            return dict(entry_px=entry,exit_px=px,exit_ms=int(ts[k]+BAR),
                        gross=side*(px/entry-1),reason='stop')
        if targeted:
            return dict(entry_px=entry,exit_px=target,exit_ms=int(ts[k]+BAR),gross=width,reason='target')
    px=opening[j]
    return dict(entry_px=entry,exit_px=px,exit_ms=int(ts[j]),gross=side*(px/entry-1),reason='timeout')


def admit(candidates):
    active={};accepted=[]
    ordered=candidates.sort_values(['entry_ms','strength','symbol'],ascending=[True,False,True])
    for r in ordered.itertuples():
        active={s:t for s,t in active.items() if t>r.entry_ms}
        if r.symbol in active or len(active)>=5:
            continue
        active[r.symbol]=r.exit_ms;accepted.append(r.Index)
    return candidates.loc[accepted].reset_index(drop=True)


def curve(trades,prices,weight,cost,start,end):
    # Sizing uses realized equity as the deployed budget; reserve principal cap.
    # PNL below is marked on completed 15m closes, not just realized exits.
    clock=np.arange(start//BAR*BAR,(end//BAR+1)*BAR,BAR,dtype=np.int64)
    realized_equity=1.;pending=[];sizes=[];reserved=0.
    t=trades.sort_values('entry_ms').copy()
    for r in t.itertuples():
        while pending and pending[0][0]<=r.entry_ms:
            _,idx,notional,net=heapq.heappop(pending)
            realized_equity+=notional*net;reserved-=notional
        size=max(0,min(realized_equity*weight,realized_equity-reserved))
        sizes.append(size);reserved+=size
        heapq.heappush(pending,(r.exit_ms,r.Index,size,r.gross-cost/1e4))
    t['notional']=sizes;t['net']=t.gross-cost/1e4;t['pnl']=t.notional*t.net
    changes=np.zeros(len(clock)+1);mark=np.zeros(len(clock))
    for r in t.itertuples():
        i=np.searchsorted(clock,r.entry_ms);j=np.searchsorted(clock,r.exit_ms)
        changes[min(j,len(clock))]+=r.pnl
        if i<j:
            p=prices[r.symbol];pidx=np.searchsorted(p['ts']+BAR,clock[i:j],side='right')-1
            assert (pidx>=0).all()
            mark[i:j]+=r.notional*(r.side*(p['close'][pidx]/r.entry_px-1)-cost/2e4)
    eq=1+np.cumsum(changes[:-1])+mark
    dd=eq/np.maximum.accumulate(np.r_[1.,eq])[1:]-1
    n=len(t);win=t.net>0
    recovery=[];under=None
    for stamp,d in zip(clock,dd):
        if d< -1e-12 and under is None:under=stamp
        elif d>=-1e-12 and under is not None:
            recovery.append((stamp-under)/86400000);under=None
    if under is not None:recovery.append((clock[-1]-under)/86400000)
    stats=dict(trades=n,wins=int(win.sum()),win_pct=float(win.mean()*100) if n else None,
        return_pct=float(t.pnl.sum()*100),marked_max_dd_pct=float(dd.min()*100),
        hold_h=float(((t.exit_ms-t.entry_ms)/3600000).mean()) if n else None,
        avg_net_bps=float(t.net.mean()*1e4) if n else None,
        profit_factor=float(t.loc[win,'net'].sum()/-t.loc[~win,'net'].sum()) if n and (~win).any() else None,
        longest_underwater_days=max(recovery,default=0),unrecovered=bool(dd[-1]<-1e-12),
        notional_cap_clips=int((t.notional<=0).sum()))
    return stats,t,pd.DataFrame({'ts':pd.to_datetime(clock,unit='ms',utc=True),'equity':eq})


def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    oos=pd.read_csv('results/crypto/oos_dir_c_roll730.csv.gz')
    prices={}
    for symbol in SYMBOLS:
        f=pd.read_csv(f'data/klines/{symbol}.csv.gz',usecols=['ts','open','high','low','close']).sort_values('ts').drop_duplicates('ts')
        prices[symbol]={c:f[c].to_numpy() for c in f}
    candidates=[];thresholds=[];missing=[]
    for tail in [.01,.005,.001]:
        hi=f'hi_{tail}';lo=f'lo_{tail}'
        for fold,g in oos.groupby('fold'):
            thresholds.append(dict(tail=tail,fold=fold,hi=g[hi].iloc[0],lo=g[lo].iloc[0],
                raw_signals=int(((g.p_up>=g[hi])|(g.p_up<=g[lo])).sum())))
        sub=oos[(oos.p_up>=oos[hi])|(oos.p_up<=oos[lo])]
        for r in sub.itertuples(index=False,name=None):
            row=dict(zip(sub.columns,r));sym=row['symbol'];p=prices[sym]
            if not np.isfinite(row['held_c']) or not np.isfinite(row['exit_ret_c']):continue
            side=1 if row['p_up']>=row[hi] else -1
            entry=int(row['ts'])+BAR
            actual=raw_exit(p['ts'],p['open'],p['high'],p['low'],int(row['ts']),row['width_c'],side)
            if actual is None:
                missing.append(dict(symbol=sym,signal_ts=row['ts'],tail=tail));continue
            base=dict(symbol=sym,entry_ms=entry,fold=row['fold'],side=side,tail=tail,
                strength=abs(row['p_up']-.5),signal_ms=int(row['ts']),width=row['width_c'])
            candidates.append(dict(base,exit_mode='guarded',**actual))
            gross=side*row['exit_ret_c']
            candidates.append(dict(base,exit_mode='label_reference',entry_px=actual['entry_px'],
                exit_px=actual['entry_px']*(1+side*gross),gross=gross,
                exit_ms=int(row['ts']+(row['held_c']+1)*BAR),reason='archived_label'))
    c=pd.DataFrame(candidates);c.to_csv(ROOT/'candidates.csv',index=False)
    pd.DataFrame(thresholds).to_csv(ROOT/'thresholds.csv',index=False)
    pd.DataFrame(missing).to_csv(ROOT/'missing_paths.csv',index=False)
    print('candidates',len(c),'missing',len(missing),flush=True)
    start=int(oos.ts.min()+BAR);end=int(oos.ts.max()+49*BAR)
    windows={'all_oos':(start,end),'year2026':(int(pd.Timestamp('2026-01-01T00:00Z').timestamp()*1000),end)}
    for fold,g in oos.groupby('fold'):
        windows[f'fold{fold}']=(int(g.ts.min()+BAR),int(g.ts.max()+BAR+BAR))
    summary=[];recent_selected=[]
    for (tail,mode),g in c.groupby(['tail','exit_mode']):
        for window,(a,b) in windows.items():
            eligible=g[(g.entry_ms>=a)&(g.exit_ms<=b)]
            tr=admit(eligible)
            for cost in [10,20]:
                for weight in [.2,.1]:
                    stats,t,eq=curve(tr,prices,weight,cost,a,b)
                    tag=f'{window}_{tail}_{mode}_{cost}_{weight}'
                    summary.append(dict(stats,window=window,tail=tail,exit_mode=mode,cost_bps=cost,weight=weight))
                    t.to_csv(ROOT/f'trades_{tag}.csv',index=False)
                    eq.set_index('ts').equity.resample('1D').last().to_csv(ROOT/f'equity_daily_{tag}.csv')
        print('finished',tail,mode,flush=True)
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
    # Actual live input scores; never backfill Binance scores into live OKX history.
    db=sqlite3.connect('file:data/okx_demo.sqlite3?mode=ro',uri=True)
    live=pd.read_sql_query("SELECT ts,symbol,bar_ts,p_up,signal_side FROM strategy_snapshots WHERE ts>='2026-09-07T07:56:41' AND ts<'2026-09-11T07:35' AND asset_type='crypto'",db)
    live=live[live.p_up.notna()]
    rows=[]
    for tail in [.01,.005,.001]:
        f=oos[oos.fold==5].iloc[0];hi=f[f'hi_{tail}'];lo=f[f'lo_{tail}']
        fire=live[(live.p_up>=hi)|(live.p_up<=lo)]
        rows.append(dict(tail=tail,snapshots=len(live),unique_symbol_bars=len(live.drop_duplicates(['symbol','bar_ts'])),
            eligible_snapshots=len(fire),eligible_symbol_bars=len(fire.drop_duplicates(['symbol','bar_ts'])),
            min_p=live.p_up.min(),max_p=live.p_up.max(),hi=hi,lo=lo))
    pd.DataFrame(rows).to_csv(ROOT/'live_signal_check.csv',index=False)
    # Current account-sized implementation check independent from idealized historical curves.
    u=pd.read_csv('results/crypto_replacement_20260911/universe.csv')
    lots=[]
    for r in u.itertuples():
        sym=r.instId.replace('-USDT-SWAP','USDT');price=float(prices[sym]['close'][-1])
        for weight in [.2,.1]:
            target=70*weight;unit=float(r.ctVal)*price;lot=float(r.lotSz)
            qty=np.floor((target/unit)/lot+1e-10)*lot
            lots.append(dict(symbol=sym,reference_price=price,reference_ts=int(prices[sym]['ts'][-1]),weight=weight,
                target_usdt=target,qty=qty,minimum_qty=float(r.minSz),notional_usdt=qty*unit,
                executable=bool(qty>=float(r.minSz))))
    pd.DataFrame(lots).to_csv(ROOT/'small_account_lots.csv',index=False)
    print(pd.DataFrame(summary).query("window=='all_oos' and cost_bps==20").to_string(index=False))
    print('live',rows)


if __name__=='__main__':main()
