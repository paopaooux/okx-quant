"""Conditional execution ablation on frozen live trades; never places orders.

This holds the actual entry opportunities, sizes and fills fixed. Rejected entries
remain cash; freed slots do NOT generate replacement trades. It is not a portfolio
backtest or an estimate of the opportunity cost of an entry guard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def entry_checks(signal, quote_px, quote_ts, decision_ts, *, max_age_s=120,
                 max_adverse_bps=30, trigger_bps=600):
    """Only pre-order information; never inspect the eventual fill/outcome."""
    event = pd.Timestamp(signal['event_ts'])
    quote_ts, decision_ts = pd.Timestamp(quote_ts), pd.Timestamp(decision_ts)
    price = float(signal['close'])
    side = 1 if signal['side'] == 'long' else -1
    if quote_ts > decision_ts or event > quote_ts:
        raise ValueError('future quote or quote predates signal')
    if price <= 0 or quote_px <= 0:
        raise ValueError('nonpositive price')
    deviation = -side * abs(float(signal['observe_move_bps'])) / 1e4
    anchor = price / (1 + deviation)
    remaining_bps = -side * (quote_px / anchor - 1) * 1e4
    adverse_bps = side * (quote_px / price - 1) * 1e4
    age_s = (decision_ts - event).total_seconds()
    reasons = []
    if age_s > max_age_s:
        reasons.append('signal_age')
    if (decision_ts-quote_ts).total_seconds() > 5:
        reasons.append('quote_age')
    if remaining_bps < trigger_bps:
        reasons.append('displacement_below_trigger')
    if adverse_bps > max_adverse_bps:
        reasons.append('adverse_price_move')
    return dict(accepted=not reasons, reject_reason='|'.join(reasons),
                signal_age_s=age_s, quote_age_s=(decision_ts-quote_ts).total_seconds(),
                remaining_bps=remaining_bps, adverse_bps=adverse_bps)


def protective_exit(candles, entry_ts, entry_px, side, stop_width, observed_exit_ts,
                    observed_exit_px, *, slip_bps=10, activation_s=2):
    """Last-price-trigger scenario, 1m OHLC, gap-aware assumed market fill.

    Exclude partial activation and observed-exit minutes: their extrema may
    lie outside the actual holding interval.
    Return trigger interval, not an invented exact intraminute timestamp.
    Slippage is an assumption; OHLC does not reconstruct historical order books.
    Existing actual exit is retained if no earlier stop candle exists.
    """
    active = (entry_ts + pd.Timedelta(seconds=activation_s)).ceil('min')
    stop = entry_px * (1-side*stop_width)
    for ts, bar in candles.loc[(candles.index >= active) &
                               (candles.index + pd.Timedelta(minutes=1) <= observed_exit_ts)].iterrows():
        touch = float(bar.low) <= stop if side == 1 else float(bar.high) >= stop
        if not touch:
            continue
        base = min(float(bar.open), stop) if side == 1 else max(float(bar.open), stop)
        px = base * (1-side*slip_bps/1e4)
        upper = ts+pd.Timedelta(minutes=1)
        return dict(exit_px=px, exit_ts=upper, trigger_start=ts, trigger_end=upper,
                    reason='protective_stop', stop_px=stop)
    return dict(exit_px=observed_exit_px, exit_ts=observed_exit_ts,
                trigger_start=pd.NaT, trigger_end=pd.NaT,
                reason='observed_exit', stop_px=stop)


def aggregate(rows):
    result=[]
    for (scope,variant,slip),g in rows.groupby(['scope','variant','slip_bps'],sort=False):
        kept=g[g.accepted]
        pnl=kept.net_usdt
        result.append(dict(scope=scope,variant=variant,slip_bps=slip,
            opportunities=len(g),trades=len(kept),rejected=len(g)-len(kept),
            wins=int((pnl>0).sum()),win_pct=float((pnl>0).mean()*100) if len(kept) else np.nan,
            net_usdt=float(pnl.sum()),gross_usdt=float(kept.gross_usdt.sum()),
            fees_usdt=float(kept.fees_usdt.sum()),funding_usdt=float(kept.funding_usdt.sum()),
            mean_hold_h=float(kept.hold_h.mean()) if len(kept) else np.nan,
            avg_win_usdt=float(pnl[pnl>0].mean()),avg_loss_usdt=float(pnl[pnl<0].mean())))
    return pd.DataFrame(result)


def run(source: Path, out: Path):
    positions=pd.read_csv(source/'closed_positions.csv',dtype={'posId':str})
    for name in ['cTime','uTime']:
        positions[name]=pd.to_datetime(positions[name],utc=True)
    quotes={(x['posId'],pd.Timestamp(x['entry_ts'])):x
            for x in json.loads((out/'pretrade_quotes.json').read_text())}
    raw_bills=json.loads((source/'bills.json').read_text())
    funding=[(r['instId'],pd.to_datetime(int(r['ts']),unit='ms',utc=True),float(r['pnl']))
             for r in raw_bills if r['type']=='8' and r['subType'] in ['173','174']]
    candles={}
    for inst in {x['position']['inst_id'] for x in quotes.values()}:
        f=pd.read_csv(out/'market'/f'{inst}.csv')
        f.ts=pd.to_datetime(f.ts,utc=True)
        candles[inst]=f.set_index('ts').sort_index()
    records=[]; guard_rows=[]; sensitivity=[]
    for p in positions.itertuples():
        side=1 if p.direction=='long' else -1
        entry_px,exit_px=float(p.openAvgPx),float(p.closeAvgPx)
        actual_ret=side*(exit_px/entry_px-1)
        # Recover the fixed actual notional from exchange PNL and price change.
        # Independent check below against closeTotalPos for the six stock swaps.
        notional=float(p.pnl)/actual_ret
        assert notional>0
        base=dict(trade_id=f'{p.instId}:{p.cTime.isoformat()}',inst_id=p.instId,side=side,
            entry_ts=p.cTime,exit_ts=p.uTime,entry_px=entry_px,exit_px=exit_px,
            notional_usdt=notional,accepted=True,reject_reason='',gross_usdt=p.pnl,
            fees_usdt=p.fee,funding_usdt=p.fundingFee,net_usdt=p.realizedPnl,
            hold_h=p.hold_h,reason='observed_exit',trigger_start=pd.NaT,trigger_end=pd.NaT)
        meta=quotes.get((p.posId,p.cTime))
        variants={}
        if meta:
            assert np.isclose(notional,float(p.closeTotalPos)*entry_px)
            sig=meta['signal'];pos=meta['position']
            assert pos['policy_version']=='original_combo_v1'
            assert np.isclose(float(pos['entry_px']),entry_px)
            quote_ts=pd.to_datetime(meta['quote_ts'],unit='ms',utc=True)
            # Quote collected before submission; its timestamp is the decision
            # proxy. Exact order submission time was not archived.
            assert quote_ts<=p.cTime and (p.cTime-quote_ts).total_seconds()<5
            check=entry_checks(sig,float(meta['quote_px']),quote_ts,quote_ts)
            guard_rows.append(dict(trade_id=base['trade_id'],inst_id=p.instId,
                entry_ts=p.cTime,quote_ts=quote_ts,quote_px=meta['quote_px'],**check))
            for age in [60,120,300,600]:
                c=entry_checks(sig,float(meta['quote_px']),quote_ts,quote_ts,max_age_s=age)
                sensitivity.append(dict(trade_id=base['trade_id'],age_limit_s=age,**c,
                                        baseline_net_usdt=p.realizedPnl))
            for slip in [0,10,30,60]:
                protection=protective_exit(candles[p.instId],p.cTime,entry_px,side,
                    float(pos['width']),p.uTime,exit_px,slip_bps=slip)
                protected={**base,**protection}
                if protection['reason']=='protective_stop':
                    newpx=protection['exit_px'];ex=protection['exit_ts']
                    fee_rate=-float(p.fee)/(notional*(1+exit_px/entry_px))
                    assert np.isclose(fee_rate,.0005)
                    gross=notional*side*(newpx/entry_px-1)
                    fees=-fee_rate*notional*(1+newpx/entry_px)
                    relevant=[(t,amt) for inst,t,amt in funding if inst==p.instId and p.cTime<=t<=p.uTime]
                    assert np.isclose(sum(amt for _,amt in relevant),p.fundingFee)
                    assert not any(protection['trigger_start']<=t<=ex for t,_ in relevant), 'funding within ambiguous trigger minute'
                    fund=sum(amt for t,amt in relevant if t<protection['trigger_start'])
                    protected.update(gross_usdt=gross,fees_usdt=fees,funding_usdt=fund,
                                     net_usdt=gross+fees+fund,hold_h=(ex-p.cTime).total_seconds()/3600)
                for label,record in [('baseline',base),('protection',protected),
                                     ('entry_check',base),('both',protected)]:
                    r=dict(record)
                    if label in ['entry_check','both'] and not check['accepted']:
                        r.update(accepted=False,reject_reason=check['reject_reason'],exit_ts=pd.NaT,
                                 gross_usdt=0,fees_usdt=0,funding_usdt=0,net_usdt=0,hold_h=np.nan,
                                 reason='entry_rejected',exit_px=np.nan,trigger_start=pd.NaT,trigger_end=pd.NaT)
                    variants[(label,slip)]=r
                    records.append(dict(r,scope='new_policy_6',variant=label,slip_bps=slip))
        # Whole-account bridge keeps the 16 legacy trades untouched, explicitly.
        for slip in [0,10,30,60]:
            for label in ['baseline','protection','entry_check','both']:
                records.append(dict(variants.get((label,slip),base),scope='all22_legacy_unchanged',
                                    variant=label,slip_bps=slip))
    rows=pd.DataFrame(records);rows.to_csv(out/'trades.csv',index=False)
    summary=aggregate(rows);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(guard_rows).to_csv(out/'entry_checks.csv',index=False)
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(out/'entry_sensitivity_trades.csv',index=False)
    sens=sensitivity.groupby('age_limit_s').apply(lambda g:pd.Series(dict(
        accepted=int(g.accepted.sum()),net_usdt=g.loc[g.accepted,'baseline_net_usdt'].sum(),
        wins=int((g.loc[g.accepted,'baseline_net_usdt']>0).sum()))),include_groups=False)
    sens.to_csv(out/'entry_sensitivity_summary.csv')
    main=rows[(rows.scope=='new_policy_6')&(rows.slip_bps==10)]
    pivot=main.pivot(index=['trade_id','inst_id'],columns='variant',values='net_usdt')
    pivot.to_csv(out/'paired_net_usdt.csv')
    manifest=dict(source=str(source),window_start=str(positions.cTime.min()),
        window_end='2026-09-11T07:35:00Z',new_policy_start='2026-09-07T07:56:41Z',
        max_age_s=120,max_adverse_bps=30,trigger_bps=600,main_stop_slip_bps=10,
        stop_slip_sensitivity_bps=[0,10,30,60],activation_s=2,
        scope='Conditional on actual completed trades, actual entry fills and sizes. No replacement entries. Legacy 16 unchanged.',
        caveats=['Last-price trigger; tick size and historical order book execution not reconstructed.',
                 'Activation minute excluded to avoid using pre-entry extrema; may miss an immediate stop.',
                 'Partial observed-exit minute excluded: extrema could occur after actual exit; observed exit retained if earlier complete bars never touch.',
                 'Stop time reported as 1m interval; hold uses upper bound.',
                 'Pretrade ticker timestamp is decision-time proxy; quote precedes fill by less than 5s.',
                 'Fees use actual inferred 5bp per side; funding uses actual cash events before hypothetical exit.',
                 'Existing sample inspected before choosing guard thresholds: in-sample exploratory evidence, not out-of-sample validation.',
                 'Filtering changes available slots and quota; future replacement opportunities unmodeled.',
                 'Current open trades are not included; earlier 16 trades not retroactively assigned new policy.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary[summary.slip_bps==10].to_string(index=False))
    print(pd.DataFrame(guard_rows).to_string(index=False))
    print('Age sensitivity (all price checks unchanged):\n',sens)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/live_audit_20260911'))
    parser.add_argument('--output',type=Path,default=Path('results/execution_backtest_20260911'))
    args=parser.parse_args();run(args.source,args.output)
