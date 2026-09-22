import pandas as pd
import pytest

from scripts.analysis.execution_guard_study import entry_checks, protective_exit


def stamp(s):
    return pd.Timestamp('2026-09-10T'+s+'Z')


def signal(side='short', close=106):
    return dict(event_ts=stamp('00:00:00'), side=side, close=close,
                observe_move_bps=600)


def test_revalidate_same_direction_and_only_known_quotes():
    # Original anchor is 100. A drop to 105 no longer supports the 6% short.
    r=entry_checks(signal(),105,stamp('00:01:00'),stamp('00:01:01'))
    assert not r['accepted']
    assert 'displacement_below_trigger' in r['reject_reason']
    assert r['remaining_bps']==pytest.approx(500)
    with pytest.raises(ValueError,match='future quote'):
        entry_checks(signal(),107,stamp('00:01:02'),stamp('00:01:01'))
    # A move to the opposite side of the anchor is not a valid same-direction entry.
    assert not entry_checks(signal(),94,stamp('00:01:00'),stamp('00:01:01'))['accepted']


def test_long_check_can_accept_improved_price_but_blocks_expired_signal():
    s=signal('long',94)
    assert entry_checks(s,93,stamp('00:01:00'),stamp('00:01:01'))['accepted']
    r=entry_checks(s,93,stamp('00:03:00'),stamp('00:03:01'))
    assert r['reject_reason']=='signal_age'


def candles(rows):
    return pd.DataFrame(rows,columns=['ts','open','high','low']).assign(
        ts=lambda d:d.ts.map(stamp)).set_index('ts')


def test_stop_excludes_partial_entry_minute_and_preserves_observed_exit():
    f=candles([('00:00:00',100,101,90),('00:01:00',100,101,99)])
    r=protective_exit(f,stamp('00:00:20'),100,1,.03,stamp('00:02:00'),101)
    assert r['reason']=='observed_exit'
    assert r['exit_px']==101


@pytest.mark.parametrize('side,opening,high,low,expected',[
    (1,95,98,94,95*.999),(-1,105,106,102,105*1.001)])
def test_gap_execution_is_worse_than_barrier_not_magically_filled_at_stop(side,opening,high,low,expected):
    f=candles([('00:01:00',opening,high,low)])
    r=protective_exit(f,stamp('00:00:20'),100,side,.03,stamp('00:03:00'),100,slip_bps=10)
    assert r['exit_px']==pytest.approx(expected)
    assert r['trigger_start']==stamp('00:01:00')
    assert r['trigger_end']==stamp('00:02:00')


def test_bar_after_actual_exit_cannot_create_a_hypothetical_early_stop():
    f=candles([('00:03:00',95,98,94)])
    r=protective_exit(f,stamp('00:00:20'),100,1,.03,stamp('00:02:00'),101)
    assert r['reason']=='observed_exit'


def test_partial_exit_minute_cannot_use_extrema_after_actual_close():
    f=candles([('00:01:00',100,101,90)])
    r=protective_exit(f,stamp('00:00:20'),100,1,.03,stamp('00:01:30'),99)
    assert r['reason']=='observed_exit'
    assert r['exit_px']==99


def test_complete_minute_ending_at_actual_close_is_eligible():
    f=candles([('00:01:00',100,101,96)])
    r=protective_exit(f,stamp('00:00:20'),100,1,.03,stamp('00:02:00'),99)
    assert r['reason']=='protective_stop'
    assert r['exit_px']==pytest.approx(97*.999)
