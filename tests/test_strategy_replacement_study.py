import numpy as np
import pandas as pd
import pytest

from scripts.analysis.strategy_replacement_study import add_features, path_exit


def bars():
    return pd.DataFrame(dict(open=[100.,100.,101.],high=[101.,104.,110.],
        low=[99.,96.,90.],close=[100.,101.,105.],volume_quote=[10000.]*3),
        index=pd.date_range('2026-08-03T14:00Z',periods=3,freq='15min'))


def test_same_bar_stop_and_target_uses_stop():
    f=bars()
    result=path_exit(f,f.index[0],1,.03,.03,f.index[2])
    assert result[1]==97
    assert result[3]=='stop'


def test_timeout_bar_extrema_are_not_used_before_open_exit():
    f=bars(); f.loc[f.index[1],['high','low']]=[102,98]
    result=path_exit(f,f.index[0],1,.03,.03,f.index[2])
    assert result[1]==101
    assert result[3]=='deadline'


def test_gap_stop_uses_adverse_open_and_missing_path_is_rejected():
    f=bars(); f.loc[f.index[1],'open']=95
    assert path_exit(f,f.index[0],1,.03,.03,f.index[2])[1]==95
    assert path_exit(f.drop(f.index[1]),f.index[0],1,.03,.03,f.index[2]) is None


def test_features_are_prefix_invariant():
    f=pd.concat([bars().reset_index(drop=True)]*50,ignore_index=True)
    f.index=pd.date_range('2026-08-01',periods=len(f),freq='15min',tz='UTC')
    before=add_features(f.iloc[:110]);full=add_features(f)
    pd.testing.assert_frame_equal(before,full.iloc[:110])
