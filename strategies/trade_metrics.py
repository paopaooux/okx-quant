"""Common trade diagnostics; returns are decimal, net of trading costs."""

import numpy as np
import pandas as pd
from scipy.stats import t


METRIC_NOTE = (
    "SQN 按逐笔净收益率计算：sqrt(N) × 均值 / 样本标准差（非 R 倍数口径）。"
    "Mean profit p-value 为均值等于零的双侧单样本 t 检验；"
    "假定交易独立且均值的 t 检验近似适用，未校正交易相关性及多重策略筛选，"
    "不能解读为随机运气的概率或策略质量保证。"
    "做多/做空盈利（%）为各方向逐笔净收益率之和 × 100，包含亏损单，"
    "不是利润占比，也不是考虑仓位和复利的账户收益。"
    "无交易的方向累计值为 0；不足两笔或零方差时 SQN 和 p 值不可用。"
)


def trade_metrics(trades: pd.DataFrame) -> dict:
    """Report descriptive direction totals and an unadjusted two-sided t-test."""
    result = dict(sqn=np.nan, mean_profit_pvalue=np.nan,
                  long_profit_pct=0.0, short_profit_pct=0.0)
    if trades.empty:
        return result
    net = trades["net"].to_numpy(dtype=float)
    if not np.isfinite(net).all():
        raise ValueError("trade metrics require finite net returns")
    if not trades["side"].isin([1, -1]).all():
        raise ValueError("trade metrics require side +1 or -1")
    if len(net) > 1:
        sd = net.std(ddof=1)
        if sd > 0:
            result["sqn"] = float(np.sqrt(len(net)) * net.mean() / sd)
            result["mean_profit_pvalue"] = float(2 * t.sf(abs(result["sqn"]), df=len(net) - 1))
    for side, name in ((1, "long"), (-1, "short")):
        result[f"{name}_profit_pct"] = float(net[trades.side.to_numpy() == side].sum() * 100)
    return result
