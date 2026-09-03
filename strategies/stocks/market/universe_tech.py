"""科技标的集合。

划线原则：**8-K 的驱动因素是不是科技业务本身**。COIN / HOOD / MSTR /
BMNR / CRCL 这几个在行业分类上算金融科技，但它们的公告实质是比特币
持仓和融资，方向由币价决定而不是由业务决定，属于另一条机制，放进来
只会污染样本。JNJ / KO / LLY / UNH 是医药消费，GME / DKNG 是零售博彩，
USAR 是稀土，都排除。

VRT / GEV / BE 留下：数据中心供电和散热已经是 AI 资本开支的直接组成，
它们的 8-K 内容读起来就是订单和产能，和半导体同源。
"""

TECH = {
    # 半导体与设备
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TSM", "ASML", "AMAT", "KLAC",
    "TER", "ON", "ARM", "ALAB", "MRVL", "SNDK", "COHR", "AAOI", "GLW", "CIEN",
    # 软件与互联网
    "MSFT", "GOOGL", "META", "AMZN", "AAPL", "ORCL", "CRM", "NOW", "ADBE",
    "SNOW", "PLTR", "CRWD", "OKTA", "ZM", "TWLO", "APP", "NFLX", "RDDT", "HIMS",
    # 硬件 / 基础设施 / 算力
    "DELL", "HPE", "SMCI", "IBM", "CSCO", "CRWV", "NBIS", "APLD", "IREN", "ONDS",
    "VRT", "GEV", "BE", "ROK", "SKHY", "CBRS",
    # 科技制造
    "TSLA", "RKLB", "ASTS", "ISRG", "RIVN",
}
