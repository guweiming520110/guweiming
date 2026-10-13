# 数字彩票 AI 分析系统

大乐透(dlt) / 双色球(ssq) / 快乐8(kl8) 数据分析与预测系统。基于几何图案引擎 + 机器学习融合，支持数据爬取、预测、回测、走势图、钉钉推送。

## ⚠️ 重要声明（请先阅读）

**彩票开奖是均匀随机事件，任何基于历史数据的模型都无法可靠预测开奖号码。**
本项目实测（见 `evaluate.py` 的 walk-forward 回测 + 统计检验）：
- dlt / ssq：所有策略（图案引擎 / ML / 热号 / 冷号）均 **≈随机水平**（z<1.96，不显著）
- kl8：图案引擎 **显著优于随机**（z≈+2.15），ML 集合显著劣于随机（z≈-2.06）

系统价值在于**诚实的评估与风控**：自适应融合会把低于随机的模型权重归零，避免"负学习"拉低结果。请理性对待预测结果，切勿据此重注。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键全自动运行（更新数据 + 验证 + 预测 + 生成走势图 + 钉钉推送）
python run_all.py
python run_all.py --no-notify --skip-charts   # 无推送、跳过图表

# 数据补全
python fill_dlt_data.py    # 大乐透
python fill_ssq_data.py    # 双色球
python kl8_data.py         # 快乐8

# 预测
python predict.py --type dlt [--ml]        # 大乐透（--ml 启用 ML 融合）
python predict.py --type ssq [--ml]        # 双色球
python predict.py --kl8                    # 快乐8

# 走势图
python generate_charts.py --type dlt --predict-front 3,10,12,20,25 --predict-back 1,9
python generate_charts.py --kl8 --predict-front 7,46,21,8,39,42

# 模型训练
python ml_models.py                 # RF/XGB（dlt+ssq）
python deep_learning_models.py      # LSTM/Transformer（需 tensorflow，耗时）
python auto_tune.py --trials 50     # 图案引擎自动调参
```

## 诚实评估

```bash
python evaluate.py                  # 全部彩种 walk-forward 回测 + 随机基线对照
python evaluate.py --type kl8       # 单彩种
```

输出各策略平均命中/期、z 值、结论（显著优于/≈/劣于随机）。**不显著优于随机的模型不应进入融合。**

## 模型架构

| 模块 | 说明 |
|---|---|
| `pattern_learner/` | 几何图案引擎（垂直/水平/对角/方块等 30+ 图案），评分含时间衰减、稀有度、冷热号 |
| `ml_models.py` | `LotteryPredictor`（位置独立 RF/XGB）+ `SetNumberPredictor`（集合预测，推荐） |
| `deep_learning_models.py` | LSTM/Transformer 序列预测（需单独训练） |
| `predict.py` | 融合入口：图案引擎评分 + ML 概率，自适应权重（低于随机归零） |
| `evaluate.py` | walk-forward 回测 + 超几何 z 检验评估基准 |
| `auto_tune.py` | Optuna 自动调参（独立留出集防乐观偏差） |

## 目录结构

```
Chao-Ji-Da-Le-Tou-AIYu-Ce/     # 主代码（git 子模块）
├── config.py                  # 统一配置（路径/凭据/爬虫参数）
├── lottery_types.py           # 彩种配置 + 期号推算（跨年）
├── schema.py / db_utils.py    # 数据库表结构 + 连接管理
├── fill_dlt_data.py 等        # 三彩种爬虫
├── charts/                    # 走势图（复合/趋势/频率/网络）
├── pattern_learner/           # 几何图案引擎
├── tests/                     # pytest 回归测试
└── logs/                      # 轮转日志
../lottery.db                  # SQLite 权威数据（根目录）
```

## 数据源

- 大乐透/双色球：500.com 区间接口（按年拉取）
- 快乐8：官方 cwl.gov.cn 接口（主源，日更），失败回退 500.com

## 测试

```bash
python -m pytest tests/ -v
```

覆盖：跨年期号推算、空库崩溃、图案去重、集合预测、特征对齐、奖级计算、自适应融合权重等关键修复点。

## 已知限制

- `yearly_max` 期数表需在跨年时补充（见 `lottery_types.py` 的 `YEARLY_ACTUAL_PERIODS`）
- 深度学习模型需 M4 归一化后重训（`python deep_learning_models.py`）
- 数据源改版可能导致爬虫失效，建议定期跑 `evaluate.py` 与线上解析对照

## 配置

复制 `.env.example` 为 `.env` 并填入钉钉机器人凭据（已 gitignore）：
```
DINGTALK_TOKEN=your_access_token_here
DINGTALK_SECRET=your_secret_here
```
