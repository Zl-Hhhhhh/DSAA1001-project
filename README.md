# DSAA1001：Chicago Food Inspection Data Analysis

课程项目：探索历史食品检查数据，并用检查开始前已知的信息预测 Fail。
项目说明见 [项目 PDF](docs/DSAA1001_Project.pdf)。

## 目录

~~~text
DSAA1001-project/
├── config/                           # 数据范围、标签定义与 Assi1 时间划分
├── data/
│   ├── raw/                          # 原始 CSV.gz、下载记录、官方元数据；应提交
│   ├── processed/                    # 清洗历史表、带标签表、校验清单；应提交
│   └── test/                         # 仅放教学团队提供的隐藏测试输入
├── docs/                             # 课程 PDF
├── notebooks/
│   ├── 00_data_preparation.ipynb      # 离线数据准备及核查
│   ├── 01_assessment1_eda.ipynb       # Assi1 主提交：8 组图、质量分析、建模计划
│   └── examples/01_example_eda.ipynb  # 已适配目录的课程示例；不是完整作业
├── scripts/
│   ├── download_data.py               # 联网下载，默认整个允许的历史范围
│   ├── prepare_data.py                # 离线清洗、标签、质量报告和校验
│   ├── ass1_utils.py                  # Assi1 读取、统计与严格历史特征
│   └── run_assessment1.py              # 执行并保存 Assi1 Notebook 输出
├── tests/                            # 数据边界及泄漏防护测试
├── reports/                          # 质量报告、字段可用性、统计及 figures/
├── outputs/                          # 后续 predictions.csv 等输出
├── requirements.in                   # 直接依赖
├── requirements.txt                  # 经安装验证的精确依赖版本
└── AI-USAGE.md                        # AI 使用记录
~~~

## 运行

建议 Python 3.13；下载和准备脚本仅使用标准库，不依赖 pandas。

已有原始快照时，以下命令均无需联网：

~~~powershell
python scripts/prepare_data.py
python scripts/prepare_data.py --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

运行 Notebook 前创建环境并安装依赖；这些步骤需要联网，属于环境安装，
Notebook 自身的 Run All 不会下载数据或安装依赖：

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

在 VS Code / Jupyter 中选择该环境，打开
[数据准备 Notebook](notebooks/00_data_preparation.ipynb)，执行 Run All。
Notebook 支持从仓库根目录或 notebooks/ 及其子目录启动。

需要在课程截止日前更新原始快照时，显式运行：

~~~bash
python scripts/download_data.py --overwrite
~~~

默认从 2016-01-01 下载到“运行当天 UTC 日期和 2026-10-30 中较早的日期”。
指定日期也不能越过当天或项目截止日。截止日后下载器会拒绝运行；
最终评估使用已保存的截止日前快照。下载成功后重新运行离线准备脚本。
下载记录保留请求范围、实际范围、时间、官方更新时间、行数和 SHA-256。

## 当前数据快照

<!-- DATA-SNAPSHOT:START -->

- 数据来源：[Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5)（4ijn-s7e5）。
- 下载时间（UTC）：**2026-10-12T07:59:09+00:00**。
- 本次请求范围：2016-01-01 至 2026-10-12。
- 实际记录日期：**2016-01-04 至 2026-10-09**。
- 原始记录：**197,882**；去除完全重复后：**197,882**。
- 有标签记录：**167,538**；无 Pass/Fail 判断的历史记录：**30,344**。
- 标签 0：**129,675**；标签 1：**37,863**（Fail 占有标签记录 **22.60%**）。
- 重复 Inspection ID：0；完全重复并移除：0。
- 多次检查的初步店铺分组：25,757。
- 请求截止日期不代表数据已完整上报；此快照不包含尚未公开的记录，不能称为截至 10 月 30 日的完整数据。

原始字段缺失率（空值或仅空白字符）：

| 原始字段 | 缺失数 | 缺失率 |
|---|---:|---:|
| Inspection ID | 0 | 0.00% |
| DBA Name | 0 | 0.00% |
| AKA Name | 500 | 0.25% |
| License # | 7 | 0.00% |
| Facility Type | 1,030 | 0.52% |
| Risk | 39 | 0.02% |
| Address | 1 | 0.00% |
| City | 125 | 0.06% |
| State | 73 | 0.04% |
| Zip | 12 | 0.01% |
| Inspection Date | 0 | 0.00% |
| Inspection Type | 1 | 0.00% |
| Results | 0 | 0.00% |
| Violations | 56,409 | 28.51% |
| Latitude | 689 | 0.35% |
| Longitude | 689 | 0.35% |
| Location | 689 | 0.35% |

<!-- DATA-SNAPSHOT:END -->

## 数据使用约定

- raw/ 保留全部原始字段，processed/inspections.csv.gz 是含结果的历史分析表。
- labelled_inspections.csv.gz 只保留有有效标签的记录，不包含本次 results / violations。
  它也保留编号等辅助字段，不能把除 label 外的所有列直接当作模型特征；
  默认候选特征及各列用途见 reports/feature_availability.csv。
- Fail=1；Pass 和 Pass w/ Conditions=0。其他结果不参与有标签训练/评估，
  但保留在历史表中，供严格早于当前检查的历史特征使用。
- 初步 establishment_key 优先使用有效 License #，缺失时回退到名称和地址，
  信息不足时使用单条检查键。后续需核查许可证跨地址和实体识别问题。
- 不做随机训练/测试划分，不在全数据上拟合填补、缩放或编码器。
  这些决策应在 EDA 后按时间顺序完成。
- data/test/ 仅接受教学团队提供的输入；不查询隐藏测试期结果。
- 所有项目输入、数据来源说明及必要输出应加入 Git；不要忽略整个 data/。

## Assessment 1

主提交文件为 [01_assessment1_eda.ipynb](notebooks/01_assessment1_eda.ipynb)，使用英文撰写。
其中包含 8 组探索图、12 项有证据的数据质量分析、字段可用性审计和约一页建模计划。
建模计划已完整嵌入 Notebook，另有 [Markdown 参考版](docs/assessment1_modelling_plan.md)。

在项目 .venv 环境中直接 Run All，或从仓库根目录执行：

~~~powershell
.\.venv\Scripts\python.exe scripts/run_assessment1.py
~~~

图表保存在 reports/figures/assessment1/，支持统计表保存在 reports/assessment1/。
本次 Assi1 固定使用 2026-10-12 下载的快照。更新原始快照后，需要重新核查分析文字；
Notebook 会校验快照哈希，避免新数据与旧结论混用。
结果分析仅使用 2016-2025 年；2026 年用于后续本地留出评估，本次仅统计其记录数量和店铺重叠。

## Assessment 2

离线核查现有数据：`.\.venv\Scripts\python.exe scripts/ass2_audit_data.py`。
结果直接在终端显示，固定数据版本见 [assessment2_data.json](config/assessment2_data.json)。

基础与历史特征接口在 [ass2_features.py](scripts/ass2_features.py)。执行
`.\.venv\Scripts\python.exe scripts/ass2_features.py` 可在内存中构造训练期、验证期特征，
并比较两种店铺匹配方式；结果只显示在终端。

从仓库根目录调用：

~~~python
import sys
sys.path.insert(0, "scripts")
from ass2_features import load_history, build_features

history = load_history()
train_rows = history.loc[history.inspection_date.le("2024-12-31")]
valid_rows = history.loc[history.inspection_date.between("2025-01-01", "2025-12-31")]

train = build_features(train_rows, history=train_rows, identity="license")
valid = build_features(
    valid_rows, history=train_rows, batch_start="2025-01-01", identity="license"
)
# 先用全部访问构造历史，再筛选有标签的行用于训练或评分。
keep = train.y.notna()
X_train, y_train = train.X.loc[keep], train.y.loc[keep]
~~~

返回值包含 `X`（模型特征）、`y`（可为0或1或空的标签）和 `meta`（ID、日期、店铺键、历史截止时间及已知店铺标记）。
三者按输入行顺序对齐，并重置索引。验证数据同样用 `valid.y.notna()` 筛选评分行。
接口同时接受原始字段名和清洗字段名；预测输入可以没有 Results、Violations 或 label。

| 特征(`X`) | 字段与含义 |
|---|---|
| 基础类别 | facility_type、risk、inspection_type、zip、inspection_month、inspection_weekday；月份和星期作为类别，星期一为 0 |
| 基础数值 | latitude、longitude；缺失或非法坐标保留为 NaN，留给训练流程处理 |
| 历史次数 | prior_visits、prior_judgements、prior_failures；分别为此前访问数、有效判断数和失败数 |
| 历史失败比例 | prior_fail_rate = prior_failures / prior_judgements；没有有效判断时为 NaN |
| 最近结果 | previous_outcome：最近有判断日期的非 Fail、Fail、同日混合结果或没有此前判断 |

默认得到 8 个基础特征和 5 个历史特征。`include_history=False` 仅返回基础特征。
`identity="license_address"` 可切换为许可证加地址分组，地址匹配同时考虑城市、州和 ZIP；
有效许可证缺少地址时使用单条检查键，不把缺失地址统一合并。
没有有效许可证时，沿用名称和地址等信息的回退规则。

训练行只使用更早日期的历史。验证或预测必须传入 `history` 和 `batch_start`，
批次开始当天及之后的访问和结果全部排除。
无判断访问增加 prior_visits，但不进入失败比例分母，也不覆盖最近有效结果。
当前结果、违规文本、ID 和原始名称不进入 `X`；编码、填补和缩放由 B 在训练折内拟合。

## 课程提交

| 阶段 | 截止时间 | 提交 |
|---|---|---|
| Assessment 1 | 2026-10-30 23:59 | EDA Notebook、数据质量分析、约一页建模计划 |
| Assessment 2 | 2026-12-04 23:59 | 最终 Notebook、predictions.csv、现场展示或视频 |

Assessment 1 的 EDA、数据质量分析和建模计划已完成，后续按该计划开展 Assessment 2。
