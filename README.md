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
│   └── examples/01_example_eda.ipynb  # 课程 EDA 示例
├── scripts/
│   ├── download_data.py               # 联网下载，默认整个允许的历史范围
│   ├── prepare_data.py                # 离线清洗、标签、质量报告和校验
│   ├── ass1_utils.py                  # Assi1 读取、统计与严格历史特征
│   ├── run_assessment1.py              # 执行并保存 Assi1 Notebook 输出
│   ├── ass2_audit_data.py              # 核查固定快照及全部历史记录
│   └── ass2_features.py                # 生成基础、历史和预测特征
├── tests/                            # 数据边界及泄漏防护测试
├── reports/                          # 质量报告、字段可用性、统计及 figures/
├── outputs/                          # 后续 predictions.csv 等输出
├── requirements.in                   # 直接依赖
├── requirements.txt                  # 经安装验证的精确依赖版本
└── AI-USAGE.md                        # AI 使用记录
~~~

## 运行

建议 Python 3.13；下载和准备脚本仅使用标准库。

已有原始快照时，以下命令均无需联网：

~~~powershell
python scripts/prepare_data.py
python scripts/prepare_data.py --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

先联网创建环境并安装依赖，再使用保存的数据离线运行 Notebook：

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
- 快照反映下载时已公开的记录，实际覆盖范围以上述检查日期为准；记录仍可能存在上报延迟。

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
- labelled_inspections.csv.gz 保存有标签记录及编号等辅助字段；模型输入使用 ass2_features.py 返回的 `X`。
  各原始字段的用途见 reports/feature_availability.csv。
- Fail=1；Pass 和 Pass w/ Conditions=0。其余结果的标签为空，作为历史访问保留。
- 初步 establishment_key 优先使用有效 License #，缺失时回退到名称和地址，
  信息不足时使用单条检查键。后续需核查许可证跨地址和实体识别问题。
- 按时间顺序划分训练、验证和测试数据；填补、缩放和编码规则仅在对应训练数据上拟合。
- data/test/ 保存教学团队提供的检查前输入，隐藏测试期的 Results 和 Violations 属于禁用信息。
- 将项目输入、数据来源说明及必要输出加入 Git。

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

首次克隆仓库后，先运行 `.\.venv\Scripts\python.exe scripts/prepare_data.py`，重建清洗表和校验清单。
Git 会统一部分文本文件的换行符，这一步让校验值与当前检出文件一致。

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

返回三个按输入行顺序对齐、索引从 0 开始的对象：

- `X`：交给模型的特征表，默认包含下表中的 13 个特征。
- `y`：真实标签，Fail 为 1，Pass / Pass w/ Conditions 为 0，其余为空。训练和评分只选有标签的行。
- `meta`：辅助信息，包括 `inspection_id`（对齐预测）、`inspection_date`（检查日期）、`establishment_key`（店铺匹配键）、`history_before`（只使用此日期之前的历史）和 `known_establishment`（此前是否有过访问，含无判断访问）。

验证数据用 `valid.y.notna()` 筛选评分行。接口接受原始字段名和清洗字段名，也支持仅含检查前信息的预测输入。

| 特征（`X`） | 字段与含义 |
|---|---|
| 基础类别 | facility_type、risk、inspection_type、zip、inspection_month、inspection_weekday；月份和星期作为类别，星期一为 0 |
| 基础数值 | latitude、longitude；缺失或非法坐标保留为 NaN，留给训练流程处理 |
| 历史次数 | prior_visits、prior_judgements、prior_failures；分别为此前访问数、有效判断数和失败数 |
| 历史失败比例 | prior_fail_rate = prior_failures / prior_judgements；没有有效判断时为 NaN |
| 最近结果 | previous_outcome：最近有判断日期的非 Fail、Fail、同日混合结果或没有此前判断 |

默认得到 8 个基础特征和 5 个历史特征。`include_history=False` 仅返回基础特征。
`identity="license_address"` 可切换为许可证加地址分组，地址匹配同时考虑城市、州和 ZIP；
有效许可证缺少地址时，每条检查使用独立的键。
没有有效许可证时，沿用名称和地址等信息的回退规则。

训练历史截至每条检查的前一天。验证和预测须传入 `history` 与 `batch_start`，整个批次共用起始日前的历史。
例如，2025 年所有验证记录均使用 2025-01-01 之前的访问。
无判断访问仅增加 `prior_visits`；失败比例和最近结果按有判断的访问计算。
`X` 仅包含上表特征，编码、填补和缩放由 B 在训练折内拟合。

正式预测使用专用入口，`test_inputs` 为老师提供的输入表：

~~~python
from ass2_features import build_prediction_features

prediction = build_prediction_features(
    test_inputs, history=history, batch_start="2026-10-31", identity="license"
)
# prediction.X 用于预测，prediction.meta.inspection_id 用于对齐输出。
~~~

该入口先移除预测输入中的 Results、label、Violations 及违规文本缺失标记，返回的 `y` 全为空。
历史表须保留 Results 或 label 列，无判断访问在该列留空。程序先保留批次起始日前的历史，再解析其中的标签。

现有 [EDA Notebook](notebooks/01_assessment1_eda.ipynb) 的第 13 节说明每组特征的依据与预测前可用条件，
并检查移除标签后的特征一致性、历史边界和 ID 对齐。老师真实测试文件发布后，仍需核对其实际字段。

## 课程提交

| 阶段 | 截止时间 | 提交 |
|---|---|---|
| Assessment 1 | 2026-10-30 23:59 | EDA Notebook、数据质量分析、约一页建模计划 |
| Assessment 2 | 2026-12-04 23:59 | 最终 Notebook、predictions.csv、现场展示或视频 |

Assessment 1 的 EDA、数据质量分析和建模计划已完成，后续按该计划开展 Assessment 2。
