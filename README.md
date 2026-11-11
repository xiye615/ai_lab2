# 实验二 搜索算法

本项目实现 A*、Minimax 和 Alpha-Beta，并完成迷宫寻路、加权地形、极大极小树和宝藏分配四个题目。实验还比较了启发函数、搜索深度、访问顺序和置换表缓存对搜索开销的影响。

## 项目结构

```text
experiment2/
├─ src/                         算法实现
│  ├─ astar.py                  通用 A* 搜索
│  ├─ task1_maze.py             迷宫寻路
│  ├─ task2_weighted.py         加权地形
│  ├─ game_search.py            Minimax 与 Alpha-Beta
│  ├─ treasure_game.py          四宝藏综合题
│  └─ extended_treasure_game.py 六宝藏拓展场景
├─ tests/                       自动测试
├─ experiments/                 实验与图表脚本
├─ results/
│  ├─ data/                     CSV 实验数据
│  └─ figures/                  报告图表
├─ report/                      实验报告
└─ requirements.txt
```

## 环境

- Python 3.10 或更高版本
- 核心算法只使用 Python 标准库
- 生成图表需要 Matplotlib

安装图表依赖：

```powershell
python -m pip install -r requirements.txt
```

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

当前包含 16 个测试，覆盖给定样例、无解地图、启发函数一致性、不同访问顺序、置换表和两个拓展场景。

给定样例的主要结果：

- 题目一：最短路径代价为 7；
- 题目二：最小移动代价为 7；
- 题目三：最佳得分为 5；
- 题目四：H=2 时选择 T2、最优值为 -1，H=4 时选择 T3、最优值为 3。

## 运行实验

基础对照实验：

```powershell
python experiments/run_baseline.py
```

改进与拓展实验：

```powershell
python experiments/run_enhancements.py
```

重新生成报告图表：

```powershell
python experiments/generate_complete_figures.py
```

实验脚本会更新以下结果：

- `results/data/baseline_results.csv`
- `results/data/distance_matrix.csv`
- `results/data/enhancement_results.csv`
- `results/figures/improvement_ablation_complete.png`
- `results/figures/extension_results_complete.png`

## 改进与拓展

- 使用一步收益对 Alpha-Beta 的子节点排序；
- 使用带 `EXACT`、`LOWER`、`UPPER` 标记的置换表；
- 比较基础版本、排序、缓存、排序加缓存四种配置；
- 增加六宝藏场景，观察搜索规模变化；
- 构造零代价地形，验证普通曼哈顿距离可能失去最优性保证。

最终报告位于 `report/实验二实验报告_完整版.md`。
