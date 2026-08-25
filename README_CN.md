# BoostKit pandas

BoostKit pandas 是基于上游社区 pandas 的鲲鹏平台性能优化项目，属于鲲鹏
BoostKit 应用使能套件生态的一部分，旨在提升 pandas 在 ARM64 / 鲲鹏环境下
的数据处理性能。

本项目面向 pandas 的数据结构与操作、数学与统计计算、时间序列分析及数据处理
工具进行优化，并通过功能测试和 ASV benchmark 验证优化效果。在提升性能的
同时，本项目保持与上游 pandas 公共 API、行为语义和计算结果的一致性。

## 项目状态

本项目处于持续开发和优化阶段。

当前工作重点包括 ARM64 / 鲲鹏平台性能优化、benchmark 结果验证，以及保持与
上游 pandas 行为的兼容性。

## 环境要求

推荐在 Linux ARM64 / aarch64 环境下构建和运行本项目，重点目标平台为鲲鹏
服务器。其他 Linux 平台可用于兼容性和性能回归验证。

推荐环境如下：

- Python 3.14
- NumPy 2.4.3
- Cython 3.2.4
- Meson / meson-python
- PyArrow 24.0.0
- Numba 0.66.0
- NumExpr 2.14.1
- ASV
- pytest / pytest-xdist

具体构建依赖和版本要求请以 `pyproject.toml`、`meson.build` 和
`doc/source/development/contributing_environment.rst` 中的说明为准。

## 源码构建

安装测试和版本管理依赖：

```bash
python -m pip install pytest-xdist "versioneer[toml]"
```

进入源码目录并执行可编辑安装：

```bash
cd pandas
python -m pip install -e . --no-build-isolation
```

使用 `--no-build-isolation` 时，当前 Python 环境需要预先安装
`pyproject.toml` 中声明的构建依赖。

## 特性概览

### ARM64 / 鲲鹏性能优化

本项目面向 ARM64 架构进行 pandas 性能优化，重点关注 pandas 在鲲鹏服务器上
的数据处理表现。优化工作结合算法改进、热点路径下沉、SIMD 向量化、缓存访问
优化及不必要开销消除等方式，提升常见数据分析场景的执行效率。

相较上游基线，本项目在数据构造与变换、分组聚合、窗口计算、索引与合并、
哈希密集型操作、字符串处理和时间序列分析等场景具有性能优势。具体优化内容以
版本变更、设计文档和 benchmark 覆盖为准。

### 哈希与通用算法

本项目引入支持 SIMD 加速的 SwissTable 哈希表实现，并优化 pandas 通用算法和
哈希密集型数据处理路径，改善数值数据的哈希表构建、批量查找、因子化、唯一值
处理、成员判断、值计数和重复值处理等场景。

优化实现会根据数据类型、数据规模和运行平台选择合适的执行路径，并在不满足
优化条件时使用原有实现，以保持 pandas 的行为和顺序语义。

### 分组与聚合计算

本项目优化 GroupBy 的分组构建、数值与字符串聚合、变换、累计计算和自定义函数
处理路径，减少分组过程中的中间对象、重复计算和数据重排，并改进底层聚合循环
在 ARM64 平台上的执行效率。

### 窗口与滚动计算

本项目优化 Rolling 和 Expanding 的窗口边界生成、数值聚合和中位数计算等路径，
通过改进底层数据结构、增量计算方式和内存访问模式，提升固定窗口、可变窗口及
扩展窗口场景的处理效率。

### 索引、合并与数据结构操作

本项目优化 DataFrame / Series 构造、索引选择、数据对齐、缺失值处理、布尔索引、
算术运算和函数应用等数据结构操作，同时改善 RangeIndex、MultiIndex、数据合并
和连接等场景中的查找、对齐和结果构造开销。

### 字符串、分类数据与时间序列

本项目优化字符串拼接与处理、分类数据索引与统计等场景，并改善时间偏移、
Timedelta、Datetime 属性访问和重采样等时间序列处理路径，降低类型转换、对象
创建和重复计算带来的开销。

### Benchmark 性能验证

本项目使用 ASV benchmark 对性能变化进行评估，并增加面向鲲鹏平台、SwissTable
和典型数据处理工作负载的 benchmark。测试结果用于验证优化收益、分析性能波动，
并识别可能存在的性能劣化场景。

### 上游 pandas 兼容性

本项目基于上游社区 pandas，保持与上游 pandas 公共 API、用户可见行为、异常行为
和计算语义的一致性。

性能优化不应改变 pandas 原有接口语义，也不应影响已有测试用例的正确性。针对
平台特定的实现使用运行时检测或独立路径进行隔离，并在不适用时保留原有实现。

## SwissTable 配置

SwissTable 是本项目引入的实验性 SIMD 加速哈希表后端，用于支持部分数值类型的
哈希密集型操作。该后端在 ARM64 / aarch64 平台默认启用，在其他平台默认关闭。

查看当前配置：

```python
import pandas as pd

pd.get_option("compute.use_swisstable")
```

可以使用上下文配置临时启用或关闭 SwissTable，退出上下文后自动恢复原配置：

```python
with pd.option_context("compute.use_swisstable", True):
    result = df["key"].isin(values)

with pd.option_context("compute.use_swisstable", False):
    result = df["key"].isin(values)
```

也可以全局设置该选项：

```python
pd.set_option("compute.use_swisstable", True)
pd.set_option("compute.use_swisstable", False)
```

## 功能测试

为减少多线程数学库与并行测试进程之间的资源竞争，建议在运行测试前设置线程数：

```bash
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

运行功能测试，并跳过依赖网络、数据库、剪贴板和绘图环境的测试：

```bash
pytest pandas -n 4 -m "not network and not db and not clipboard" --ignore=pandas/tests/plotting
```

也可以根据需要运行指定测试：

```bash
pytest pandas/path/to/test.py
```

## 性能测试

本项目使用 ASV 进行性能测试。进入 `asv_bench` 目录后运行指定 benchmark：

```bash
cd asv_bench
asv run -b benchmark_name
```

对比基线提交与优化提交：

```bash
asv continuous BASELINE_COMMIT OPTIMIZED_COMMIT -b benchmark_name
```

## 更多文档

详细的设计说明、构建指南、开发说明和 benchmark 请参阅以下文档或目录：

- [Pandas 数据结构与操作、数学与统计计算、时间序列分析及数据处理工具性能优化 RFC](<doc/rfc/26.2.0/RFC - Pandas 数据结构与操作、数学与统计计算、时间序列分析及数据处理工具性能优化.md>) - 本项目的优化目标、技术方案、功能设计和验证要求。
- [鲲鹏 920B benchmark](asv_bench/benchmarks/aggregate_920b) - 面向鲲鹏 920B 平台的数据处理 benchmark。
- [鲲鹏 950 benchmark](asv_bench/benchmarks/aggregate_950) - 面向鲲鹏 950 平台的数据处理 benchmark。
- [SwissTable benchmark](asv_bench/benchmarks/swisstable.py) - SwissTable 哈希表相关性能测试。
- [自定义工作负载 benchmark](asv_bench/benchmarks/custom_case.py) - 典型数据处理工作负载性能测试。
- [开发环境说明](doc/source/development/contributing_environment.rst) - 上游 pandas 源码构建和开发环境说明。
- [测试与 benchmark 指南](doc/source/development/contributing_codebase.rst) - 上游 pandas 功能测试和 ASV 使用说明。
- [pandas 用户文档](https://pandas.pydata.org/docs/) - pandas 公共 API 和用户指南。

## License

本项目遵循上游 pandas 的 BSD 3-Clause License，并遵循相关第三方依赖的许可证
要求。
