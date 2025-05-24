# SGEMM Optimization Examples

## 实现版本

本项目包含以下SGEMM实现版本：

### 1. sgemm_v1.cuh

基础版本实现，使用了简单的CUDA矩阵乘法模式。

### 2. sgemm_v2.cuh

改进版本，引入了共享内存和线程块优化。

### 3. sgemm_v2_naive.cuh

v2消除 bank conflict 的朴素实现版本，作为对比参考。

### 4. sgemm_v2_bitop.cuh

在 v2_naive 的基础上使用位操作优化的版本，减少分支判断并提高性能。

### 5. sgemm_v3.cuh

双缓冲优化版本

## 测试

所有实现都使用 `test_sgemm.cuh` 中的函数进行测试，测试矩阵大小为 512x512x512。

## 性能比较

可以使用以下命令运行基准测试并比较不同版本的性能：

```bash
make run sgemm -- --bench --times 10
```

## 性能分析

使用NVIDIA Compute Profiler (NCU) 分析性能：

```bash
make ncu sgemm
```

这将收集关键性能指标，包括：
- GPU活跃时间
- 共享内存的bank冲突数
- 加载/存储操作中的内存冲突数

## 参考资料

- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)

- [How to understand the bank conflict of shared_mem](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900)