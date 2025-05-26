# SGEMM Optimization Examples

## 实现版本

本项目包含以下SGEMM实现版本：

### 1. sgemm_v1.cuh

基础版本实现，使用了简单的CUDA矩阵乘法模式。

### 2. sgemm_v2.cuh

改进版本，消除了 load 的 bank conflict，提高了性能，但是引入了 store 的 bank conflict。

为什么 load 没有 bank conflict？

分析后可以得到 tid=0 和 tid=8 访问了 smem 的同一 bank0，但是 smem 一次 transaction 最大为 128 字节，tid0-tid7 在同一批 transaction 中，tid8-tid15 在下一批 transaction 中，因此不会发生 bank conflict。具体细节查看 [How to understand the bank conflict of shared_mem](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900)。

```txt
sgemm::sgemm_v2(float *, float *, float *, int, int, int) (4, 4, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    gpu__time_active.sum                                         usecond        72.16
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                         32,768
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                   32,768
    -------------------------------------------------------- ----------- ------------
```

store 的 bank conflict 由以下代码引起：

```cpp
s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
```

对于每次 store 操作，相邻线程的 store 操作会访问到同一个 bank，如（0，1）线程访问 bank 0，（2，3）线程访问 bank 1，（4，5）线程访问 bank 2， ... （30，31）线程访问 bank 15。即 1 次 store 发生 1 次 2way bank conflict。

4 次 store 操作共冲突 4 次。

计算一次 (BM, BN, BK) 的线程数量为 (BM * BN) / (TM * TN)，即 128 * 128 / (8 * 8) = 256。 warp 数量为 256 / 32 = 8。

计算一次 (BM, BN, BK) 的 bank conflict 数量为 4 * 8 = 32。

完成最终 (BM, BN) 大小的矩阵计算需要循环 (K + BK - 1) / BK 次。即 (512 + 8 - 1) / 8 = 64 次。

因此，完成最终 (BM, BN) 大小的矩阵计算会发生 64 * 32 = 2048 次 bank conflict。

(BM, BN) 大小的矩阵共有 (M + BM - 1) / BM * (N + BN - 1) / BN = (512 + 128 - 1) / 128 * (512 + 128 - 1) / 128 = 16 次计算。

因此，完成最终 (M, N) 大小的矩阵计算会发生 16 * 2048 = 32768 次 bank conflict。

### 3. sgemm_v2_naive.cuh

v2消除 store bank conflict 的朴素实现版本，作为对比参考。

此版本虽然消除了 store bank conflict，但是由于存在过多分支判断，导致 warp divergence，降低了性能。

### 4. sgemm_v2_bitop.cuh

在 v2_naive 的基础上使用位操作优化的版本，减少分支判断并提高性能。

```txt
sgemm::sgemm_v2_bitop(float *, float *, float *, int, int, int) (4, 4, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    gpu__time_active.sum                                         usecond        68.51
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                              0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    -------------------------------------------------------- ----------- ------------
```

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