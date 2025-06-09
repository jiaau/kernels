# Transpose

## Analysis

1024×2048 / (32×32) = 2048 tiles

65536 / 2048 = 32 requests / tile

(32 * 8) / 32 = 8 warps

32 / 8 = 4 requests / warp

## Question

### 32x32 线程相比 32x8 线程，会引入额外的冲突

### 使用 32x32 线程时，transpose_cuda_coalesced 和 transpose_cute_coalesced 的冲突次数表现存在差异

```txt
void transpose::transpose_cuda_coalesced<(unsigned int)32, (unsigned int)8>(float *, const float *, int, int) (64, 32, 1)x(32, 8, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    gpu__time_active.sum                                                   usecond        19.49
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                                2,031,616
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                  0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                          2,031,616
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  2,031,616
    ------------------------------------------------------------------ ----------- ------------

  void transpose::transpose_cuda_coalesced<(unsigned int)32, (unsigned int)32>(float *, const float *, int, int) (64, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    gpu__time_active.sum                                                   usecond        30.59
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                                2,048,985
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                  0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                          2,048,522
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  2,031,616
    ------------------------------------------------------------------ ----------- ------------

  void transpose::transpose_cute_coalesced<(unsigned int)32, (unsigned int)8>(float *, const float *, int, int) (64, 32, 1)x(32, 8, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    gpu__time_active.sum                                                   usecond        20.54
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                                2,031,616
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                  0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                          2,031,616
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  2,031,616
    ------------------------------------------------------------------ ----------- ------------

  void transpose::transpose_cute_coalesced<(unsigned int)32, (unsigned int)32>(float *, const float *, int, int) (64, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ------------------------------------------------------------------ ----------- ------------
    Metric Name                                                        Metric Unit Metric Value
    ------------------------------------------------------------------ ----------- ------------
    gpu__time_active.sum                                                   usecond        33.41
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum                                2,052,765
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                              3,527
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                          2,048,406
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum                        0
    sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  2,031,616
    ------------------------------------------------------------------ ----------- ------------
```

## Reference

- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

- [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)

- [CUDA shared memory避免bank conflict的swizzling机制解析](https://zhuanlan.zhihu.com/p/4746910252)

- [CUDA Training Series: HW8](https://github.com/olcf/cuda-training-series/tree/master/exercises/hw8)
