# Warp Divergence

在 CUDA 中，**Warp Divergence** 是指在一个 **Warp**（由 32 个线程组成的基本执行单元）内的线程由于条件分支（如 `if` 语句或循环）而执行了不同的指令路径。这种情况会导致性能下降，因为 Warp 中的线程必须按顺序执行每个分支，而不是并行执行。

---

## 什么是 Warp？

在理解 Warp Divergence 之前，首先需要了解 **Warp** 的概念。在 NVIDIA GPU 架构中，线程被组织成 Warp。一个 Warp 通常包含 32 个线程。这些线程共享同一个程序计数器，并以 **SIMT（Single Instruction, Multiple Thread）** 的方式执行。这意味着在理想情况下，一个 Warp 中的所有 32 个线程在同一时刻执行相同的指令，只是处理的数据不同。

---

## Warp Divergence 的原因

Warp Divergence 主要发生在遇到 **条件分支** 时。当一个 Warp 中的线程根据其各自的数据或线程 ID 评估条件时，可能会出现以下情况：

* **`if-else` 语句**：如果 Warp 中的一些线程满足 `if` 条件，而另一些线程满足 `else` 条件（或者不满足 `if` 条件），那么这些线程就会走向不同的执行路径。
* **`switch` 语句**：类似于 `if-else`，不同的线程可能会进入不同的 `case`。
* **循环**：如果 Warp 中的线程因为不同的条件而在不同的迭代次数退出循环，也会导致 divergence。
* **短路逻辑运算符**：`&&` 和 `||` 运算符可能会导致一些线程提前退出评估，从而产生不同的执行路径。
* **三元运算符**：`?:` 运算符本质上是一个 `if-else` 结构。

---

## Warp Divergence 的影响

Warp Divergence 对性能的主要影响是 **序列化执行**。由于 Warp 中的所有线程必须共享同一个程序计数器，当发生 divergence 时，GPU 无法同时执行不同的分支。相反，它会按顺序执行每个分支：

1.  执行第一个分支，此时走向其他分支的线程会被 **禁用**（masked out），它们虽然不执行指令，但仍然占用执行资源。
2.  执行完第一个分支后，执行第二个分支，此时之前执行第一个分支的线程被禁用。
3.  这个过程会持续到所有分支都执行完毕。

这种序列化执行意味着 Warp 的整体执行时间变成了 **所有分支执行时间的总和**，而不是最长分支的执行时间。这大大降低了 SIMT 架构的效率，因为在任何给定时间点，只有一部分线程在进行有效的工作，导致 GPU 核心的利用率下降。

---

## 如何避免或减少 Warp Divergence

虽然完全避免 Warp Divergence 有时很困难，但可以通过以下策略来减少其影响：

* **代码重构**：
    * **避免条件分支**：尽可能使用 **无分支** 的代码。例如，使用数学运算（如乘法）来代替简单的 `if` 语句来选择值。
    * **统一条件**：尝试组织数据或计算，使得 Warp 内的线程尽可能走向相同的分支。例如，对数据进行排序，使得具有相似条件的数据由同一个 Warp 处理。
* **数据重排**：将需要相似处理的数据分组，以便它们可以由同一个 Warp 处理，从而减少分支的可能性。
* **分支融合**：将多个小的分支合并成一个，以减少分支的数量。
* **使用 Predication**：在某些情况下，编译器可以使用 **predication** 来避免真正的分支。这意味着所有线程都执行相同的指令，但结果只写回那些条件为真的线程。这对于非常短的分支可能比序列化更有效。
* **Warp-Level Primitives**：使用 CUDA 提供的 Warp 级原语（如 `__ballot_sync`、`__shfl_sync` 等）可以在某些情况下更有效地处理 Warp 内的不同条件。
* **使用不同的内核**：如果存在非常不同的处理路径，可以考虑为每个路径启动一个单独的内核，但这会增加内核启动的开销。

---

## 示例

考虑以下简单的 CUDA 内核代码：

```c++
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (data[idx] > 0.0f) {
            data[idx] = sqrt(data[idx]); // 分支 A
        } else {
            data[idx] = 0.0f; // 分支 B
        }
    }
}
```

在这个例子中，如果一个 Warp 内的一些线程处理的数据 `data[idx]` 大于 0，而另一些线程处理的数据小于或等于 0，那么这个 Warp 就会发生 **divergence**。GPU 将首先执行分支 A（计算平方根），此时执行分支 B 的线程被禁用。然后，GPU 将执行分支 B（设置为 0），此时执行分支 A 的线程被禁用。这使得 Warp 的执行时间几乎是两个分支执行时间的总和，而不是单个分支的时间。

通过优化代码以减少这种 divergence，可以显著提高 CUDA 应用程序的性能。

## Reference

- [Performance Penalty Due to Warp Divergence](https://forums.developer.nvidia.com/t/performance-penalty-due-to-warp-divergence/253670/6)

- [Difference between thread divergence and warp divergence](https://forums.developer.nvidia.com/t/difference-between-thread-divergence-and-warp-divergence/64860/2)