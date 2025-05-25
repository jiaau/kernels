# Kernels

## 编译与运行

### 编译项目

```bash
make build
make install <kernel_name>
```

### 运行测试

```bash
make run <kernel_name>
```

### 使用NVIDIA Compute Profiler进行性能分析

```bash
make ncu <kernel_name>
```

### 清理构建文件

```bash
make clean
```

## 命令行选项

运行SGEMM测试时支持以下选项：

- `--bench`: 启用基准测试模式
- `--times N`: 指定基准测试迭代次数（默认：3）
- `--help`: 显示帮助信息

例如：

```bash
make run <kernel_name> -- --bench --times 10
```

## Acknowledgments

- 本项目使用了 [Chtholly-Boss/swizzle](https://github.com/Chtholly-Boss/swizzle) 的一些工具函数