.PHONY: build install run ncu clean

STATS := gpu__time_active.sum

# shared memory bank conflicts
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

BUILD_DIR := /root/kernel-workspace/kernels/build

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

install:
	cd $(BUILD_DIR) && make $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

run: 
	$(BUILD_DIR)/$(filter-out $@,$(MAKECMDGOALS))

ncu:
	ncu --metrics $(STATS) $(BUILD_DIR)/$(filter-out $@,$(MAKECMDGOALS))

clean:
	rm -rf $(BUILD_DIR)