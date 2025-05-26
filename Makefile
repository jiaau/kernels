.PHONY: build install run ncu clean

STATS := gpu__time_active.sum

# shared memory bank conflicts
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
STATS := $(STATS),l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

BUILD_DIR := /root/kernel-workspace/kernels/build
EXE_DIR := $(BUILD_DIR)/src
EXE_EXAMPLE_DIR := $(BUILD_DIR)/examples

%:
	@:

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

install:
	cd $(BUILD_DIR) && make $(filter-out $@,$(MAKECMDGOALS))

run:
	@CMDARGS="$(filter-out $@,$(MAKECMDGOALS))"; \
	PROG=$$(echo "$$CMDARGS" | awk '{print $$1}'); \
	if [ -z "$$PROG" ]; then \
		echo "Error: Program name not specified. Usage: make run <program_name> [arguments]"; \
		exit 1; \
	elif [ -f "$(EXE_DIR)/$$PROG" ]; then \
		PROGPATH="$(EXE_DIR)/$$PROG"; \
	elif [ -f "$(EXE_EXAMPLE_DIR)/$$PROG" ]; then \
		PROGPATH="$(EXE_EXAMPLE_DIR)/$$PROG"; \
	else \
		echo "Error: $$PROG not found in $(EXE_DIR) or $(EXE_EXAMPLE_DIR)"; \
		exit 1; \
	fi; \
	ARGS=$$(echo "$$CMDARGS" | sed -E 's/^[^ ]+[ ]*//'); \
	echo "Executing: $$PROGPATH $$ARGS"; \
	$$PROGPATH $$ARGS

ncu:
	@PROG=$(filter-out $@,$(MAKECMDGOALS)) && \
	if [ -z "$$PROG" ]; then \
		echo "Error: Program name not specified. Usage: make ncu <program_name>"; \
		exit 1; \
	elif [ -f "$(EXE_DIR)/$$PROG" ]; then \
		ncu --metrics $(STATS) $(EXE_DIR)/$$PROG; \
	elif [ -f "$(EXE_EXAMPLE_DIR)/$$PROG" ]; then \
		ncu --metrics $(STATS) $(EXE_EXAMPLE_DIR)/$$PROG; \
	else \
		echo "Error: $$PROG not found in $(EXE_DIR) or $(EXE_EXAMPLE_DIR)"; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)