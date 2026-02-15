.PHONY: test test-core test-sandbox test-recorder test-reward test-hub test-trainer test-integration lint build clean

PACKAGES = core sandbox recorder reward hub trainer

# 测试全部包（含集成测试）
test:
	@for pkg in $(PACKAGES); do \
		echo "\n=== Testing $$pkg ==="; \
		cd packages/$$pkg && python -m pytest tests/ -v && cd ../..; \
	done
	@echo "\n=== Integration tests ==="
	python -m pytest tests/integration/ -v

# 单独测试
test-core:
	cd packages/core && python -m pytest tests/ -v

test-sandbox:
	cd packages/sandbox && python -m pytest tests/ -v

test-recorder:
	cd packages/recorder && python -m pytest tests/ -v

test-reward:
	cd packages/reward && python -m pytest tests/ -v

test-hub:
	cd packages/hub && python -m pytest tests/ -v

test-trainer:
	cd packages/trainer && python -m pytest tests/ -v

test-integration:
	python -m pytest tests/integration/ -v

# Lint
lint:
	ruff check packages/ tests/

lint-fix:
	ruff check packages/ tests/ --fix

# 构建全部包
build:
	@for pkg in $(PACKAGES); do \
		echo "\n=== Building $$pkg ==="; \
		cd packages/$$pkg && python -m build && cd ../..; \
	done

# 开发模式安装全部包
install-dev:
	@for pkg in $(PACKAGES); do \
		echo "\n=== Installing $$pkg (editable) ==="; \
		cd packages/$$pkg && pip install -e ".[dev]" && cd ../..; \
	done

# 清理构建产物
clean:
	@for pkg in $(PACKAGES); do \
		rm -rf packages/$$pkg/dist packages/$$pkg/build packages/$$pkg/src/*.egg-info; \
	done
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
