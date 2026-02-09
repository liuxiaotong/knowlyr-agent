.PHONY: test test-sandbox test-recorder test-reward test-hub lint build clean

PACKAGES = sandbox recorder reward hub

# 测试全部包
test:
	@for pkg in $(PACKAGES); do \
		echo "\n=== Testing $$pkg ==="; \
		cd packages/$$pkg && python -m pytest tests/ -v && cd ../..; \
	done

# 单独测试
test-sandbox:
	cd packages/sandbox && python -m pytest tests/ -v

test-recorder:
	cd packages/recorder && python -m pytest tests/ -v

test-reward:
	cd packages/reward && python -m pytest tests/ -v

test-hub:
	cd packages/hub && python -m pytest tests/ -v

# Lint
lint:
	ruff check packages/

lint-fix:
	ruff check packages/ --fix

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
