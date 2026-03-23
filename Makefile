.PHONY: install test lint typecheck clean

install:
	pip install -e ".[dev,kubeflow]" --break-system-packages

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/brahmx_sdg/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build *.egg-info

# ── Pipeline Operations ──
gold-run:
	brahmx-sdg gold run --spec $(SPEC) --kb-path $(KB) --output-dir data/gold/

silver-run:
	brahmx-sdg silver run --spec $(SPEC) --gold-ref data/gold/ --output-dir data/silver/

corpus-assemble:
	brahmx-sdg corpus assemble --manifest $(MANIFEST) --output data/corpora/

train-launch:
	brahmx-sdg train launch --run-spec $(RUN_SPEC)

eval-run:
	brahmx-sdg eval run --checkpoint $(CHECKPOINT)
