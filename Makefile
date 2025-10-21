
SHELL := /bin/bash
.PHONY: up down logs pull-models health etl index-docs test

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

pull-models:
	docker compose exec -T ollama bash -lc 'ollama pull $(CHAT_MODEL) && ollama pull $(EMBEDDING_MODEL)'

health:
	@echo 'API:' && curl -s http://localhost:$(API_PORT)/health || true
	@echo && echo 'Ollama:' && curl -s http://localhost:$(OLLAMA_PORT)/api/version || true

etl:
	docker compose exec -T api python -m etl.run

index-docs:
	docker compose exec -T api python ingest_docs.py

test:
	docker compose exec -T api pytest -q || true
