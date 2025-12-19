.PHONY: launch
launch:
	../aiwolf-nlp-server/aiwolf-nlp-server-darwin-arm64 -c ../aiwolf-nlp-server/default_5.yml


.PHONY: run
run:
	uv run python src/main.py