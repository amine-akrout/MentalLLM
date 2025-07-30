.PHONY: ollama
ollama:
	@echo "Creating Ollama model from modelfile..."
	@ollama create mental-llm-phi3 -f Modelfile
	@echo "Ollama model created successfully."
# run the ollama model
.PHONY: run-ollama
run-ollama:
	@echo "Running Ollama model..."
	@ollama run mental-llm-phi3
	@echo "Ollama model run successfully."


