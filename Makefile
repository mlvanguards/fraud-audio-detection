build: # Build Docker image locally
	docker build -f Dockerfile -t fraud-detection:latest .

run: # Run Docker image locally
	docker run -p 8511:8501 --rm fraud-detection:latest
