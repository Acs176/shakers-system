pg-up:
	docker compose -f "../postgres-indexing/docker-compose.yml" up -d

run-aws-pipeline:
	python -m src.app.data_ingestor.aws_event_based.worker