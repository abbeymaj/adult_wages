#!/bin/bash
nohup mlflow server \
	--backend-store-uri sqlite:///model_db/mlflow.db \
	--default-artifact-root ./model_db \
	--host 0.0.0.0 \
	--port 5001 &