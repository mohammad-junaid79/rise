# AWS Strands SDK FastAPI Backend

This backend service provides agent orchestration using AWS Strands SDK, supporting both streaming and non-streaming execution. Configuration is managed via YAML files.

## Project Structure
```
backend/
├── app/
│   ├── main.py
│   ├── models/
│   ├── services/
│   ├── routers/
│   └── utils/
├── configs/
│   ├── agents/
│   └── tools/
├── requirements.txt
└── README.md
```

## Setup
1. Create and activate a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Run the server:
	```bash
	uvicorn app.main:app --reload --app-dir backend/app
	```

## Configuration
- Agent configs: `configs/agents/*.yaml`
- Tool configs: `configs/tools/*.yaml`

## Endpoints
- Health: `GET /health`
- Agent execution: `POST /agents/execute`, `POST /agents/stream`
- Session management, config reload, and more (see code for details)

## Testing
- Add unit and integration tests under `tests/`

## License
MIT
