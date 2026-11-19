"""Deploy the fraud scoring API on Modal (serverless).

    modal deploy modal_app.py     # deploy (the GitHub workflow .github/workflows/modal-deploy.yml does this on push)
    modal serve modal_app.py      # temporary dev copy that reloads on edits
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements(str(HERE / "requirements-api.txt"))
    .env({"FRAUD_ROOT": "/root/project", "PYTHONPATH": "/root/project/src:/root/project"})
    .add_local_dir(str(HERE / "configs"), remote_path="/root/project/configs")
    .add_local_dir(str(HERE / "src"), remote_path="/root/project/src")
    .add_local_dir(str(HERE / "app"), remote_path="/root/project/app")
    .add_local_dir(str(HERE / "artifacts"), remote_path="/root/project/artifacts")
)

app = modal.App("fraud-detection-api", image=image)


@app.function(cpu=1.0, memory=1024, scaledown_window=300, timeout=60)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path[:0] = ["/root/project/src", "/root/project"]
    from app.main import app as web_app
    return web_app
