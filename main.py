"""
Root entry point for Azure Web App (Azure App Service).
Allows Azure Web App to find and run `app` via:
    uvicorn main:app --host 0.0.0.0 --port 8000
or
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
"""
from app.main import app
from app.config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
