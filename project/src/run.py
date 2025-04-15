import subprocess
from utils.logger import logger
from utils.service_port import find_free_port
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=ENV_PATH)

logger.info('Model building...')
try:
    result = subprocess.run(['python', 'main.py'], capture_output=True, text=True, check=True)
except subprocess.CalledProcessError as e:
    logger.error(f'Error while running main.py:\n{e.stderr}')


logger.info('Exporting model...')
try:
    result = subprocess.run(['python', 'export_model.py'], cwd='project/server', capture_output=True, text=True, check=True)
except subprocess.CalledProcessError as e:
    logger.error(f'Error while running export_model.py:\n{e.stderr}')


free_port = find_free_port(ENV_PATH)
logger.info(f'Start BentoML service on port {free_port}')
try:
    subprocess.Popen(
        ['bentoml', 'serve', 'service:MedicalRegressorService', f'--port={free_port}'],
        cwd='project/server'
    )
except Exception as e:
    logger.error(f'Failed to start BentoML: {e}')


logger.info('Start Streamlit app...')
try:
    subprocess.Popen(['streamlit', 'run', 'app.py'], cwd='project/client')
except Exception as e:
    logger.error(f'Failed to start Streamlit app: {e}')
