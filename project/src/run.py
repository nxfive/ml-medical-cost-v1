import subprocess



subprocess.run(['python', 'main.py'])
subprocess.run(['python', 'export_model.py'])
subprocess.Popen(['bentoml', 'serve'], cwd='project/server')
subprocess.Popen(['streamlit', 'run', 'app.py'], cwd='project/client')