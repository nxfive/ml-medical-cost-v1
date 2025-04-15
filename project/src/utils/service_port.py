import socket
from dotenv import set_key
import os


def find_free_port(env_path, key: str = 'BENTO_PORT', max_tries=10):
    start_port = int(os.getenv(key))
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                if start_port != port:
                    set_key(str(env_path), key, str(port))
                return port
    raise OSError('No free port found')
