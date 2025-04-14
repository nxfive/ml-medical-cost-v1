import logging
import yaml
import os
from colorama import Fore, Style, init


with open('config/params.yaml', 'r') as f:
    params = yaml.safe_load(f)


init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Style.DIM + Fore.WHITE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
    }

    def format(self, record):
        record.asctime = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        log_time = f'{self.COLORS.get(record.levelname, Fore.WHITE)}[{record.asctime}]{Style.RESET_ALL}'  
        log_level = f'{self.COLORS.get(record.levelname, Fore.WHITE)}[{record.levelname}]{Style.RESET_ALL}'
        message = record.getMessage()
        return f'{log_time} {log_level} {message}'


def setup_logger():
    logging_params = params.get('logging', {})

    logger = logging.getLogger('app_logger')
    logger.setLevel(logging_params.get('log_level', 'DEBUG'))
    logger.handlers.clear()  
    formatter = ColorFormatter('%(asctime)s  %(levelname)s  %(message)s')

    if logging_params.get('log_to_file', False):
        os.makedirs(f"../{logging_params.get('output_dir', 'logs')}", exist_ok=True)
        file_handler = logging.FileHandler('../logs/app_logs.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if logging_params.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
   
    return logger


logger = setup_logger()
