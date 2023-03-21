import logging
import sqlite3
import typing as tp

from omegaconf import OmegaConf


class DatabaseMetrics(object):
    def __init__(self, db_name: str):
        self.db_path = f'{db_name}.db'
        self._db_name = db_name
        self._sqlite_connection = None

    def create_or_connect_db(self):
        self._sqlite_connection = sqlite3.connect(self.db_path)
        cursor = self._sqlite_connection.cursor()
        logging.info(f'Connection to database {self._db_name} is successful.')
        cursor.close()
    
    def create_table(self, table_name: str):
        sqlite_create_table_query = f'''CREATE TABLE {table_name} (
                                    id INTEGER PRIMARY KEY,
                                    blur_type TEXT NOT NULL,
                                    blur_dataset text NOT NULL,
                                    kernel text NOT NULL,
                                    image_dataset text NOT NULL,
                                    image text NOT NULL,
                                    discretization text NOT NULL,
                                    noised bool NOT NULL,
                                    model text NOT NULL,
                                    ssim REAL NOT NULL,
                                    psnr REAL NOT NULL);'''

        cursor = self._sqlite_connection.cursor()

        try:
            cursor.execute(sqlite_create_table_query)
            self._sqlite_connection.commit()
            logging.info(f'Table {table_name} was created')

        except sqlite3.OperationalError as error:
            logging.error(error)
        
        finally:
            cursor.close()


class DatabaseConfigs(object):
    def __init__(self, db_name: str):
        self.db_path = f'{db_name}.db'
        self._db_name = db_name
        self._table_name = None
        self._sqlite_connection = None

    def create_or_connect_db(self):
        self._sqlite_connection = sqlite3.connect(self.db_path)
        cursor = self._sqlite_connection.cursor()
        logging.info(f'Connection to database {self._db_name} is successful.')
        cursor.close()
    
    def create_table(self, table_name: str):
        sqlite_create_table_query = f'''CREATE TABLE {table_name} (
                                    id INTEGER PRIMARY KEY,
                                    model TEXT NOT NULL,
                                    gauss_blur bool NOT NULL,
                                    motion_blur bool NOT NULL,
                                    eye_blur bool NOT NULL,
                                    no_noise_params text NOT NULL,
                                    noise_params text NOT NULL,
                                    model_path text);'''

        cursor = self._sqlite_connection.cursor()

        try:
            cursor.execute(sqlite_create_table_query)
            self._sqlite_connection.commit()
            self._table_name = table_name
            logging.info(f'Table {table_name} was created')

        except sqlite3.OperationalError as error:
            self._table_name = table_name
            logging.error(error)
        
        finally:
            cursor.close()
    
    def add(self, config_path: str, selected_models: tp.List[str]):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        insert_query = f'''INSERT INTO {self._table_name}
            (model, gauss_blur, motion_blur, eye_blur, no_noise_params, noise_params, model_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?);'''

        config = OmegaConf.load(config_path)
        cf_models = config['models']

        for model in cf_models.keys():
            if model not in selected_models:
                continue
            
            cursor.execute(
                insert_query,
                (
                    model,
                    cf_models[model].get('gauss_blur'),
                    cf_models[model].get('motion_blur'),
                    cf_models[model].get('eye_blur'),
                    str(cf_models[model].get('no_noise_params')),
                    str(cf_models[model].get('noise_params')),
                    cf_models[model].get('model_path')
                ),
            )
            connection.commit()
        cursor.close()

