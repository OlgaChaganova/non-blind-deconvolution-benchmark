import logging
import sqlite3


class Database(object):
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

