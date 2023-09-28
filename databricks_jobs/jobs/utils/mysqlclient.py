import sys

import pymysql


class MySqlClient:

    def __init__(self, db_host, db_user, db_password, db_name, logger):
        self.host = db_host
        self.username = db_user
        self.password = db_password
        self.dbname = db_name
        self.conn = None
        self.logger = logger

    def open_connection(self):
        """Connect to MySQL Database."""
        try:
            if self.conn is None:
                self.conn = pymysql.connect(host=self.host,
                                            user=self.username,
                                            passwd=self.password,
                                            db=self.dbname,
                                            connect_timeout=5,
                                            cursorclass=pymysql.cursors.DictCursor)
                self.logger.info("Connection opened successfully.")
        except pymysql.MySQLError as e:
            self.logger.error(e)
            sys.exit()

    def run_query(self, queries):
        """Execute SQL query."""
        try:
            self.open_connection()
            self.conn.ping(reconnect=True)
            cur = self.conn.cursor()
            for query in queries:
                self.logger.info(f"Running statement: {query}")
                cur.execute(query)
            self.conn.commit()
        except pymysql.MySQLError as e:
            raise Exception(e)
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.logger.info('Database connection closed.')
