import sqlite3
from datetime import datetime

class SQLiteManager:
    @staticmethod
    def createDatabase(databaseName, filepath):
        sample_data = {
            'id':                   1
            ,'experiment':          123
            ,'sequence':            'test01'
            ,'timestamp_created':   str(datetime.now())
            ,'timestamp_completed': None
            ,'state':               'a'
            ,'operation':           'example_operation'
            ,'parameter':           1.23
            ,'deviceList':          ['device1', 'device2']
            ,'auc_cc':              0.98
            ,'auc_pce':             0.95
        }

        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ''' + databaseName + '''(
                id                   INTEGER PRIMARY KEY AUTOINCREMENT
                ,wl_id               TEXT
                ,experiment          INTEGER
                ,sequence            TEXT
                ,timestamp_created   TEXT
                ,timestamp_completed TEXT
                ,operation           TEXT
                ,parameter           REAL
                ,deviceList          TEXT
                ,auc_cc              REAL
                ,auc_pce             REAL
            )
        ''')

        cursor.execute('''
            INSERT INTO ''' + databaseName + '''
            (wl_id, experiment, sequence, timestamp_created, timestamp_completed, operation, parameter, deviceList, auc_cc, auc_pce)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(sample_data['id'])
            ,sample_data['experiment']
            ,sample_data['sequence']
            ,sample_data['timestamp_created']
            ,sample_data['timestamp_completed']
            ,sample_data['operation']
            ,sample_data['parameter']
            ,str(sample_data['deviceList'])
            ,sample_data['auc_cc']
            ,sample_data['auc_pce']
        ))

        conn.commit()
        conn.close()

    @staticmethod
    def createHistoryDatabase(databaseName, filepath):
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ''' + databaseName + '''(
                id                   INTEGER PRIMARY KEY AUTOINCREMENT
                ,time                TEXT
                ,message             TEXT
            )
        ''')

        cursor.execute('''
            INSERT INTO ''' + databaseName + '''
            (time, message)
            VALUES (?, ?)
        ''', (
            str(datetime.now())
            ,'initial log message'
        ))

        conn.commit()
        conn.close()

    @staticmethod
    def logHistoryMessage(message, filepath = 'db/history.db', databaseName = 'history'):
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ''' + databaseName + '''
            (time, message)
            VALUES (?, ?)
        ''', (
            str(datetime.now())
            ,message
        ))

        conn.commit()
        conn.close()

        return None

    @staticmethod
    def logResult(result_data, filepath = 'db/results.db', databaseName = 'results'):
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ''' + databaseName + '''
            (wl_id, experiment, sequence, timestamp_created, timestamp_completed, operation, parameter, deviceList, auc_cc, auc_pce)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_data['id']
            ,result_data['experiment']
            ,result_data['sequence']
            ,result_data['timestamp_created']
            ,result_data['timestamp_completed']
            ,result_data['operation']
            ,result_data['parameter']
            ,str(result_data['deviceList'])
            ,result_data['auc_cc']
            ,result_data['auc_pce']
        ))

        conn.commit()
        conn.close()

        return None


if False:
    SQLiteManager.createDatabase('results', 'db/results.db')
if False:
    SQLiteManager.createHistoryDatabase('history', 'db/history.db')