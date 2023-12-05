import json, os, traceback, hashlib, time
import numpy as np
from datetime import datetime

from classes.PRNUManager   import PRNUManager
from classes.SQLiteManager import SQLiteManager

class WorkloadManager(PRNUManager, SQLiteManager):

    ALLOWED_KEYS = {
        'id':                  str,
        'experiment':          int,
        'sequence':            str,
        'timestamp_created':   (str, datetime),  # Represented as string in JSON
        'timestamp_completed': (str, datetime, type(None)),  # Represented as string in JSON, can be None
        'state':               str,  # Should be one of 'awaiting', 'complete', 'error'
        'manipulation':        (dict, type(None)),  # Python dict with keys 'operation' : string, 'parameter' : float, can be None
        'deviceList':          list,  # List of strings
        'error_message':       (str, type(None)),
        'timestamp_error':     (str, datetime, type(None)),  # Represented as string in JSON, can be None
        'traceback':           (str, type(None))
    }

    def __init__(self, workloadDir):
        self.workloadDir = workloadDir
        self.current_workload_path = None

    @staticmethod
    def getHash(data):
        def default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        json_data = json.dumps(data, sort_keys=True, default=default)
        hash_object = hashlib.sha256()
        hash_object.update(json_data.encode())
        hash_value = hash_object.hexdigest()
        return hash_value

    def setWorkload(self, workload):
        self.current_workload_path = self.workloadDir + workload

    def _validate_data(self, data):
        for key, value_type in self.ALLOWED_KEYS.items():
            if key not in data:
                print(f"Key '{key}' is missing in the data.")
                return False
            else:
                if not isinstance(data[key], value_type):
                    print(f"Invalid type for key '{key}'. Expected {value_type}, got {type(data[key])}.")
                    return False
        return True

    def save(self, data):
        if self.current_workload_path is None:
            raise ValueError("Workload not defined.")

        if not self._validate_data(data):
            raise ValueError("JSON data not valid: " + self.current_workload_path)

        # Convert timestamps to string representation
        for timestamp in ['timestamp_created', 'timestamp_completed']:
            if isinstance(data[timestamp], datetime):
                data[timestamp] = data[timestamp].isoformat()

        with open(self.current_workload_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)

    def load(self):
        if self.current_workload_path is None:
            raise ValueError("Workload not defined.")
        try:
            with open(self.current_workload_path, 'r') as json_file:
                data = json.load(json_file)
            if not self._validate_data(data):
                raise ValueError("JSON data not valid: " + self.current_workload_path)
            # Convert string representation of timestamps to datetime objects
            for timestamp in ['timestamp_created', 'timestamp_completed']:
                if isinstance(timestamp, datetime):
                    data[timestamp] = datetime.fromisoformat(data[timestamp])
            return data
        except FileNotFoundError:
            print(f"File '{self.current_workload_path}' not found.")
        return None

    def edit(self, key, new_value):
        if self.current_workload_path is None:
            raise ValueError("Workload not defined.")

        data = self.load()
        if not data:
            raise ValueError("Cannot edit. JSON file not loaded: " + self.current_workload_path)

        if key not in self.ALLOWED_KEYS:
            raise ValueError(f"Key '{key}' not allowed in the JSON schema.")

        if isinstance(new_value, datetime):
            new_value = new_value.isoformat()
        data[key] = new_value
        self.save(data)
        print(f"Key '{key}' updated with value '{new_value}'.")

    def generateWorkloads(self
                        ,experiment_num = 2
                        ,sequence = 'a'
                        ,deviceList = ['D01', 'D02', 'D03']
                        ,operations = ['brightness', 'contrast', 'saturation']
                        ,parameters = {
                            'brightness' :  {'min':-1.0, 'max':1.1, 'step':0.2}
                            ,'contrast' :   {'min': 0.2, 'max':2.0, 'step':0.2}
                            ,'saturation' : {'min': 0.2, 'max':2.0, 'step':0.2}
                            }
                          ):
        if isinstance(operations, str):
            operations = [operations]

        if operations is not None:
            intersection = set(operations) & set(parameters.keys())
            if len(intersection) != len(operations):
                raise ValueError('Incorrect parameters dictionary')

        data_to_save = {
                        'id':                   str(time.time())
                        ,'experiment':          experiment_num
                        ,'sequence':            sequence
                        ,'timestamp_created':   datetime.now()
                        ,'timestamp_completed': None
                        ,'state':               'awaiting'
                        ,'manipulation':        {'operation': None, 'parameter': None}
                        ,'deviceList':          deviceList
                        ,'error_message':       None
                        ,'timestamp_error':     None
                        ,'traceback':           None
                    }
        if operations is not None:
            for operation in operations:
                for i in np.arange(parameters[operation]['min']
                                ,parameters[operation]['max']
                                ,parameters[operation]['step']):
                    data_to_save['manipulation']['operation'] = operation
                    data_to_save['manipulation']['parameter'] = i
                    self.setWorkload(self.getHash(data_to_save) + '.json')
                    self.save(data_to_save)
                    print(f'Created workload: {(self.getHash(data_to_save))}')
        else:
            data_to_save['manipulation']['operation'] = None
            data_to_save['manipulation']['parameter'] = None
            self.setWorkload(self.getHash(data_to_save) + '.json')
            self.save(data_to_save)
            print(f'Created workload: {(self.getHash(data_to_save))}')

        return None

    def workloadSummary(self):
        current_workloads = os.listdir(self.workloadDir)
        state_summary = {'awaiting':0, 'error':0,'complete':0}

        for wl in current_workloads:
            self.setWorkload(wl)
            wl_data = self.load()
            state_summary[wl_data['state']] += 1
        print('Workloads summary: ')
        print(state_summary)
        return state_summary

    def findWorkload(self, id):
        current_workloads = os.listdir(self.workloadDir)
        for wl in current_workloads:
            self.setWorkload(wl)
            wl_data = self.load()
            if str(wl_data['id']) == str(id):
                print(f'\nWorkload file: {wl}')
                print(json.dumps(wl_data, indent=2))
                return None

        print('Provided id does not match any existing workload.')
        return None

    def runWorkloadSingle(self, workload):
        PRNUHandler = PRNUManager()
        self.setWorkload(workload)
        wl = self.load()

        print(f'\nRunning Workload: {self.current_workload_path}')
        print(json.dumps(wl, indent=2))
        print('\n')

        try:
            result = PRNUHandler.runWorkload(wl)
            wl_id = wl['id']
            print(f"Workload '{wl_id}' run OK.")
            result_data = {
                            'id':                   wl['id']
                            ,'experiment':          wl['experiment']
                            ,'sequence':            wl['sequence']
                            ,'timestamp_created':   wl['timestamp_created']
                            ,'timestamp_completed': wl['timestamp_completed']
                            ,'operation':           wl['manipulation']['operation']
                            ,'parameter':           wl['manipulation']['parameter']
                            ,'deviceList':          ','.join(wl['deviceList'])
                            ,'auc_cc':              result['auc_cc']
                            ,'auc_pce':             result['auc_pce']
                            }
            self.edit('state', 'complete')
            self.edit('timestamp_completed', datetime.now())
            SQLiteManager.logHistoryMessage(f"Workload '{wl['id']}' marked as completed.")
            SQLiteManager.logResult(result_data)
            SQLiteManager.logHistoryMessage(f"Workload '{wl['id']}' experiment results recorded.")

        except Exception as e:
            traceback.print_exc()
            self.edit('state', 'error')
            self.edit('timestamp_error', datetime.now())
            self.edit('error_message', str(e))
            self.edit('traceback', traceback.format_exc())
            SQLiteManager.logHistoryMessage(f"Workload '{wl['id']}' marked as error.")

        return None

    def deleteWorkloadsByExperiment(self, experiment: int):
        workloads = os.listdir(self.workloadDir)
        PRNUHandler = PRNUManager()

        for workload in workloads:
            self.setWorkload(workload)
            wl = self.load()

            if wl['experiment'] != experiment :
                continue

            try:
                os.remove(self.current_workload_path)
                print(f"File '{self.current_workload_path}' deleted successfully.")
            except OSError as e:
                print(f"Error deleting file '{self.current_workload_path}': {e}")

            self.workloadSummary()

        return None

    def runWorkloads(self, state = 'awaiting', wl_ID = None, experiment = None):
        workloads = os.listdir(self.workloadDir)
        PRNUHandler = PRNUManager()

        for workload in workloads:
            self.setWorkload(workload)
            wl = self.load()
            if wl['state'] != state:
                continue
            if (experiment is not None) and (str(wl['experiment']) != str(experiment)):
                continue
            if (wl_ID is not None) and (wl['id'] != wl_ID):
                continue
            self.runWorkloadSingle(workload)
            self.workloadSummary()

        return None




