import pickle
import io
import inspect
import json
from ftplib import FTP

class PickleHandler:
    pickleDir =   './data-pickle/'
    config_file = 'classes/server.json'

    # ftp_server = '51.68.199.40'
    # ftp_user =   'ubuntuftp'
    # ftp_pass =   'ubuntuftp'

    def __init__(self, pickleDir = None):
        self.pickleDir = './data-pickle' if pickleDir is not None else pickleDir

    @staticmethod
    def load_config():
        with open(PickleHandler.config_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def save(obj, filename):
        filepath = PickleHandler.pickleDir + '/' + filename
        with open(filepath, "wb+") as file:
            pickle.dump(obj, file)
        return None
    # Note: Saving history dict (history.history) with Pickle does NOT require .item() when loading with np.load

    @staticmethod
    def load(filename):
        with open(PickleHandler.pickleDir + '/' + filename, 'rb') as file:
            object_reloaded = pickle.load(file)
            return object_reloaded
    # Load numpy's binary object using np.load: np.load(filepath, allow_pickle = True).item()

    """
    FTP link

    """

    @staticmethod
    def upload(obj, remote_directory = None):
        config =            PickleHandler.load_config()
        remote_directory =  config['pickleDir'] if remote_directory is None else remote_directory
        ftp_server =        config['ftp_server']
        ftp_user =          config['ftp_user']
        ftp_pass =          config['ftp_pass']

        try:
            caller_frame = inspect.currentframe().f_back
            variable_name = [name for name, value in caller_frame.f_locals.items() if value is obj][0]

            pickle_bytes = pickle.dumps(obj)

            ftp = FTP(ftp_server)
            ftp.login(ftp_user, ftp_pass)

            remote_file_path = remote_directory + variable_name
            ftp.storbinary(f'STOR {remote_file_path}', io.BytesIO(pickle_bytes))

            ftp.quit()

            return None

        except Exception as e:
            print(f'An error occurred: {e}')
            return None

    @staticmethod
    def download(filename, remote_directory = None):
        config =            PickleHandler.load_config()
        remote_directory =  config['pickleDir'] if remote_directory is None else remote_directory
        ftp_server =        config['ftp_server']
        ftp_user =          config['ftp_user']
        ftp_pass =          config['ftp_pass']

        try:
            ftp = FTP(ftp_server)
            ftp.login(ftp_user, ftp_pass)

            remote_file_path = remote_directory + filename
            buffer = io.BytesIO()
            ftp.retrbinary(f'RETR {remote_file_path}', buffer.write)

            ftp.quit()

            buffer.seek(0)
            object_reloaded = pickle.load(buffer)

            return object_reloaded

        except Exception as e:
            print(f'An error occurred: {e}')
            return None