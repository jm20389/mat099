import pickle

class PickleHandler:
    pickleDir = './data-pickle'

    def __init__(self, pickleDir = None):
        self.pickleDir = './data-pickle' if pickleDir is not None else pickleDir

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