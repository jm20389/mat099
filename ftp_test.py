# Testing pickle object handler: save and load, to external ftp server

from classes import *

PickleHandler.nlst()
quit()

action = 'upload1'
testFile762 = 'Freddy'

if action == 'upload':
    PickleHandler.uploadOld(testFile762)
    print('File was uploaded')
else:
    retrieved_name = PickleHandler.download('testFile762')
    print('downloaded the file: ')
    print(retrieved_name)
