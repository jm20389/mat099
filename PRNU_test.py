from classes import *

def main():
    PRNUProcessor.testPRNU(ff_dir =  'datasets/data-korus/serialized/tampered/*.TIF',
                           nat_dir = 'datasets/data-korus/serialized/natural/*.TIF')

if __name__ == '__main__':
    main()