from classes import *

workloadHandler = WorkloadManager('workloads/')
workloadHandler.generateWorkloads(experiment_num = 3
                                 ,sequence = 'a'
                                 ,deviceList = ['D01', 'D02', 'D03']
                                 ,operations = ['brightness', 'contrast', 'saturation']
                                 ,parameters = {
                                     'brightness' : {'min':-1.0, 'max':1.1, 'step':0.2}
                                    ,'contrast' :   {'min': 0.2, 'max':2.0, 'step':0.2}
                                    ,'saturation' : {'min': 0.2, 'max':2.0, 'step':0.2}
                                    }
                                )

