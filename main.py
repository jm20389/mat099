import argparse
from classes import *

def runInstruction(args):
    instruction     = args.instruction
    wl_ID           = args.wl_ID
    wl_state        = args.wl_state
    experiment      = args.experiment
    workloadHandler = WorkloadManager('workloads/')

    if experiment is not None and isinstance(experiment, int):
        experiment = int(experiment)

    if instruction == 'run_workloads':
        print('Run workloads...')
        state = 'awaiting' if wl_state is None else wl_state
        workloadHandler.runWorkloads(state = state, wl_ID = wl_ID, experiment = experiment)
        print('All workloads matching provided conditions have been run.')

    elif instruction == 'find_workload':
        print('Find workload...')
        if wl_ID is None:
            print('No Workload ID was provided.')
            quit()
        workloadHandler.findWorkload(wl_ID)

    elif instruction == 'generate_workloads':
        print('Generate workloads...')
        # workloadHandler.generateWorkloads(experiment_num = 4 # Failure to process locally
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D01']
        #                          ,operations = ['style_transfer']
        #                          ,parameters = {
        #                              'style_transfer' : {'min':-1.0, 'max':1.1, 'step':0.2}
        #                             }
        #                         )

        # workloadHandler.generateWorkloads(experiment_num = 5
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D01', 'D02', 'D03']
        #                          ,operations = ['clarendon'
        #                                         ,'juno'
        #                                         ,'gingham'
        #                                         ,'lark'
        #                                         ,'sierra'
        #                                         ,'ludwig'
        #                                         ]
        #                          ,parameters =  {
        #                                         'clarendon': {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                         'juno':      {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                         'gingham':   {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                         'lark':      {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                         'sierra':    {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                         'ludwig':    {'min': 0.5, 'max': 2.0, 'step': 0.2},
        #                                     }
        #                         )

        # Style transfer devices
        # workloadHandler.generateWorkloads(experiment_num = 6 # style transfer low
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D41', 'D42', 'D43']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )
        # workloadHandler.generateWorkloads(experiment_num = 6 # style transfer med
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D44', 'D45', 'D46']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )
        # workloadHandler.generateWorkloads(experiment_num = 6 # style transfer high
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D47', 'D48', 'D49']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )

        # workloadHandler.generateWorkloads(experiment_num = 7 # style transfer low3
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D51', 'D52', 'D53']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )

        # workloadHandler.generateWorkloads(experiment_num = 7 # style transfer low4
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D54', 'D55', 'D56']
        #                          ,operations = None
        #                          ,parameters = None
                                # )

        # workloadHandler.generateWorkloads(experiment_num = 8 # normal test, just to try the pce histogram plot
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D01', 'D02', 'D03']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )

        # workloadHandler.generateWorkloads(experiment_num = 9 # repeat low4 style transfer test (exp 7) but plotting pce histogram
        #                     ,sequence = 'a'
        #                     ,deviceList = ['D54', 'D55', 'D56']
        #                     ,operations = None
        #                     ,parameters = None
        #                 )

        # workloadHandler.generateWorkloads(experiment_num = 10 # style transfer med
        #                          ,sequence = 'a'
        #                          ,deviceList = ['D44', 'D45', 'D46']
        #                          ,operations = None
        #                          ,parameters = None
        #                         )

        workloadHandler.generateWorkloads(experiment_num = 11 # repeat exp 5 but only sierra and saving histograms
                                 ,sequence = 'a'
                                 ,deviceList = ['D01', 'D02', 'D03']
                                 ,operations = ['sierra']
                                 ,parameters =  {'sierra':    {'min': 0.5, 'max': 2.0, 'step': 0.2}}
                                )


        print('Workloads created.')
        quit()
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

    elif instruction == 'delete_workloads_by_experiment':
        print(f'Deleting workloads from experiment: {str(experiment)} ...')
        workloadHandler.deleteWorkloadsByExperiment(experiment)
        print('All workloads matching provided condtions have been deleted.')

    else:
        print('No instruction provided.')
        quit()

def main():
    parser = argparse.ArgumentParser(description='MAT099 Image Forensics Project')
    parser.add_argument('--instruction', help='Instruction to run')
    parser.add_argument('--wl_state',    help='Workload states to run, default awaiting')
    parser.add_argument('--wl_ID',       help='Workload ID')
    parser.add_argument('--experiment',  help='Experiment ID')
    args = parser.parse_args()

    runInstruction(args)

if __name__ == "__main__":
    main()