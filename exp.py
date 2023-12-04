
import sys
import json

BASE_COMMAND = "./{partitioner} -d {dataset} -p {partition_count} -b {imbalance} -vb {vertex_balanced}"

def generate_run_commands(config):
    IS_ASYNC = config['run_async']

    commands = []
    for dataset in config['datasets']:
        for n_partitions in config['num_partitions']:
            for partitioner in config['partitioners']:
                cmd = BASE_COMMAND.format(partitioner=partitioner['name'], 
                                          dataset=dataset, 
                                          partition_count=n_partitions,
                                          imbalance=config['imbalance'],
                                          vertex_balanced=config['vertex_balanced'])
                
                for param in partitioner:
                    if param == 'name':
                        continue
                    cmd += ' -{} {}'.format(param, partitioner[param])
                
                commands += [cmd]
    
    if IS_ASYNC:
        print(' &\n'.join(commands))
        print('wait')
    else:
        print('\n'.join(commands))

if __name__ == '__main__':
    
    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)
        generate_run_commands(config)
        