import sys
from collections import defaultdict

CUT_DISPLAY_FACT = 1e5

def parse():
    results = sys.stdin.readlines()

    result_dict = defaultdict(lambda: defaultdict
                              (lambda: defaultdict
                               (lambda: defaultdict(int))))
    datasets = set()
    partitions = set()
    sub_partitions = set()

    for result in results:
        result_arr = result.rstrip().split()
        dataset = result_arr[1]

        result = list(map(int,result_arr[2:]))

        partition_count = result[0]
        sub_partition_count = result[1]
        stream_edge_cut = result[2]
        refine_phase1_edge_cut = result[3]
        refine_phase2_edge_cut = result[4]
        program_runtime = result[5]
        edge_count = result[6]

        program_runtime_formatted = str(program_runtime//60) + "m " + str(program_runtime%60) + "s"
        program_runtime_formatted += " (" + str(program_runtime) + "s)"

        result_dict[dataset][partition_count][sub_partition_count] = {
            "stream_edge_cut       ": int(stream_edge_cut//CUT_DISPLAY_FACT),
            "refine_phase1_edge_cut": int(refine_phase1_edge_cut//CUT_DISPLAY_FACT),
            "refine_phase2_edge_cut": int(refine_phase2_edge_cut//CUT_DISPLAY_FACT),
            "program_runtime       ": program_runtime_formatted,
            "Stream ƛ              ": round((stream_edge_cut/edge_count)*100,2),
            "Phase1 ƛ              ": round((refine_phase1_edge_cut/edge_count)*100,2),
            "Phase2 ƛ              ": round((refine_phase2_edge_cut/edge_count)*100,2),
        }

        datasets.add(dataset)
        partitions.add(partition_count)
        sub_partitions.add(sub_partition_count)
    
    datasets = sorted(list(datasets))
    partitions = sorted(list(partitions))
    sub_partitions = sorted(list(sub_partitions))

    for dataset in datasets:
        print("====== Dataset: ",dataset," ======\n")
        for partition in partitions:
            print("----- Partition count   = ", partition, " -----\n")
            for sub_partition in sub_partitions:
                print("Sub partition count     = ", sub_partition)
                for key,value in result_dict[dataset][partition][sub_partition].items():
                    print(key," = ",value);
                print(" --- ")
            print()
    
if __name__ == '__main__':
    parse();
        