# Cuttana 🗡

This repository has the code for the partitioner. 

### Prerequisites and Package Versions

To use this software, make sure you have the following prerequisites installed:

- **make:** Version 4.3 or higher
- **g++ compiler:** Version 11.4 or higher
- **OpenMP:** Version 4.5 (201511) or higher

#### Installation on Ubuntu 22.04 LTS
To install the prerequisites, run the following command:

```bash
sudo apt-get install libomp-dev make g++
```

### Dataset Format

An example dataset is located at `examples/emailEnron.txt`. The file should start with number of vertices and edges following by one line for each vertex that has `vertex_id, degree, space separated list of neighbours`. Basically the file format should be like this:

```
vertex_count edge_count
vertex_id_1 deg_1 nei_1 nei_2 ... 
vertex_id_2 deg_2 nei_1 nei_2 ... 
.
.
.
```

### Building Project and Partitioning

You can easily build project by simply doing:

```
make all
```

The only requirement is OpenMP library and we put the queue inside the project for easier build. 

After building project you can run experiment by:

```
./ogpart -d {dataset} -p {partition_count} -b {imbalance} -vb {vertex_balanced}
```

Where:
1. `dataset` is the relative path to the dataset file. (Specified below)

2. `partition_count` is K in the paper or number of the partitions which is a single integer. 
3. `imbalance` is the epsilon in the paper that controls or relax imbalance. Imbalance is a float number. 
4. `vb` is boolean value that controls whether the balance is edge or vertex. 

So for example a valid command is:

```
./ogpart -d data/twitter -p 4 -b 0.05 -vb 1
```

You can also determine number of sub-partitions and buffer size as well. 

### Structure of Project

The code is mostly in `partitioners/ogpart.cpp` where the buffering logic and the control flow is implemented. 

There is two function in the class that you can call. 

The first is ```write_to_file``` which write the output of partitioning in a file in this format for each vertex:

```
vertex_id,partition_id
vertex_id,partition_id
.
.
.
vertex_id,partition_id
```

and a `verify` function which independetly restream the file again and measure the quality metrics. You can also verify and measure quality metrics from the output file of `write_to_file` function. 

The refinement logic is implemented in `partitioners/refine.cpp` and finally a segment tree is implemented in `util` directory. 

### Running Batch Experiment

We facilitate running a group of different experiments on various datasets, partition count, and settings. 

You can read and manipulate `exp.json` which contains the specification and using:

```
./exp.sh exp.json
```

You can run all experiments. 
Also, for an efficient single thread implementation you can use `exp_single_thread.sh`.
### Communication Volume Mode

To optimize for communication volume you should compile with defining `CV` flag. 

This mode has optimizations for communication volume particularly for synchronous graph analytics. 


