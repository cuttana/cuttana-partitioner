JSON_FILE=$1
mkdir -p results/
make ogpart_single_thread_test
make clean

python3 exp.py $JSON_FILE > results/temp_commands.sh 
bash results/temp_commands.sh > results/results_$1.txt