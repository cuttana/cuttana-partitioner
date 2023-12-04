JSON_FILE=$1
mkdir -p results/
make all
make clean
python3 exp.py $JSON_FILE > results/temp_commands.sh 
bash results/temp_commands.sh > results/results_$1.txt
# grep Result: results/results_$1.txt | python3 exp_parser.py > results/results_parsed_$1.txt