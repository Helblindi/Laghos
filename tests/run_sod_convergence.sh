#! /bin/bash

# This is a script to run convergence analysis on Laghos using Release version
# These values are specific to Whistler
laghos_dir="/Users/madisonsheridan/Workspace/Laghos"
bin_dir="${laghos_dir}/build"
create_convergence="${laghos_dir}/create_convergence_table.py"
results_dir="${bin_dir}/results"

cd $bin_dir

# EDIT THESE PARAMS
final_time=0.2
cfl=0.5
ot=1
ok=2 # must be ot + 1
oq=4 # must be >= 4
solver_type=11
output_location="${results_dir}/tests/ot1/sod"
output_file="${output_location}/out-sod-r"
########

options="-dim 1 -fa -p 2 -tf ${final_time} -cfl ${cfl} "
options+="-ot ${ot} -ok ${ok} -oq ${oq} -k ${output_location} "
options+="-s ${solver_type} -idp "

# echo $options

./laghos ${options} -rs 3 > ${output_file}3 &
./laghos ${options} -rs 4 > ${output_file}4 &
./laghos ${options} -rs 5 > ${output_file}5 &
./laghos ${options} -rs 6 > ${output_file}6 &
./laghos ${options} -rs 7 > ${output_file}7 &
./laghos ${options} -rs 8 > ${output_file}8 &
./laghos ${options} -rs 9 > ${output_file}9 &

echo "Run convergence with command:"
echo "python3 ${create_convergence} ${output_location}"
