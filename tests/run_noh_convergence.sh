#! /bin/bash

# This is a script to run convergence analysis on Laghos using Release version
# These values are specific to Whistler
laghos_dir="/Users/madisonsheridan/Workspace/Laghos"
bin_dir="${laghos_dir}/build"
create_convergence="${laghos_dir}/create_convergence_table.py"
results_dir="${bin_dir}/results"

cd $bin_dir

# EDIT THESE PARAMS
mesh="${laghos_dir}/../Laglos/data/noh-10.mesh"
final_time=0.6
cfl=0.5
ot=1
ok=2 # must be ot + 1
oq=4 # must be >= 4
solver_type=4
output_location="${results_dir}/tests/ot1/noh"
output_file="${output_location}/out-noh-r"
########

options="-m ${mesh} -fa -p 10 -tf ${final_time} -cfl ${cfl} "
options+="-ot ${ot} -ok ${ok} -oq ${oq} -k ${output_location} "
options+="-s ${solver_type} "
# options+="-idp "

echo $options

./laghos ${options} -rs 0 > ${output_file}0 &
./laghos ${options} -rs 1 > ${output_file}1 &
./laghos ${options} -rs 2 > ${output_file}2 &
# ./laghos ${options} -rs 3 > ${output_file}3 &
# ./laghos ${options} -rs 4 > ${output_file}4 &
#./laghos ${options} -rs 5 > ${output_file}5 &
#./laghos ${options} -rs 6 > ${output_file}6 &
