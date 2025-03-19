#! /bin/bash

# This is a script to run convergence analysis on Laghos using Release version
# These values are specific to Whistler
laghos_dir="/home/sheridanm/software/Laghos"
scripts_dir="${laghos_dir}/scripts"
bin_dir="${laghos_dir}/build"
create_convergence="${laghos_dir}/create_convergence_table.py"
results_dir="/home/sheridanm/scratch/Laghos-results"
temp_output="${bin_dir}/results/convergence/temp_output/"
state_vectors="${bin_dir}/results/state_vectors/"
convergence_dir="${bin_dir}/results/convergence/"


cd $bin_dir

# EDIT THESE PARAMS
mesh_file="${laghos_dir}/../Laglos/data/ref-square-N15.mesh"
problem=1
final_time=.8
cfl=0.5
ot=1
ok=2 # must be ot + 1
output_location="${results_dir}/testing/ot1/sedov"
output_file="${output_location}/outsedov-r"
########

options="-m ${mesh_file} -p ${problem} -tf ${final_time} -cfl ${cfl} "
options+="-ot ${ot} -ok ${ok} "
options+="-k ${output_location} "

echo $options

srun-smt -n 8 ./laghos ${options} -rs 0 > ${output_file}0 &
srun-smt -n 8 ./laghos ${options} -rs 1 > ${output_file}1 &
srun-smt -n 8 ./laghos ${options} -rs 2 > ${output_file}2 &
srun-smt -n 8 ./laghos ${options} -rs 3 > ${output_file}3 &
#srun-smt -n 8 ./laghos ${options} -rs 4 > ${output_file}4 &
#srun-smt -n 8 ./laghos ${options} -rs 5 > ${output_file}5 &


# Lastly, copy this script to the output_file location
# for documentation on run parameters
cp "${laghos_dir}/whistler_convergence_script.sh" "${output_location}/." 

