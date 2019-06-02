gcc ./Part_1/main.c -o ./Part_1/ektelesimo -msse4.2 
gcc ./Part_2/main.c -o ./Part_2/ektelesimo -msse4.2 -pthread
mpicc ./Part_3/main.c -o ./Part_3/ektelesimo -msse4.2 -pthread

sleep_time=3
sleep_time_between_results=1

echo -e "\n\n"
echo -e "---------------------- We will SEE code --------------------------------------------------"
echo -e "------------------------- to demonstrate that everything works ---------------------------"

./Part_1/ektelesimo 100000000
sleep ${sleep_time}

echo -e "\n\n"
echo -e "---------------------- We will SEE + Pthreads code --------------------------------------------------"
echo -e "------------------------- to demonstrate that everything works ---------------------------"

echo -e "------------------------- First for 2 threads ---------------------------"
./Part_2/ektelesimo 100000000 2
sleep ${sleep_time_between_results}
echo -e "------------------------- Second for 4 threads ---------------------------"
./Part_2/ektelesimo 100000000 4
sleep ${sleep_time}

echo -e "\n\n"
echo -e "---------------------- We will SEE + Pthreads + MPI code --------------------------------------------------"
echo -e "------------------------- to demonstrate that everything works ---------------------------"

echo -e "------------------------- First for 2 processes and 2 threads ---------------------------"
mpiexec -n 2 ./Part_3/ektelesimo 100000000 2
sleep ${sleep_time_between_results}
echo -e "------------------------- Second for 2 processes and 4 threads ---------------------------"
mpiexec -n 2 ./Part_3/ektelesimo 100000000 4
sleep ${sleep_time}
echo -e "------------------------- First for 2 processes and 2 threads ---------------------------"
mpiexec -n 4 ./Part_3/ektelesimo 100000000 2
sleep ${sleep_time_between_results}
echo -e "------------------------- Second for 4 processes and 4 threads ---------------------------"
mpiexec -n 4 ./Part_3/ektelesimo 100000000 4
sleep ${sleep_time}






