declare -a arr=(5 10 15 20 25 30 35 40)
for k in "${arr[@]}"
do
   #Call python script that generates output of new run.pbs
   echo "Submit $k workers"
   python -i runSamplingWorkerExperiments.py $k 5  &
done
