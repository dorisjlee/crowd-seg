declare -a arr=(10 30 50 70 80 100)
for k in "${arr[@]}"
do
   # echo "Simulated experiment with Nregions = $k"
   # # Run with different parameters of number of regions
   # python main.py 10 $k 100 200
   echo "Simulated experiment with Nworker = $k"
   # Run with different parameters of number of regions
   python main.py $k 10 100 200
done