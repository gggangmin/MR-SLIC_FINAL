seg="500 1000 1500 2000"

for i in $seg
do
 ./eval_average_cli --summary-file /home/ubuntu/exp/original_result/$i/summary.csv
done
