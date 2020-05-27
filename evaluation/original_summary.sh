seg="500 1000 1500 2000"

for i in $seg
do
 ./eval_summary_cli --sp-directory /home/ubuntu/exp/original_result/$i --img-directory /home/ubuntu/superpixel-benchmark/data/BSDS500/images/all --gt-directory /home/ubuntu/superpixel-benchmark/data/BSDS500/csv_groundTruth/all
done
