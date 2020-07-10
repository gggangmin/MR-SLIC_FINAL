#!/usr/bin/env bash
echo -e '\nexport PATH=$HOME/anaconda3/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc
# bind conda to spark
echo -e "\nexport PYSPARK_PYTHON=/home/hadoop/anaconda3/bin/python" >> /etc/spark/conf/spark-env.sh
echo "export PYSPARK_DRIVER_PYTHON=/home/hadoop/anaconda3/bin/jupyter" >> /etc/spark/conf/spark-env.sh
echo "export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --ip=0.0.0.0'" >> /etc/spark/conf/spark-env.sh
