#mrslic experiment
import  os
for shape in ['h','w']:
    for segments in ['500','1000','1500','2000']:
        dir_path = '/home/ubuntu/exp/grid/'+shape+'/'+segments
        file_list = os.listdir(dir_path)
        for overlap in file_list:
            os.system('./eval_average_cli --summary-file /home/ubuntu/exp/grid/'+shape+'/'+segments+'/'+overlap+'/summary.csv')

