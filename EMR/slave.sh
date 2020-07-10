for name in ec2-44-234-117-81.us-west-2.compute.amazonaws.com ec2-44-234-21-188.us-west-2.compute.amazonaws.com ec2-44-229-31-6.us-west-2.compute.amazonaws.com ec2-44-233-220-53.us-west-2.compute.amazonaws.com ec2-34-223-23-161.us-west-2.compute.amazonaws.com ec2-100-20-156-225.us-west-2.compute.amazonaws.com ec2-34-223-83-211.us-west-2.compute.amazonaws.com ec2-44-234-110-103.us-west-2.compute.amazonaws.com
do
 echo $name
 tar -cp anaconda3 | ssh -i /home/hadoop/SLIC.pem hadoop@$name tar xvp -C /home/hadoop/
done
