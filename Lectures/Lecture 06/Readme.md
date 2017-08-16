## How to download sources?  
cd ~/  
mkdir sources  
cd sources  
git clone git@github.com:ulstu/  robotics_ml.git  
cd robotics_ml/Lectures/Lecture\ 06/  
cp my_pkg ~/catkin_ws/src  
cd ~/catkin_ws  
catkin_make  
roslaunch my_pkg my_pkg.launch   
