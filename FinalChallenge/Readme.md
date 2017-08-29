# Final competition
## Preparation
Before performing a task you need to download your world file that is based on WillowGarage world file, it  contains additional objects (spheres and cubes with different colors for each student).

Also you need to register on the form: https://goo.gl/HBTT8g
On the form you need to provide nickname for competition (you will send requests to server with your nickname).

## Task
* Make a map of WollowGarage model with gmapping (with keyboard teleop or write your own explorer with python). Save map to a file.
* With given map write python node for ros that allow robot to explore a map by given points (with amcl server and ActionLib client, example of code located in fist_robotics_labs package in lab2.py node).
* During movement you need to search for the given object (spheres and cubes with different colors for each student) and calculate a position of this object (you can get current position (x, y, w) of robot from 'lab2.py' and calculate coordinates from distance, distance is calculated from proportion of real size of object and it's size on picture)
	* Object recognition you can perform with opencv library
	* If you want to get additional scores you should perform object recognition with deep learning (keras). You can make training and test sets by saving images during map exploration (sample code located here: http://learn.turtlebot.com/2015/02/04/3/) and performing you own markup of the test sample.
* Send object's position to the server with the following code:
```python
import urllib2

def send_goal(nickname, x, y):
	tmpl = "http://summerschool.simcase.ru/challenge/sendgoal/?name={}&x={}&y={}"
	surl = tmpl.format(nickname, x, y)
	content = urllib2.urlopen(surl).read()
	print content
```
For every right object's position you will get scores. For wrong positions you will get penalty scores.   

Table with results is published on the page http://summerschool.simcase.ru/challenge/standings/