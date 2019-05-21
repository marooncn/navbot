### turtlebot_description

a revised compact version of official turtlebot_description package to check if collision happened  between turtlebot and obstacles in Gazebo by adding libgazebo_ros_bumper.so plugin in [kobuki_gazebo.urdf.xacro](https://github.com/marooncn/navbot/blob/master/turtlebot_description/urdf/kobuki/kobuki_gazebo.urdf.xacro). The original file is from [turtlebot_description](http://wiki.ros.org/turtlebot_description) and [kobuki_description](http://wiki.ros.org/kobuki_description) package.

* Other changes

The width and height of image captured is scaled to 64 and 48 respectively. The size can be change in [turtlebot_gazebo.urdf.xacro](https://github.com/marooncn/navbot/blob/master/turtlebot_description/urdf/turtlebot_gazebo.urdf.xacro).

* To use it you need delete the official turtlebot_description
      
      sudo apt-get remove ros-kinetic-turtlebot-description
       
