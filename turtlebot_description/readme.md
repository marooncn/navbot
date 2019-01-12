### turtlebot_description
       a revised compact version of official turtlebot_description package to check if collision happened 
       between turtlebot and obstacles in Gazebo by adding libgazebo_ros_bumper.so plugin in kobuki_gazebo
       .urdf.xacro file. The original file is from turtlebot_description and kobuki_description package.
* The other changes

       the width and height of image captured is scaled to 64 and 48 respectively. the size can be change 
       in ./urdf/turtlebot_gazebo.urdf.xacro
* To use it you need delete the official turtlebot_description
      
      sudo apt-get remove ros-kinetic-turtlebot-description
       
