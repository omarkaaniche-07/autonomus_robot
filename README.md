# autonomus_robot
Autonomous robot control for a robotics competition. A Raspberry Pi runs the match sequence from a start cord. It reads a wall number (1/2), identifies shapes (cube/sphere/pyramid) with YOLO-World, and reads the wall letters (E/W/S/N). It drives an STM32 for movement and an ESP32 for the actuators.
