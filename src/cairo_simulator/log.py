import logging
import os

if os.environ.get('ROS_DISTRO'):
    import rospy


class Logger():

    def __init__(self, handlers=['std'], level='debug'):
        self.level = level
        self.handlers = handlers
        if 'logging' == handlers:
            levels = {
                "info": logging.INFO,
                "debug": logging.DEBUG,
                "warn": logging.WARN,
                "err": logging.ERROR,
                "crit": logging.CRITICAL
            }
            logging.basicConfig(level=levels[self.level])

    def add_handler(self, handle):
        self.handlers.append(handle)

    def debug(self, msg):
        if 'std' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logdebug(msg)
        if 'logging' in self.handlers:
            logging.debug(msg)

    def info(self, msg):
        if 'std' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.loginfo(msg)
        if 'logging' in self.handlers:
            logging.info(msg)

    def warn(self, msg):
        if 'std' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logwarn(msg)
        if 'logging' in self.handlers:
            logging.warning(msg)

    def err(self, msg):
        if 'std' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logerr(msg)
        if 'logging' in self.handlers:
            logging.error(msg)

    def crit(self, msg):
        if 'std' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logfatal(msg)
        if 'logging' in self.handlers:
            logging.critical(msg)