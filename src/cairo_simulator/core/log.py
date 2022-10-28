import logging
import os
import sys

if os.environ.get('ROS_DISTRO'):
    import rospy

class Logger():

    def __init__(self, name="cairo_sim", handlers=['stdout'], level='debug'):
        self.level = level
        self.handlers = handlers
        if 'logging' in handlers:
            levels = {
                "info": logging.INFO,
                "debug": logging.DEBUG,
                "warn": logging.WARN,
                "err": logging.ERROR,
                "crit": logging.CRITICAL
            }
            
            self.logger = logging.getLogger(name)
            formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')
            self.logger.setLevel(levels[self.level])
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler(stream=sys.stdout)
                handler.setLevel(levels[self.level])
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        for handler in self.handlers:
            if handler not in ['stdout', 'logging', 'ros']:
                raise ValueError("Handlers must be one of 'stdout', 'logging', or 'ros'")

    def add_handler(self, handle):
        self.handlers.append(handle)

    def debug(self, msg=None):
        msg = '' if msg is None else msg
        if 'stdout' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logdebug(msg)
        if 'logging' in self.handlers:
            self.logger.debug(msg)

    def info(self, msg):
        msg = '' if msg is None else msg
        if 'stdout' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.loginfo(msg)
        if 'logging' in self.handlers:
            self.logger.info(msg)

    def warn(self, msg):
        msg = '' if msg is None else msg
        if 'stdout' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logwarn(msg)
        if 'logging' in self.handlers:
            self.logger.warning(msg)

    def err(self, msg):
        msg = '' if msg is None else msg
        if 'stdout' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logerr(msg)
        if 'logging' in self.handlers:
            self.logger.error(msg)

    def crit(self, msg):
        msg = '' if msg is None else msg
        if 'stdout' in self.handlers:
            print(msg)
        if 'ros' in self.handlers:
            rospy.logfatal(msg)
        if 'logging' in self.handlers:
            self.logger.critical(msg)