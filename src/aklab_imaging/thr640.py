import configparser
import time
import serial
import logging

"""
Define Global Variables
"""
CR = "\r\n"
PORT = "COM3"
AXIS = "1"
STARTING_SPEED = "20"
OPERATING_SPEED = "10000"
ACCELERATION_PATTERN = "1 50"
ROTATIONAL_DIRECTION = "+"

"""
Define Commands
"""
H = "H{} {}".format(AXIS, ROTATIONAL_DIRECTION)
RAMP = "RAMP{} {}".format(AXIS, ACCELERATION_PATTERN)
V = "V{} {}".format(AXIS, OPERATING_SPEED)
VS = "VS{} {}".format(AXIS, STARTING_SPEED)
## Clear current position
RTNCR = "RTNCR{}".format(AXIS)
## Performs a continuous operation
SCAN = "SCAN{}".format(AXIS)
## Execute a mechanical home seeking
MHOME = "MHOME{}".format(AXIS)
## Performs an absolute operation until the current position matches the specified position.
ABS = "ABS{}".format(AXIS)
## End sequence programs
END = "END"
## Get status
R = "R{}".format(AXIS)

"""
configure logging
"""
logger = logging.getLogger("Logging")
logger.setLevel(50)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class THR640:
    def __init__(self, port=PORT, config_file=None):
        self.position = None
        self._port = port
        self._open_port()
        if config_file is None:
            self._config = None
        else:
            self._config = configparser.ConfigParser()
            # self._config.open(config_file)

    def __del__(self):
        self.ser.close()

    def get_configuration(self):
        """
        get controller`s configuration
        """
        print('THR640: getting config over Serial Port')
        self.ser.write((R + CR).encode("utf-8"))
        logger.info("getting information...")
        time.sleep(0.1)
        return self._readline()

    def goto(
        self, count: int = None,
    ):
        """
        Move the moter to the required position.
        Wait until the processing by the controller is done. 
        """
        self._send_goto(count)
        self.position = count

    def waitUntilReady(self):
        """
        wait until check ready is true
        """
        is_ready = self._check_ready()
        while is_ready == False:
            is_ready = self._check_ready()
            time.sleep(0.1)

    def _readline(self):
        """print response from controller"""
        lines = self.ser.readlines()
        lines = [line.strip().decode("utf-8") for line in lines]
        return lines

    def _check_ready(self) -> bool:
        """ returns if the spectrometer is ready """
        self._get_status()
        lines = self.ser.readlines()

        for line in lines:
            stripline = line.strip().decode("utf-8").split()
            if "Idle" in stripline:
                return True
            elif "Busy" in stripline:
                return False
        return False

    def _get_status(self):
        self.ser.write((R + CR).encode("utf-8"))
        logger.info("getting status...")

    def _send_goto(self, count):
        """ send a command to go to the specified count """
        if count < 0:
            r_direction = "-"
        else:
            r_direction = "+"
        count = abs(count)
        D = "D{} {} {}".format(AXIS, r_direction, count)

        self.waitUntilReady()

        logger.info("Move to {}{}".format(r_direction, count))
        self.ser.write((D + CR).encode("utf-8"))
        time.sleep(0.1)
        self.ser.write((ABS + CR).encode("utf-8"))

    def _open_port(self, port=PORT, baudrate=9600, timeout=3):
        """get connection"""
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        if self.ser.is_open == False:
            try:
                self.ser.open()
            except:
                raise Exception("cannot open port!!!!")

        ## MEMO: time.sleepがないとうまくいかない
        self.ser.write((H + CR).encode("utf-8"))
        time.sleep(0.1)
        self.ser.write((V + CR).encode("utf-8"))
        time.sleep(0.1)
        self.ser.write((VS + CR).encode("utf-8"))
        time.sleep(0.1)
        self.ser.write((RAMP + CR).encode("utf-8"))
        time.sleep(0.1)
