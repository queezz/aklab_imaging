"""
Initialize camera and motor controller.
Allow only one instance. 
Doesn't work with importlib.reload, so keeping it simple.
"""
from aklab_imaging.thr640 import THR640
from aklab_imaging.FLI import FLI
import aklab_imaging.textcolor as tc


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            print("Initializing spectrometer")
            instances[cls] = cls(*args, **kw)
        else:
            print(
                f"Spectrometer instance {tc.RED} already exists{tc.RESET}.\n"
                "To reconnect, use `Spectrometer.disconnect()`,\n"
                "then call `spectrometer = Spectrometer()` again."
            )

        return instances[cls]

    def reset():
        """Reset the singleton instance."""
        if cls in instances:
            print("Resetting spectrometer instance")
            del instances[cls]

    # Attach reset functionality to the class
    _singleton.reset = reset
    return _singleton


@singleton
class Spectrometer:
    """
    Usage:
    spm = Spectrometer()
    if reconnection is needed, use 
    spm.disconnect()
    Then spm = Spectrometer() works again.
    """

    def __init__(self):
        self.camera = FLI()
        self.grating_motor = THR640()
        self.position = None
        self.get_position()
        print(f"Spectrometer: camera and motor {tc.GREEN}connected, yay!{tc.RESET}")

    def get_position(self):
        """get grating motor position
        I'm using requested position as an actual position in THR640,
        this is erorr prone, maybe need to improve.
        """
        if self.grating_motor.position is None:
            lines = self.grating_motor.get_configuration()
            self.position = self.find_position(lines)
            self.grating_motor.position = self.position
        else: 
            self.position = self.grating_motor.position
        return self.position

    def find_position(self, lines):
        """ """
        for i, line in enumerate(lines):
            if line.strip() == "Position:":
                if i + 1 < len(lines) and "PC1 =" in lines[i + 1]:
                    return int(lines[i + 1].split("PC1 =")[1].strip())
        return None

    def disconnect(self):
        """Disconnect devices and reset singleton instance."""
        print("Disconnecting Spectrometer...")
        try:
            self.camera.FLIClose()
            self.grating_motor.__del__()
        except Exception as e:
            print(f"Error during disconnect: {e}")
        finally:
            Spectrometer.reset()
