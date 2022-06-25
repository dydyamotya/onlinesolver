SLAVE_ID = 11
import threading
from pymodbus.client.sync import ModbusSerialClient
import time
import logging

logger = logging.getLogger(__name__)

class ModbusThread(threading.Thread):
    def __init__(self, modbus_port, frame, timeout=10):
        super(ModbusThread, self).__init__()

        self.frame = frame

        self.daemon = True
        self.stopEvent = threading.Event()

        self.timeout = timeout

        self.modbus_serial = ModbusSerialClient(method="rtu", port=modbus_port, baudrate=19200, timeout=self.timeout)
        self.start()




    def run(self):
        while not self.stopEvent.is_set():
            time.sleep(self.timeout)
            try:
                coils = self.modbus_serial.read_coils(0x0000, count=4, unit=SLAVE_ID).bits[:4]
                self.frame.print_gasstatus(str(self.transform_to_gas_number(coils)))
            except Exception as e:
                self.stop_thread(message=str(e))
            else:
                logger.info("{}".format(self.transform_to_gas_number(coils)))

    def stop_thread(self, message="Modbus stopped"):
        self.stopEvent.set()
        self.frame.notice_stop_modbus(message)

    def is_stopped(self):
        return self.stopEvent.is_set()

    @staticmethod
    def transform_to_gas_number(values):
        return values.index(True)