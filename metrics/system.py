import psutil
from pynvml import *

nvmlInit()


class CPU:
    @staticmethod
    def get_percent():
        return psutil.cpu_percent()

    @staticmethod
    def get_mem():
        return psutil.virtual_memory().percent

    @staticmethod
    def get_percent_per_core():
    	return psutil.cpu_percent(percpu=True)


class GPU:
    @staticmethod
    def _handle():
        return nvmlDeviceGetHandleByIndex(0)

    @staticmethod
    def get_utilization():
        handle = GPU._handle()
        util = nvmlDeviceGetUtilizationRates(handle)
        return util.gpu

    @staticmethod
    def get_memory_percent():
        handle = GPU._handle()
        mem = nvmlDeviceGetMemoryInfo(handle)
        return round((mem.used / mem.total) * 100, 2)

    @staticmethod
    def get_memory_used_mb():
        handle = GPU._handle()
        mem = nvmlDeviceGetMemoryInfo(handle)
        return round(mem.used / (1024 * 1024), 2)

    @staticmethod
    def get_memory_total_mb():
        handle = GPU._handle()
        mem = nvmlDeviceGetMemoryInfo(handle)
        return round(mem.total / (1024 * 1024), 2)

    @staticmethod
    def get_temperature():
        handle = GPU._handle()
        return nvmlDeviceGetTemperature(
            handle,
            NVML_TEMPERATURE_GPU
        )

    @staticmethod
    def get_name():
        handle = GPU._handle()
        return nvmlDeviceGetName(handle)


if __name__ == "__main__":
    print(GPU.get_memory_used_mb())