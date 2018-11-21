from unrealcv import client
from unrealcv.util import read_png, read_npy
import time


class FPSCounter:
    def __init__(self):
        pass
    
    def tick(self):
        self.counter += 1

    def __enter__(self):
        self.reset()
        
    def __exit__(self, exception_type, exception_value, trackback):
        self.stop()

    def reset(self):
        self.counter = 0
        self.start_time = time.time()

    def stop(self):
        self.duration = time.time()  - self.start_time

    def __str__(self):
        perf_str = '%.2f frames in %.2f seconds, %.2f FPS' \
            % (self.counter, self.duration, self.counter/self.duration)
        return perf_str

class TaskRunner:
    def __init__(self):
        pass

    def run_time_limit(self, func, time_limit):
        ''' time_limit : how many seconds '''
        start_time = time.time()
        fps_counter = FPSCounter()
        with fps_counter:
            while time.time() - start_time < time_limit:
                func()
                fps_counter.tick()
        print(str(fps_counter))

    def run_count_limit(self, func, count_limit):
        fps_counter = FPSCounter()
        with fps_counter:
            for v in range(count_limit):
                func()
                fps_counter.tick()
        print(str(fps_counter))

def main():
    def get_lit_bmp():
        client.request('vget /camera/0/lit bmp')
    def get_lit_png():
        client.request('vget /camera/0/lit png')
    def get_lit_fast_bmp():
        client.request('vget /camera/0/lit_fast bmp')
    def get_lit_fast_png():
        client.request('vget /camera/0/lit_fast png')
    def read_png_func():
        res = client.request('vget /camera/0/lit png')
        im = read_png(res)

    client.connect()
    task_runner = TaskRunner()
    task_runner.run_time_limit(get_lit_bmp, 5)
    task_runner.run_time_limit(get_lit_png, 5)
    task_runner.run_time_limit(get_lit_fast_bmp, 5)
    task_runner.run_time_limit(get_lit_fast_png, 5)
    task_runner.run_time_limit(read_png_func, 5)

if __name__ == '__main__':
    main()