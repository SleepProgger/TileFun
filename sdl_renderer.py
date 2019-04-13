'''
Created on Apr 4, 2019

@author: nope
'''

import sys
import sdl2
import sdl2.ext
import numpy as np
import threading
from time import time as now, sleep

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = (800, 600)

sdl2.ext.init()

class SDLRenderer(threading.Thread):
    def __init__(self, title, window_size, img_size):
        # TODO: flags und den ganzen bs
        super(SDLRenderer, self).__init__()
        self.window_height, self.window_width = window_size
        self.img_size = img_size
        self.window = sdl2.ext.Window(title, size=(self.window_width, self.window_height), flags=sdl2.SDL_WINDOW_RESIZABLE)
        self.renderer = sdl2.ext.Renderer(self.window)
        self.running = threading.Event()
        self.fps = 0
        self.i = 0
        self._need_update = True
        self._lock = threading.Lock() # TODO: think about this
        self._pix_written = threading.Event()
        self._pix_written.set()
        self.texture, self._pixel, self._bytes_per_pix = self.setup_colors()
        self.target_fps = 30
        # TODO: use double buffering ?

    # TODO: toogle_fullscreen

    def stop(self):
        self.running.clear()

    def setup_colors(self):
        texture = sdl2.SDL_CreateTexture(
            self.renderer.sdlrenderer,
            sdl2.SDL_PIXELFORMAT_BGRA32,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            self.img_size[1], self.img_size[0]
        )
        pixel = np.zeros(self.img_size + (4,), dtype=np.uint8)
        return texture, pixel, 4

    def update(self, pixel, wait_for_write=True):
        logger.debug("Updating")
        if not self.running.is_set():
            raise Exception("Bin kaputt")

        if wait_for_write:
            self._pix_written.wait()
        with self._lock:
            self._pix_written.clear() # TODO: if...
            self._pixel[:] = pixel
            self._need_update = True

    def handle_keypress(self, event):
        # TODO
        pass

    def handle_event(self, event):
        if event.type in (sdl2.SDL_KEYUP, sdl2.SDL_KEYDOWN):
            return self.handle_keypress(event)
        if event.type in (sdl2.SDL_MOUSEMOTION,):
            return
        # TODO: only on resize
        self._need_update = True
        

    def run(self):
        self.running.set()
        stime = now()
        last_update = now()
        target_fps = self.target_fps / 2. # todo: hu, why do we need / 2 here ?
        target_delay = 1/float(target_fps)
        i = 0
        while self.running.is_set():
            events = sdl2.ext.get_events()
            for event in events:
                self.handle_event(event)
                if event.type == sdl2.SDL_QUIT:
                    self.running.clear()
                    break

            current = now()
            elapsed = current - last_update
            last_update = current
            if self._need_update:
                i += 1
                if i >= target_fps * 2:
                    self.i += i
                    elapsed2 = current - stime
                    self.fps = float(i) / elapsed2
                    i = 0
                    stime = current
                with self._lock:
                    #sdl2.SDL_RenderClear(self.renderer.sdlrenderer)
                    sdl2.SDL_UpdateTexture(self.texture, None, self._pixel.ctypes.data, self.img_size[1] * self._bytes_per_pix)
                    sdl2.SDL_RenderCopy(self.renderer.sdlrenderer, self.texture, None, None)
                    sdl2.SDL_RenderPresent(self.renderer.sdlrenderer)
                    self._pix_written.set() # TODO: if ...
                self._need_update = False
                self.i += 1
            
            if target_delay > elapsed:
                #print("Times", target_delay, elapsed, int((target_delay - elapsed) * 1000))
                #print("%.2f" % (i / (current - stime)))
                sdl2.SDL_Delay(int((target_delay - elapsed) * 1000))


def run():
    renderer = Blinky((HEIGHT, WIDTH))
    renderer.start()
    
    def handle_key(event):
        #print(type(event), dir(event))
        if event.type != sdl2.events.SDL_KEYUP:
            return
        if event.key.keysym.sym == 27:
            renderer.stop()

        print(event.key.keysym.sym)
        
        
    renderer.handle_event = handle_key
    
    pixel = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    try:
        for y in range(HEIGHT):
            for x in range(WIDTH):
                pixel[:y, :x, :] = 255
                renderer.update(pixel)
        renderer.stop()
    except:
        pass
    finally:
        renderer.join()
    return 80085

if __name__ == "__main__":
    sys.exit(run())
