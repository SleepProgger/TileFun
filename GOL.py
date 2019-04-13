'''
Created on Apr 5, 2019

@author: nope
'''
import keras 

from sdl_renderer import SDLRenderer
import sdl2
import numpy as np
from scipy.signal import convolve2d
from numpy import uint8
import threading

#import keras
import plaidml
from plaidml import tile as ptile
from plaidml import op as pop
from plaidml.keras import backend as K
import logging
import six

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#logging.getLogger('plaidml').setLevel(logging.DEBUG)
sdl2.ext.init()

# TODO: wtf is there no uint8 in _NP_TYPES ????? TODO TODO TODO ?
plaidml._NP_TYPES[plaidml.DType.UINT8] = "uint8"


class CopyFunction(K._Function):
    def __init__(self, np_arrs, inputs, outputs, updates, name):
        K._Function.__init__(self, inputs, outputs, updates, name)
        self._np_arrs = np_arrs
    
    def __call__(self, inputs):
        for (name, val) in zip(self._input_names, inputs):
            if isinstance(val, six.integer_types):
                val = plaidml.Integer(val)
            elif isinstance(val, float):
                val = plaidml.Real(val)
            else:
                val = K.variable(val, dtype=self._input_types[name]).var
            self._invoker.set_input(name, val)
        tensors = [
            plaidml.Tensor(K._device(), self._invoker.get_output_shape(name))
            for name in self._output_names
        ]
        for (name, t) in zip(self._output_names, tensors):
            self._invoker.set_output(name, t)
        self._invoker.invoke()
        res = []
        for t, np_arr in zip(tensors, self._np_arrs):
            with t.mmap_current() as view:
                view.copy_to_ndarray(np_arr)
            res.append(np_arr)
        return res


class GolOp(ptile.Operation):
    def __init__(self, frame, kernel, rules=((3,), (2,3)), wrap=True):
        code = """
            function (I[Y, X, Z], K[KY, KX, Z], NULL, ONE, RGB) -> (O, ORGB){{
                {conv};
                O = {rules};
                ORGB = RGB * O;
            }}
        """
        # TODO: we could simplify this
        # TODO: add other rules
        rule_map = {
            ((3,), (2,3)): '( T < 12 ? (T == 3 ? ONE : NULL) : (T > 13 ? NULL : ONE) )',
            # labyrinthish
            ((3,), (2,3,4)): '( T < 12 ? (T == 3 ? ONE : NULL) : (T > 14 ? NULL : ONE) )',
        }
        if isinstance(rules, (list, tuple)):
            rules = rule_map[tuple(rules)]
        if wrap:
            # TODO:
            raise NotImplementedError("Wrap not implemented because no modulo")
        else:
            conv = 'T[y, x, z: Y, X, Z] = +(I[y -1 + ky, x -1 + kx, z] * K[ky, kx, z]), ky < KY, kx < KX'

        code = code.format(rules = rules, conv = conv)
        logger.debug("TILE CODE: %s:", code)
        rgb = np.zeros(frame.shape.dims[:-1] + (4,), dtype="uint8")
        rgb[:,:] = (255,255,255,255)
        super(GolOp, self).__init__(code, [
            ('I', frame), ('K', kernel),
            ('NULL', K.constant(np.array(0, dtype='uint8'), dtype='uint8')),
            ('ONE', K.constant(np.array(1, dtype='uint8'), dtype='uint8')),
            ('RGB', K.variable(rgb, dtype='uint8')),
        ],[
            ('O', frame.shape), ('ORGB', ptile.Shape(frame.shape.dtype, rgb.shape))
        ])


class GOL(SDLRenderer):
    def __init__(self, title, window_size, size):
        SDLRenderer.__init__(self, title, window_size, size)
        self._use_tile = True
        self._warp_mode = False
        self.size = size
        self.frame = np.zeros(self.size + (4,), dtype=np.uint8)
        self.target_fps = 300
        self.init_frame(0.3)
        self._refresh_event = threading.Lock()
        
        self._init_calcs()
        
    def init_frame(self, chance):
        self._blibs = np.zeros(self.size, dtype=np.uint8)
        if chance is not None:
            self._blibs[np.random.uniform(size=self.size) < chance] = 1
            if self._use_tile:
                self._init_calcs()
            
    def _init_calcs(self):
        if not self._use_tile:
            self.step_frame = self.step_frame_np
            return
        kernel = np.ones((3, 3, 1), dtype=np.uint8)
        kernel[1, 1, 0] = 10
        self.step_frame = self.step_frame_tile
        blibs = K.variable(self._blibs.reshape(self._blibs.shape + (1,)), dtype=self._blibs.dtype)
        tile_op = GolOp.function(blibs, K.constant(kernel, dtype=kernel.dtype), wrap=self._warp_mode)
        self._tile_func = CopyFunction([self.frame], [], [tile_op[1]], updates=[(blibs, tile_op[0])], name="Foo")

    def step_frame_tile(self):
        with self._refresh_event:
            res = self._tile_func([])[0]
            if self.i % self.target_fps == 0:
                alive = np.sum(self.frame[:,:,:1]) / 255
                self.window.title = "Game of life | Alive: %i, Iter: %i, FPS: %.2f" % (alive, self.i, self.fps)
        self.update(res)
        
    def step_frame_np(self):
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 10
        with self._refresh_event:
            conv = convolve2d(self._blibs, kernel, mode="same", boundary=("wrap" if self._warp_mode else "fill"), fillvalue=0)
            self._blibs[np.logical_or(conv < 12, conv > 13)] = 0
            self._blibs[conv == 3] = 1
            if self.i % self.target_fps == 0:
                self.window.title = "Alive: %i, Iter: %i, FPS: %.2f" % (np.sum(self._blibs), self.i, self.fps)
            self.frame[self._blibs == 1] = (255,255,255,255)
            self.frame[self._blibs == 0] = (0,0,0,0)
        self.update(self.frame)

    def handle_keypress(self, event):
        if event.type != sdl2.events.SDL_KEYUP:
            return
        if event.key.keysym.sym == 27:
            self.stop()

        if event.key.keysym.sym == 114: # r
            with self._refresh_event:
                self.init_frame(0.6)

if __name__ == '__main__':
    # TODO: cmd args
    
#     devs = plaidml.devices(K._ctx, limit=666, return_all=True)
#     for i, dev in enumerate(devs):
#         print("%i: %s" % (i, dev[0].description))
#     K._dev = plaidml.Device(K._ctx, devs[int(input("Select device: "))][0])
    
    wsize = (1080, 1920)
    gol = GOL("Game of life", wsize, (int(wsize[0]/1), int(wsize[1]/1)))
    gol.start()
    gol.running.wait()
    while gol.running.is_set():
        gol.step_frame()
    print("Waiting for shutdonw")
    gol.join()