import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTModel:
    def __init__(self, engine_path, img_size, mean, std):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std, dtype=np.float32)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def allocate_buffers(self):
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = (img - self.mean) / self.std
        return np.transpose(img, (2,0,1)).ravel()

    def infer(self, imgs):
        batch = np.vstack([self.preprocess(i) for i in imgs]).astype(np.float32)
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        self.context.execute_async_v2([int(self.d_input), int(self.d_output)], self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(len(imgs), -1)
