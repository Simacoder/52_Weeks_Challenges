import ctypes
import os

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL("COMPILED_LIB.so")

    def __init__(self):
        
        data, shape = self.flatten(data)
        self.data_ctype = (ctypes.c_float * len(data))(*data)
        self.shape_ctype = (ctypes.c_int * len(shape))(*shape)
        self.ndim_ctype = ctypes.c_int(len(shape))
       
        self.shape = shape
        self.ndim = len(shape)

        Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)

        self.tensor = Tensor._C.create_tensor(
            self.data_ctype,
            self.shape_ctype,
            self.ndim_ctype,
        )
        
    def flatten(self, nested_list):
        """
        This method simply convert a list type tensor to a flatten tensor with its shape
        
        Example:
        
        Arguments:  
            nested_list: [[1, 2, 3], [-5, 2, 0]]
        Return:
            flat_data: [1, 2, 3, -5, 2, 0]
            shape: [2, 3]
        """
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape
        
        flat_data, shape = flatten_recursively(nested_list)
        return flat_data, shape
    


def __getitem__(self, indices):
    """
    Access tensor by index tensor[i, j, k...]
    """

    if len(indices) != self.ndim:
        raise ValueError("Number of indices must match the number of dimensions")
    
    Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
    Tensor._C.get_item.restype = ctypes.c_float
                                       
    indices = (ctypes.c_int * len(indices))(*indices)
    value = Tensor._C.get_item(self.tensor, indices)  
    
    return value

def reshape(self, new_shape):
    """
    Reshape tensor
    result = tensor.reshape([1,2])
    """
    new_shape_ctype = (ctypes.c_int * len(new_shape))(*new_shape)
    new_ndim_ctype = ctypes.c_int(len(new_shape))
    
    Tensor._C.reshape_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Tensor._C.reshape_tensor.restype = ctypes.POINTER(CTensor)
    result_tensor_ptr = Tensor._C.reshape_tensor(self.tensor, new_shape_ctype, new_ndim_ctype)   

    result_data = Tensor()
    result_data.tensor = result_tensor_ptr
    result_data.shape = new_shape.copy()
    result_data.ndim = len(new_shape)
    result_data.device = self.device

    return result_data

def __add__(self, other):
    """
    Add tensors
    result = tensor1 + tensor2
    """
  
    if self.shape != other.shape:
        raise ValueError("Tensors must have the same shape for addition")
    
    Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

    result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

    result_data = Tensor()
    result_data.tensor = result_tensor_ptr
    result_data.shape = self.shape.copy()
    result_data.ndim = self.ndim
    result_data.device = self.device

    return result_data


def to(self, device):
    self.device = device
    self.device_ctype = self.device.encode('utf-8')
  
    Tensor._C.to_device.argtypes = [ctypes.POINTER(CTensor), ctypes.c_char_p]
    Tensor._C.to_device.restype = None
    Tensor._C.to_device(self.tensor, self.device_ctype)
  
    return self

def __add__(self, other):
    
  if self.shape != other.shape:
      raise ValueError("Tensors must have the same shape for addition")
  
  Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
  Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)
  
  result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)
  
  result_data = Tensor()
  result_data.tensor = result_tensor_ptr
  result_data.shape = self.shape.copy()
  result_data.ndim = self.ndim
  result_data.device = self.device
  
  result_data.requires_grad = self.requires_grad or other.requires_grad
  if result_data.requires_grad:
      result_data.grad_fn = AddBackward(self, other)
def backward(self, gradient=None):
    if not self.requires_grad:
        return
    
    if gradient is None:
        if self.shape == [1]:
            gradient = Tensor([1]) # dx/dx = 1 case
        else:
            raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")

    if self.grad is None:
        self.grad = gradient

    else:
        self.grad += gradient

    if self.grad_fn is not None: # not a leaf
        grads = self.grad_fn.backward(gradient) # call the operation backward
        for tensor, grad in zip(self.grad_fn.input, grads):
            if isinstance(tensor, Tensor):
                tensor.backward(grad) # recursively call the backward again for the gradient expression (chain rule)


def zero_grad(self):
    self.grad = None

def detach(self):
    self.grad = None
    self.grad_fn = None