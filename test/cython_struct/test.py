from test import main
import ctypes



if isinstance(user_data, ctypes.c_void_p):
    context = _get_ctypes_data(user_data)