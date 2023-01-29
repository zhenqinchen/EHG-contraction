
class Config:
    PRE_ = 'E:/code/iup/github'
    ROOT_DIR = PRE_

    ICELAND_DIR ='D:/research/data/iceland'
    ICELAND_RESOURCE = PRE_ + '/resource'
    
  
    def __init__(self):
        return
    

# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()


_C.PARAM = CN()

_C.PARAM.DATABASE = 'iceland'
_C.PARAM.CHANNEL = [9,10]

_C.PARAM.FS = 6
_C.PARAM.FILTER = [0.1,3]
_C.PARAM.FILTER_TYPE = 0
_C.PARAM.NORMALIZE = False
_C.PARAM.IS_ADD_DUR = False

_C.DETECT = CN()
_C.DETECT.METHOD_NAME = 'sample_entropy'

_C.DETECT.WINDOW_SIZE = 60
_C.DETECT.SHIFT = 20
_C.DETECT.BASE_WINDOW_SIZE = 90
_C.DETECT.BASE_SHIFT = 20
_C.DETECT.MEAN_FILTER_SIZE = 10
_C.DETECT.THRESHOLD_TYPE = 0




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def get_iup_filename(config):
    file_pre = Config.ICELAND_RESOURCE #+ 
    channel_str = str(config.PARAM.CHANNEL[0]) + '_' + str(config.PARAM.CHANNEL[1])
    return file_pre + '/iup_' + config.DETECT.METHOD_NAME + '_'+ channel_str + '_' +  str(config.DETECT.BASE_WINDOW_SIZE)+ '_' + str(config.DETECT.WINDOW_SIZE) +'_' + str(config.DETECT.SHIFT) +'_' + str(int(config.PARAM.NORMALIZE))+'_' + str(config.DETECT.MEAN_FILTER_SIZE) +'_'+ str(config.PARAM.FILTER[0])+'-'+str(config.PARAM.FILTER[1])+'-' + str(config.PARAM.FS) +'.pkl'

        