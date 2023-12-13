import logging
import sys
import os
import time
from .util import multisplit

def logger(args):
    args_str = "{}".format(args)
    args_str = multisplit(['(', ')'], args_str)
    args_str = args_str[1].replace(', ', '_')

    logger = logging.getLogger('VQA')  # logging name
    logger.setLevel(logging.DEBUG)  # 接收DEBUG即以上的log info
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    now = time.localtime(time.time())
    now_str = "{0}.{1}_{2}.{3}.{4}".format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    record_dir = "./log/{}_{}".format(now.tm_mon, now.tm_mday)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    record_file = os.path.join(record_dir,"{}_at_{}.log".format(args.v,now_str))

    fh = logging.FileHandler(record_file)  # log info 输入到文件
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)  # log info 输入到屏幕
    sh.setLevel(logging.DEBUG)

    fmt = '[%(asctime)-15s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)  # 设置每条info开头格式
    logger.addHandler(fh)  # 把FileHandler/StreamHandler加入logger
    logger.addHandler(sh)
    logger.debug(args_str)

    return logger, "{}_at_{}".format(args.v,now_str)
