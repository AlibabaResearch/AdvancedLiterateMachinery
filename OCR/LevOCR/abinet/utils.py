import logging
import os
import time

import cv2
import numpy as np
import torch
import yaml
from matplotlib import colors
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import ConcatDataset

class CharsetMapper(object):
    """A simple class to map ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    """

    def __init__(self,
                 filename='',
                 max_length=30,
                 null_char=u'\u2591'):
        """Creates a lookup table.

        Args:
          filename: Path to charset file which maps characters to ids.
          max_sequence_length: The max length of ids and string.
          null_char: A unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.
        """
        self.null_char = null_char
        self.max_length = max_length

        self.label_to_char = self._read_charset(filename)
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)
 
    def _read_charset(self, filename):
        """Reads a charset definition from a tab separated text file.

        Args:
          filename: a path to the charset file.

        Returns:
          a dictionary with keys equal to character codes and values - unicode
          characters.
        """
        import re
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                m = pattern.match(line)
                assert m, f'Incorrect charset file. line #{i}: {line}'
                label = int(m.group(1)) + 1
                char = m.group(2)
                charset[label] = char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, '')

    def get_text(self, labels, length=None, padding=True, trim=False):
        """ Returns a string corresponding to a sequence of character ids.
        """
        length = length if length else self.max_length
        labels = [l.item() if isinstance(l, Tensor) else int(l) for l in labels]
        if padding:
            labels = labels + [self.null_label] * (length-len(labels))
        text = ''.join([self.label_to_char[label] for label in labels])
        if trim: text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length

        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return '0123456789'

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                valid_chars.append(c)
        return ''.join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.data_time = 0.
        self.data_diff = 0.
        self.data_total_time = 0.
        self.data_call = 0
        self.running_time = 0.
        self.running_diff = 0.
        self.running_total_time = 0.
        self.running_call = 0

    def tic(self):
        self.start_time = time.time()
        self.running_time = self.start_time

    def toc_data(self):
        self.data_time = time.time()
        self.data_diff = self.data_time - self.running_time
        self.data_total_time += self.data_diff
        self.data_call += 1

    def toc_running(self):
        self.running_time = time.time()
        self.running_diff = self.running_time - self.data_time
        self.running_total_time += self.running_diff
        self.running_call += 1

    def total_time(self):
        return self.data_total_time + self.running_total_time

    def average_time(self):
        return  self.average_data_time() + self.average_running_time()

    def average_data_time(self):
        return self.data_total_time / (self.data_call or 1)

    def average_running_time(self):
        return self.running_total_time / (self.running_call or 1)


class Logger(object):
    _handle = None
    _root = None

    @staticmethod
    def init(output_dir, name, phase):
        format = '[%(asctime)s %(filename)s:%(lineno)d %(levelname)s {}] ' \
                '%(message)s'.format(name)
        logging.basicConfig(level=logging.INFO, format=format)

        try: os.makedirs(output_dir)
        except: pass
        config_path = os.path.join(output_dir, f'{phase}.txt')
        Logger._handle = logging.FileHandler(config_path)
        Logger._root = logging.getLogger()

    @staticmethod
    def enable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.addHandler(Logger._handle)

    @staticmethod
    def disable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.removeHandler(Logger._handle)


class Config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'phase':
                        assert v in ['train', 'test']
                    if k == 'stage':
                        assert v in ['pretrain-vision', 'pretrain-language',
                                     'train-semi-super', 'train-super']
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        with open('abinet/template.yaml') as file:
            default_config_dict = yaml.load(file, Loader=yaml.FullLoader)
        __dict2attr(default_config_dict)
        __dict2attr(config_dict)
        self.global_workdir = os.path.join(self.global_workdir, self.global_name)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str

def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask-mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask,(image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:,:,:3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255 
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1]) 
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1-color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1-alpha, 0)

    return blended_img

def onehot(label, depth, device=None):
    """ 
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar

    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot

class MyDataParallel(nn.DataParallel):

    def gather(self, outputs, target_device):
        r"""
        Gathers tensors from different GPUs on a specified device
        (-1 means the CPU).
        """
        def gather_map(outputs):
            out = outputs[0]
            if isinstance(out, (str, int, float)):
                return out
            if isinstance(out, list) and isinstance(out[0], str):
                return [o for out in outputs for o in out]
            if isinstance(out, torch.Tensor):
                return torch.nn.parallel._functions.Gather.apply(target_device, self.dim, *outputs)
            if out is None:
                return None
            if isinstance(out, dict):
                if not all((len(out) == len(d) for d in outputs)):
                    raise ValueError('All dicts must have the same number of keys')
                return type(out)(((k, gather_map([d[k] for d in outputs]))
                                for k in out))
            return type(out)(map(gather_map, zip(*outputs)))

        # Recursive function calls like this create reference cycles.
        # Setting the function to None clears the refcycle.
        try:
            res = gather_map(outputs)
        finally:
            gather_map = None
        return res


class MyConcatDataset(ConcatDataset):
    def __getattr__(self, k): 
        return getattr(self.datasets[0], k)
