from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset

from .dataset.table import Table
from .dataset.table_small import Table as Table_small


dataset_factory = {
  'table':Table,
  'table_small':Table_small
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'ctdet_small': CTDetDataset
}

def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
