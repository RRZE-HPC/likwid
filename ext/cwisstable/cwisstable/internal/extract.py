#!/bin/env python3
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script for generating the extract.h header's boilerplate.

Update it by running
$ cwisstable/internal/extract.py
"""

import os
from pathlib import Path

DEPTH = 64 
KEYS = [
  'obj_copy', 'obj_dtor',
  'key_hash', 'key_eq',
  'alloc_alloc', 'alloc_free',
  
  'slot_size', 'slot_align', 'slot_init',
  'slot_transfer', 'slot_get', 'slot_dtor',
  'modifiers',
]
FILE = Path(__file__).parent / 'extract.h'

def main():
  text = FILE.read_text()
  end = text.find('// !!!')

  lines = []
  lines.append(text[:end] + '// !!!')
  lines.append('')

  for key in KEYS:
    lines.append(f'#define CWISS_EXTRACT_{key}(key_, val_) CWISS_EXTRACT_{key}Z##key_')
    lines.append(f'#define CWISS_EXTRACT_{key}Z{key} CWISS_NOTHING, CWISS_NOTHING, CWISS_NOTHING')
  lines.append('')

  for i in range(0, DEPTH):
    lines.append(f'#define CWISS_EXTRACT{i:02X}(needle_, kv_, ...) CWISS_SELECT{i:02X}(needle_ kv_, CWISS_EXTRACT_VALUE, kv_, CWISS_EXTRACT{i+1:02X}, (needle_, __VA_ARGS__), CWISS_NOTHING)')
  lines.append('')
  for i in range(0, DEPTH):
    lines.append(f'#define CWISS_SELECT{i:02X}(x_, ...) CWISS_SELECT{i:02X}_(x_, __VA_ARGS__)')
  lines.append('')
  for i in range(0, DEPTH):
    lines.append(f'#define CWISS_SELECT{i:02X}_(ignored_, _call_, _args_, call_, args_, ...) call_ args_')
  lines.append('')
  lines.append('#endif  // CWISSTABLE_INTERNAL_EXTRACT_H_')

  FILE.write_text('\n'.join(lines) + '\n')

if __name__ == '__main__': main()