#!/usr/bin/env python3
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
Generates an agglomerated header out of various other headers.
"""

import argparse
import sys

from pathlib import Path


class Header:
  def __init__(self, include_path, open_path):
    self.path = include_path
    self.fulltext = open_path.read_text()

    lines = []
    self.includes = []

    has_seen_licence = False
    ifdef_guard = None
    for line in self.fulltext.split('\n'):
      if not has_seen_licence and line.startswith('//'):
        continue
      has_seen_licence = True

      if ifdef_guard is None and line.startswith('#ifndef'):
        ifdef_guard = line[len('#ifndef'):].strip()
      elif (ifdef_guard is not None and
            (line == f'#define {ifdef_guard}' or
             line == f'#endif  // {ifdef_guard}')):
        pass
      elif line.startswith('#include'):
        self.includes.append(line[len('#include'):].strip())
      else:
        lines.append(line)

    self.text = '\n'.join(lines).strip()


def main():
  parser = argparse.ArgumentParser(description='agglomerate some headers')
  parser.add_argument(
    '--guard',
    type=str,
    default='CWISSTABLE_H_',
    help='include guard name for the agglomerated header'
  )
  parser.add_argument(
    '--out',
    type=argparse.FileType('w', encoding='utf-8'),
    default='cwisstable.h',
    help='location to output the file to'
  )
  parser.add_argument(
    '--include_dir',
    type=Path,
    default='.',
    help='directory path that "cwisstable/*.h" includes are relative to'
  )
  parser.add_argument(
    'hdrs',
    type=Path,
    default=[Path('cwisstable/policy.h'), Path('cwisstable/declare.h')],
    nargs='*',
    help='headers to agglomerate'
  )
  args = parser.parse_args()

  hdrs = {}
  include_dir = args.include_dir.resolve()
  for open_path in args.hdrs:
    include_path = (Path.cwd() / open_path).relative_to(include_dir)
    hdrs[include_path] = Header(include_path, open_path)

  while True:
    new_hdrs = []
    for hdr in hdrs.values():
      for inc in hdr.includes:
        include_path = Path(inc.strip('"'))
        open_path = include_dir / include_path

        if include_path not in hdrs and open_path.exists():
          new_hdrs.append(Header(include_path, open_path))
    if not new_hdrs:
      break
    for hdr in new_hdrs:
      hdrs[hdr.path] = hdr

  hdr_names = {f'"{name}"' for name, _ in hdrs.items()}
  external_includes = set()
  internal_includes = set()
  for hdr in hdrs.values():
    for inc in hdr.includes:
      if inc in hdr_names:
        internal_includes.add(inc.strip('"'))
      else:
        external_includes.add(inc)

  roots = [hdr for hdr in hdrs.values() if
           str(hdr.path) not in internal_includes]
  # Order the roots predictably; this script must be idempotent
  roots.sort(key=lambda h: h.path)

  if not roots:
    print("dependency cycle detected")
    return 1

  # Toposort the include dependencies.
  unsorted = dict(hdrs)
  sorted_hdrs = []
  while roots:
    hdr = roots.pop()
    sorted_hdrs.append(hdr)
    del unsorted[hdr.path]

    new_roots = []
    for inc in hdr.includes:
      if inc not in hdr_names:
        continue

      for other_hdr in unsorted.values():
        if inc in other_hdr.includes:
          break
      else:
        new_roots.append(hdrs[Path(inc.strip('"'))])
    new_roots.sort(key=lambda h: h.path)
    roots.extend(new_roots)
  sorted_hdrs.reverse()

  o = args.out

  # Add the license, cribbing from this file itself.
  for line in Path(__file__).read_text().split('\n')[1:]:
    if not line.startswith('#'):
      break
    o.write(line.replace('#', '//') + "\n")
  o.write("\n")

  o.write("// THIS IS A GENERATED FILE! DO NOT EDIT DIRECTLY!\n")
  o.write("// Generated using unify.py, by concatenating, in order:\n")
  for hdr in sorted_hdrs:
    o.write(f'// #include "{hdr.path}"\n')
  o.write("\n")

  # Add the include guards.
  o.write(f"#ifndef {args.guard}\n")
  o.write(f"#define {args.guard}\n")
  o.write("\n")

  # Add the external includes.
  for inc in sorted(external_includes):
    o.write(f"#include {inc}\n")
  o.write("\n")

  # Add each header.
  for hdr in sorted_hdrs:
    name = f"/// {hdr.path} /"
    name += '/' * (80 - len(name))
    o.write('\n'.join([name, hdr.text, name]) + "\n\n")

  # Add the include guard tail.
  o.write(f"#endif  // {args.guard}\n")


if __name__ == '__main__': sys.exit(main() or 0)
