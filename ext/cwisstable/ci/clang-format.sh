#!/bin/bash

EXIT_CODE=0
for file in $(find . \( -iname '*.h' -o -iname '*.c' -o -iname '*.cc' \)); do
  fmt=$(clang-format $file)
  fmt_exit=$?
  if [[ $fmt_exit != 0 ]]; then
    EXIT_CODE="$fmt_exit"
  fi
  diff -u $file <(cat <<< "$fmt")
done
exit $EXIT_CODE