# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//:copts.bzl",
    "CWISS_DEFAULT_COPTS",
    "CWISS_DEFAULT_LINKOPTS",
    "CWISS_C_VERSION",
)

cc_binary(
    name = "arraymap",
    srcs = ["arraymap.c"],
    deps = ["//:cwisstable"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_UNIFIED"],
)

cc_binary(
    name = "arraymap_split",
    srcs = ["arraymap.c"],
    deps = ["//:cwisstable_split"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_SPLIT"],
)

cc_binary(
    name = "hashset",
    srcs = ["hashset.c"],
    deps = ["//:cwisstable"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_UNIFIED"],
)

cc_binary(
    name = "hashset_split",
    srcs = ["hashset.c"],
    deps = ["//:cwisstable_split"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_SPLIT"],
)

cc_binary(
    name = "hashmap",
    srcs = ["hashmap.c"],
    deps = ["//:cwisstable"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_UNIFIED"],
)

cc_binary(
    name = "hashmap_split",
    srcs = ["hashmap.c"],
    deps = ["//:cwisstable_split"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_SPLIT"],
)

cc_binary(
    name = "stringmap",
    srcs = ["stringmap.c"],
    deps = ["//:cwisstable"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_UNIFIED"],
)

cc_binary(
    name = "stringmap_split",
    srcs = ["stringmap.c"],
    deps = ["//:cwisstable_split"],
    copts = CWISS_DEFAULT_COPTS + CWISS_C_VERSION,
    linkopts = CWISS_DEFAULT_LINKOPTS,
    defines = ["CWISS_EXAMPLE_SPLIT"],
)
