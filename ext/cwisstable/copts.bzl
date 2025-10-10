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

# C/C++ compiler options selection. Taken from Abseil.

CWISS_CLANG_CL_FLAGS = [
    "/W3",
    "/DNOMINMAX",
    "/DWIN32_LEAN_AND_MEAN",
    "/D_CRT_SECURE_NO_WARNINGS",
    "/D_SCL_SECURE_NO_WARNINGS",
    "/D_ENABLE_EXTENDED_ALIGNED_STORAGE",
]

CWISS_CLANG_CL_TEST_FLAGS = [
    "-Wno-c99-extensions",
    "-Wno-deprecated-declarations",
    "-Wno-missing-noreturn",
    "-Wno-missing-prototypes",
    "-Wno-missing-variable-declarations",
    "-Wno-null-conversion",
    "-Wno-shadow",
    "-Wno-shift-sign-overflow",
    "-Wno-sign-compare",
    "-Wno-unused-function",
    "-Wno-unused-member-function",
    "-Wno-unused-parameter",
    "-Wno-unused-private-field",
    "-Wno-unused-template",
    "-Wno-used-but-marked-unused",
    "-Wno-zero-as-null-pointer-constant",
    "-Wno-gnu-zero-variadic-macro-arguments",
]

CWISS_GCC_FLAGS = [
    "-Werror",
    "-Wall",
    "-Wextra",
    "-Wcast-qual",
    # The cc1 driver whines about this in C mode.
    # "-Wconversion-null",
    "-Wformat-security",
    "-Wmissing-declarations",
    "-Woverlength-strings",
    "-Wpointer-arith",
    "-Wundef",
    "-Wunused-local-typedefs",
    "-Wunused-result",
    "-Wvarargs",
    "-Wvla",
    "-Wwrite-strings",
    "-DNOMINMAX",
]

CWISS_GCC_TEST_FLAGS = [
    "-Wno-conversion-null",
    "-Wno-deprecated-declarations",
    "-Wno-missing-declarations",
    "-Wno-sign-compare",
    "-Wno-unused-function",
    "-Wno-unused-parameter",
    "-Wno-unused-private-field",
]

CWISS_LLVM_FLAGS = [
    "-Werror",
    "-Wall",
    "-Wextra",
    "-Wcast-qual",
    "-Wconversion",
    "-Wfloat-overflow-conversion",
    "-Wfloat-zero-conversion",
    "-Wfor-loop-analysis",
    "-Wformat-security",
    "-Wgnu-redeclared-enum",
    "-Winfinite-recursion",
    "-Winvalid-constexpr",
    "-Wliteral-conversion",
    "-Wmissing-declarations",
    "-Woverlength-strings",
    "-Wpointer-arith",
    "-Wself-assign",
    "-Wshadow-all",
    "-Wstring-conversion",
    "-Wtautological-overlap-compare",
    "-Wundef",
    "-Wuninitialized",
    "-Wunreachable-code",
    "-Wunused-comparison",
    "-Wunused-local-typedefs",
    "-Wunused-result",
    "-Wvla",
    "-Wwrite-strings",
    "-Wno-float-conversion",
    "-Wno-implicit-float-conversion",
    "-Wno-implicit-int-float-conversion",
    "-Wno-implicit-int-conversion",
    "-Wno-shorten-64-to-32",
    "-Wno-sign-conversion",
    "-Wno-unknown-warning-option",
    "-DNOMINMAX",
]

CWISS_LLVM_TEST_FLAGS = [
    "-Wno-c99-extensions",
    "-Wno-deprecated-declarations",
    "-Wno-missing-noreturn",
    "-Wno-missing-prototypes",
    "-Wno-missing-variable-declarations",
    "-Wno-null-conversion",
    "-Wno-shadow",
    "-Wno-shift-sign-overflow",
    "-Wno-sign-compare",
    "-Wno-unused-function",
    "-Wno-unused-member-function",
    "-Wno-unused-parameter",
    "-Wno-unused-private-field",
    "-Wno-unused-template",
    "-Wno-used-but-marked-unused",
    "-Wno-zero-as-null-pointer-constant",
    "-Wno-gnu-zero-variadic-macro-arguments",
    # Make sure we get all the SIMD features on tests with Clang.
    "-march=native",
]

CWISS_MSVC_FLAGS = [
    "/W3",
    "/DNOMINMAX",
    "/DWIN32_LEAN_AND_MEAN",
    "/D_CRT_SECURE_NO_WARNINGS",
    "/D_SCL_SECURE_NO_WARNINGS",
    "/D_ENABLE_EXTENDED_ALIGNED_STORAGE",
    "/bigobj",
    "/wd4005",
    "/wd4068",
    "/wd4180",
    "/wd4244",
    "/wd4267",
    "/wd4503",
    "/wd4800",
]

CWISS_MSVC_LINKOPTS = [
    "-ignore:4221",
]

CWISS_MSVC_TEST_FLAGS = [
    "/wd4018",
    "/wd4101",
    "/wd4503",
    "/wd4996",
    "/DNOMINMAX",
]

CWISS_LLVM_SANTIZER_FLAGS = [
    "-fsanitize=address",
]

CWISS_DEFAULT_COPTS = select({
    "//:msvc_compiler": CWISS_MSVC_FLAGS,
    "//:clang-cl_compiler": CWISS_CLANG_CL_FLAGS,
    "//:clang_compiler": CWISS_LLVM_FLAGS,
    "//conditions:default": CWISS_GCC_FLAGS,
})

CWISS_TEST_COPTS = CWISS_DEFAULT_COPTS + select({
    "//:msvc_compiler": CWISS_MSVC_TEST_FLAGS,
    "//:clang-cl_compiler": CWISS_CLANG_CL_TEST_FLAGS,
    "//:clang_compiler": CWISS_LLVM_TEST_FLAGS,
    "//conditions:default": CWISS_GCC_TEST_FLAGS,
})

CWISS_SAN_COPTS = select({
    "//:clang_compiler": CWISS_LLVM_SANTIZER_FLAGS,
    "//conditions:default": [],
})

CWISS_DEFAULT_LINKOPTS = select({
    "//:msvc_compiler": CWISS_MSVC_LINKOPTS,
    "//conditions:default": [],
})

CWISS_CXX_VERSION = select({
    "//:msvc_compiler": ["/std:c++17"],
    "//:clang-cl_compiler": ["/std:c++17"],
    "//conditions:default": ["--std=c++17"],
})

CWISS_C_VERSION = select({
    "//:msvc_compiler": ["/std:c11"],
    "//:clang-cl_compiler": ["/std:c11"],
    "//conditions:default": ["--std=c11"],
})