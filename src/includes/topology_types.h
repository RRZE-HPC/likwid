/*
 * =======================================================================================
 *
 *      Filename:  topology_types.h
 *
 *      Description:  Types file for topology module. External definitions are
 *                    in likwid.h
 *
 *      Version:   4.3.4
 *      Released:  05.04.2019
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */
#ifndef CPUID_TYPES_H
#define CPUID_TYPES_H

/** \addtogroup CPUTopology CPU information module
*  @{
*/
/*! \brief Enum of possible CPU features

CPUs implement different features that likely improve application performance if
optimized using the feature. The list contains all features that are currently
supported by LIKWID. LIKWID does not perform any action based on these features,
it gathers the data only for output purposes. It is not a complete list.
\extends CpuInfo
*/
typedef enum {
    SSE3=0, /*!< \brief Streaming SIMD Extensions 3 */
    MMX, /*!< \brief Multi Media Extension */
    SSE, /*!< \brief Streaming SIMD Extensions */
    SSE2, /*!< \brief Streaming SIMD Extensions 2 */
    MONITOR, /*!< \brief MONITOR and MWAIT instructions (part of SSE3) */
    ACPI, /*!< \brief Advanced Configuration and Power Interface */
    RDTSCP, /*!< \brief Serializing Read of the Time Stamp Counter */
    VMX, /*!< \brief Virtual Machine eXtensions (VT-x) */
    EIST, /*!< \brief Enhanced Intel SpeedStep */
    TM, /*!< \brief Thermal Monitor */
    TM2, /*!< \brief Thermal Monitor 2 */
    AES, /*!< \brief AES instruction set */
    RDRAND, /*!< \brief Random numbers from an on-chip hardware random number generator */
    SSSE3, /*!< \brief Supplemental Streaming SIMD Extensions 3 */
    SSE41, /*!< \brief Streaming SIMD Extensions 4.1 */
    SSE42, /*!< \brief Streaming SIMD Extensions 4.2 */
    AVX, /*!< \brief Advanced Vector Extensions */
    FMA, /*!< \brief Fused multiply-add (FMA3) */
    AVX2, /*!< \brief Advanced Vector Extensions 2 */
    RTM, /*!< \brief Restricted Transactional Memory */
    HLE, /*!< \brief Hardware Lock Elision */
    HTT, /*!< \brief Hyper-Threading Technology */
    RDSEED, /*!< \brief Non-deterministic random bit generator */
    AVX512, /*!< \brief 512-bit wide vector registers for Advanced Vector Extensions */
} FeatureBit;
/** @}*/
#endif /*CPUID_TYPES_H*/
