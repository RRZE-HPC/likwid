/*
 * =======================================================================================
 *
 *      Filename:  registers_types.h
 *
 *      Description:  Header File of registers.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#ifndef REGISTERS_TYPES_H
#define REGISTERS_TYPES_H

#include <pci_types.h>

typedef enum {
    PMC0 = 0,
    PMC1, PMC2, PMC3, PMC4, PMC5, PMC6,
    PMC7, PMC8, PMC9, PMC10, PMC11, PMC12,
    PMC13, PMC14, PMC15, PMC16, PMC17, PMC18,
    PMC19, PMC20, PMC21, PMC22, PMC23, PMC24,
    PMC25, PMC26, PMC27, PMC28, PMC29, PMC30,
    PMC31, PMC32, PMC33, PMC34, PMC35, PMC36,
    PMC37, PMC38, PMC39, PMC40, PMC41, PMC42,
    PMC43, PMC44, PMC45, PMC46, PMC47, PMC48,
    PMC49, PMC50, PMC51, PMC52, PMC53, PMC54,
    PMC55, PMC56, PMC57, PMC58, PMC59, PMC60,
    PMC61, PMC62, PMC63, PMC64, PMC65, PMC66,
    PMC67, PMC68, PMC69, PMC70, PMC71, PMC72,
    PMC73, PMC74, PMC75, PMC76, PMC77, PMC78,
    PMC79, PMC80, PMC81, PMC82, PMC83, PMC84,
    PMC85, PMC86, PMC87, PMC88, PMC89, PMC90,
    PMC91, PMC92, PMC93, PMC94, PMC95, PMC96,
    PMC97, PMC98, PMC99, PMC100, PMC101, PMC102,
    PMC103, PMC104, PMC105, PMC106, PMC107, PMC108,
    PMC109, PMC110, PMC111, PMC112, PMC113, PMC114,
    PMC115, PMC116, PMC117, PMC118, PMC119, PMC120,
    PMC121, PMC122, PMC123, PMC124, PMC125, PMC126,
    PMC127, PMC128, PMC129, PMC130, PMC131, PMC132,
    NUM_PMC
} RegisterIndex;

typedef enum {
    PMC = 0, FIXED, THERMAL,
    POWER, UNCORE, MBOX0,
    MBOX1, MBOX2, MBOX3,
    MBOX0FIX, MBOX1FIX, MBOX2FIX,
    MBOX3FIX, BBOX0, BBOX1,
    RBOX0, RBOX1, RBOX2,
    WBOX, WBOX0FIX, WBOX1FIX,
    SBOX0, SBOX1, SBOX2, SBOX3,
    SBOX0FIX, SBOX1FIX, SBOX2FIX, SBOX3FIX,
    CBOX0, CBOX1, CBOX2,
    CBOX3, CBOX4, CBOX5,
    CBOX6, CBOX7, CBOX8,
    CBOX9, CBOX10, CBOX11,
    CBOX12, CBOX13, CBOX14,
    CBOX15, CBOX16, CBOX17,
    PBOX, UBOX, UBOXFIX,
    NUM_UNITS, NOTYPE, MAX_UNITS
} RegisterType;

static char* RegisterTypeNames[MAX_UNITS] = {
    [PMC] = "Core-local general purpose counters",
    [FIXED] = "Fixed counters",
    [THERMAL] = "Thermal",
    [POWER] = "Power",
    [UNCORE] = "Uncore",
    [MBOX0] = "Memory Box 0",
    [MBOX1] = "Memory Box 1",
    [MBOX2] = "Memory Box 2",
    [MBOX3] = "Memory Box 3",
    [MBOX0FIX] = "Memory box fixed counter 0",
    [MBOX1FIX] = "Memory box fixed counter 1",
    [MBOX2FIX] = "Memory box fixed counter 2",
    [MBOX3FIX] = "Memory box fixed counter 3",
    [BBOX0] = "Home Agent box 0",
    [BBOX1] = "Home Agent box 1",
    [RBOX0] = "Routing box 0",
    [RBOX1] = "Routing box 1",
    [WBOX] = "Power control box",
    [WBOX0FIX] = "Power control box fixed counter 0",
    [WBOX0FIX] = "Power control box fixed counter 1",
    [SBOX0] = "QPI Link Layer box 0",
    [SBOX1] = "QPI Link Layer box 1",
    [SBOX2] = "QPI Link Layer box 2",
    [SBOX3] = "QPI Link Layer box 3",
    [SBOX0FIX] = "QPI Link Layer box fixed 0",
    [SBOX1FIX] = "QPI Link Layer box fixed 1",
    [SBOX2FIX] = "QPI Link Layer box fixed 2",
    [SBOX3FIX] = "QPI Link Layer box fixed 3",
    [CBOX0] = "Caching Agent box 0",
    [CBOX1] = "Caching Agent box 1",
    [CBOX2] = "Caching Agent box 2",
    [CBOX3] = "Caching Agent box 3",
    [CBOX4] = "Caching Agent box 4",
    [CBOX5] = "Caching Agent box 5",
    [CBOX6] = "Caching Agent box 6",
    [CBOX7] = "Caching Agent box 7",
    [CBOX8] = "Caching Agent box 8",
    [CBOX9] = "Caching Agent box 9",
    [CBOX10] = "Caching Agent box 10",
    [CBOX11] = "Caching Agent box 11",
    [CBOX12] = "Caching Agent box 12",
    [CBOX13] = "Caching Agent box 13",
    [CBOX14] = "Caching Agent box 14",
    [CBOX15] = "Caching Agent box 15",
    [CBOX16] = "Caching Agent box 16",
    [CBOX17] = "Caching Agent box 17",
    [PBOX] = "Physical Layer box",
    [UBOX] = "System Configuration box",
    [UBOXFIX] = "System Configuration box fixed counter",
    [NUM_UNITS] = "Maximally usable register types",
    [NOTYPE] = "No Type, used for skipping unavailable counters"
};

#define REG_TYPE_MASK(type) (type < NUM_UNITS ? (0x1ULL<<type) : 0x0ULL)

typedef struct {
    char*               key;
    RegisterIndex       index;
    RegisterType        type;
    uint64_t            configRegister;
    uint64_t            counterRegister;
    uint64_t            counterRegister2;
    PciDeviceIndex      device;
    uint64_t            optionMask;
} RegisterMap;

typedef struct {
    uint32_t  ctrlRegister;
    uint32_t  statusRegister;
    uint32_t  ovflRegister;
    uint8_t   isPci;
    PciDeviceIndex device;
    uint32_t  regWidth;
} BoxMap;

#endif
