/*
 * =======================================================================================
 *
 *      Filename:  registers_types.h
 *
 *      Description:  Header File of registers.
 *
 *      Version:   4.3.3
 *      Released:  09.11.2018
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
    PMC133, PMC134, PMC135, PMC136, PMC137, PMC138,
    PMC139, PMC140, PMC141, PMC142, PMC143, PMC144,
    PMC145, PMC146, PMC147, PMC148, PMC149, PMC150,
    PMC151, PMC152, PMC153, PMC154, PMC155, PMC156,
    PMC157, PMC158, PMC159, PMC160, PMC161, PMC162,
    PMC163, PMC164, PMC165, PMC166, PMC167, PMC168,
    PMC169, PMC170, PMC171, PMC172, PMC173, PMC174,
    PMC175, PMC176, PMC177, PMC178, PMC179, PMC180,
    PMC181, PMC182, PMC183, PMC184, PMC185, PMC186,
    PMC187, PMC188, PMC189, PMC190, PMC191, PMC192,
    PMC193, PMC194, PMC195, PMC196, PMC197, PMC198,
    PMC199, PMC200, PMC201, PMC202, PMC203, PMC204,
    PMC205, PMC206, PMC207, PMC208, PMC209, PMC210,
    PMC211, PMC212, PMC213, PMC214, PMC215, PMC216,
    PMC217, PMC218, PMC219, PMC220, PMC221, PMC222,
    PMC223, PMC224, PMC225, PMC226, PMC227, PMC228,
    PMC229, PMC230, PMC231, PMC232, PMC233, PMC234,
    PMC235, PMC236, PMC237, PMC238, PMC239, PMC240,
    PMC241, PMC242, PMC243, PMC244, PMC245, PMC246,
    PMC247, PMC248, PMC249, PMC250, PMC251, PMC252,
    PMC253, PMC254, PMC255, PMC256, PMC257, PMC258,
    PMC259, PMC260, PMC261, PMC262, PMC263, PMC264,
    PMC265, PMC266, PMC267, PMC268, PMC269, PMC270,
    PMC271, PMC272, PMC273, PMC274, PMC275, PMC276,
    PMC277, PMC278, PMC279, PMC280, PMC281, PMC282,
    PMC283, PMC284, PMC285, PMC286, PMC287, PMC288,
    PMC289, PMC290, PMC291, PMC292, PMC293, PMC294,
    PMC295, PMC296, PMC297, PMC298, PMC299, PMC300,
    PMC301, PMC302, PMC303, PMC304, PMC305, PMC306,
    PMC307, PMC308, PMC309, PMC310, PMC311, PMC312,
    PMC313, PMC314, PMC315, PMC316, PMC317, PMC318,
    PMC319, PMC320, PMC321, PMC322, PMC323, PMC324,
    PMC325, PMC326, PMC327, PMC328, PMC329, PMC330,
    PMC331, PMC332, PMC333, PMC334, PMC335, PMC336,
    PMC337, PMC338, PMC339, PMC340, PMC341, PMC342,
    PMC343, PMC344, PMC345, PMC346, PMC347, PMC348,
    NUM_PMC
} RegisterIndex;

typedef enum {
    PMC = 0, FIXED, THERMAL,
    POWER, UNCORE, MBOX0,
    MBOX1, MBOX2, MBOX3,
    MBOX4, MBOX5, MBOX6, MBOX7,
    MBOX0FIX, MBOX1FIX, MBOX2FIX,
    MBOX3FIX, MBOX4FIX, MBOX5FIX,
    MBOX6FIX, MBOX7FIX,
    BBOX0, BBOX1,
    RBOX0, RBOX1, RBOX2,
    WBOX,
    WBOX0FIX, WBOX1FIX,
    SBOX0, SBOX1, SBOX2, SBOX3,
    SBOX0FIX, SBOX1FIX, SBOX2FIX, SBOX3FIX,
    CBOX0, CBOX1, CBOX2,
    CBOX3, CBOX4, CBOX5,
    CBOX6, CBOX7, CBOX8,
    CBOX9, CBOX10, CBOX11,
    CBOX12, CBOX13, CBOX14,
    CBOX15, CBOX16, CBOX17,
    CBOX18, CBOX19, CBOX20,
    CBOX21, CBOX22, CBOX23,
    CBOX24, CBOX25, CBOX26,
    CBOX27, CBOX28, CBOX29,
    CBOX30, CBOX31, CBOX32,
    CBOX33, CBOX34, CBOX35,
    CBOX36, CBOX37,
    PBOX, PBOX1, PBOX2, PBOX3,
    UBOX,
    UBOXFIX,
    IBOX0, IBOX1, IBOX2, IBOX3, IBOX4, IBOX5,
    IBOX0FIX, IBOX1FIX, IBOX2FIX, IBOX3FIX, IBOX4FIX, IBOX5FIX,
    QBOX0, QBOX1, QBOX2,
    QBOX0FIX, QBOX1FIX, QBOX2FIX,
    EUBOX0, EUBOX0FIX, EUBOX1, EUBOX1FIX,
    EUBOX2, EUBOX2FIX, EUBOX3, EUBOX3FIX,
    EUBOX4, EUBOX4FIX, EUBOX5, EUBOX5FIX,
    EUBOX6, EUBOX6FIX, EUBOX7, EUBOX7FIX,
    EDBOX0, EDBOX0FIX, EDBOX1, EDBOX1FIX,
    EDBOX2, EDBOX2FIX, EDBOX3, EDBOX3FIX,
    EDBOX4, EDBOX4FIX, EDBOX5, EDBOX5FIX,
    EDBOX6, EDBOX6FIX, EDBOX7, EDBOX7FIX,
    NUM_UNITS, NOTYPE, MAX_UNITS
} RegisterType;

static char* RegisterTypeNames[MAX_UNITS] = {
    [PMC] = "Core-local general purpose counters",
    [FIXED] = "Fixed counters",
    [THERMAL] = "Thermal",
    [POWER] = "Energy/Power counters (RAPL)",
    [UNCORE] = "Socket-local general/fixed purpose counters",
    [MBOX0] = "Memory Controller 0 Channel 0",
    [MBOX1] = "Memory Controller 0 Channel 1",
    [MBOX2] = "Memory Controller 0 Channel 2",
    [MBOX3] = "Memory Controller 0 Channel 3",
    [MBOX4] = "Memory Controller 1 Channel 0",
    [MBOX5] = "Memory Controller 1 Channel 1",
    [MBOX6] = "Memory Controller 1 Channel 2",
    [MBOX7] = "Memory Controller 1 Channel 3",
    [MBOX0FIX] = "Memory Controller 0 Channel 0 Fixed Counter",
    [MBOX1FIX] = "Memory Controller 0 Channel 1 Fixed Counter",
    [MBOX2FIX] = "Memory Controller 0 Channel 2 Fixed Counter",
    [MBOX3FIX] = "Memory Controller 0 Channel 3 Fixed Counter",
    [MBOX4FIX] = "Memory Controller 1 Channel 0 Fixed Counter",
    [MBOX5FIX] = "Memory Controller 1 Channel 1 Fixed Counter",
    [MBOX6FIX] = "Memory Controller 1 Channel 2 Fixed Counter",
    [MBOX7FIX] = "Memory Controller 1 Channel 3 Fixed Counter",
    [BBOX0] = "Home Agent box 0",
    [BBOX1] = "Home Agent box 1",
    [RBOX0] = "Routing box 0",
    [RBOX1] = "Routing box 1",
    [RBOX2] = "Routing box 2",
    [WBOX] = "Power control box",
    [WBOX0FIX] = "Power control box fixed counter 0",
    [WBOX1FIX] = "Power control box fixed counter 1",
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
    [CBOX18] = "Caching Agent box 18",
    [CBOX19] = "Caching Agent box 19",
    [CBOX20] = "Caching Agent box 20",
    [CBOX21] = "Caching Agent box 21",
    [CBOX22] = "Caching Agent box 22",
    [CBOX23] = "Caching Agent box 23",
    [CBOX24] = "Caching Agent box 24",
    [CBOX25] = "Caching Agent box 25",
    [CBOX26] = "Caching Agent box 26",
    [CBOX27] = "Caching Agent box 27",
    [CBOX28] = "Caching Agent box 28",
    [CBOX29] = "Caching Agent box 29",
    [CBOX30] = "Caching Agent box 30",
    [CBOX31] = "Caching Agent box 31",
    [CBOX32] = "Caching Agent box 32",
    [CBOX33] = "Caching Agent box 33",
    [CBOX34] = "Caching Agent box 34",
    [CBOX35] = "Caching Agent box 35",
    [CBOX36] = "Caching Agent box 36",
    [CBOX37] = "Caching Agent box 37",
    [PBOX] = "Physical Layer box",
    [PBOX1] = "Physical Layer box",
    [PBOX2] = "Physical Layer box",
    [PBOX3] = "Physical Layer box",
    [UBOX] = "System Configuration box",
    [UBOXFIX] = "System Configuration box fixed counter",
    [IBOX0] = "Coherency Maintainer for IIO traffic",
    [IBOX1] = "Coherency Maintainer for IIO traffic",
    [IBOX2] = "Coherency Maintainer for IIO traffic",
    [IBOX3] = "Coherency Maintainer for IIO traffic",
    [IBOX4] = "Coherency Maintainer for IIO traffic",
    [IBOX5] = "Coherency Maintainer for IIO traffic",
    [QBOX0] = "QPI Link Layer 0",
    [QBOX1] = "QPI Link Layer 1",
    [QBOX0FIX] = "QPI Link Layer rate status 0",
    [QBOX1FIX] = "QPI Link Layer rate status 1",
    [EUBOX0] = "Embedded DRAM controller 0",
    [EUBOX1] = "Embedded DRAM controller 1",
    [EUBOX2] = "Embedded DRAM controller 2",
    [EUBOX3] = "Embedded DRAM controller 3",
    [EUBOX4] = "Embedded DRAM controller 4",
    [EUBOX5] = "Embedded DRAM controller 5",
    [EUBOX6] = "Embedded DRAM controller 6",
    [EUBOX7] = "Embedded DRAM controller 7",
    [EUBOX0FIX] = "Embedded DRAM controller 0 fixed counter",
    [EUBOX1FIX] = "Embedded DRAM controller 1 fixed counter",
    [EUBOX2FIX] = "Embedded DRAM controller 2 fixed counter",
    [EUBOX3FIX] = "Embedded DRAM controller 3 fixed counter",
    [EUBOX4FIX] = "Embedded DRAM controller 4 fixed counter",
    [EUBOX5FIX] = "Embedded DRAM controller 5 fixed counter",
    [EUBOX6FIX] = "Embedded DRAM controller 6 fixed counter",
    [EUBOX7FIX] = "Embedded DRAM controller 7 fixed counter",
    [EDBOX0] = "Embedded DRAM controller 0",
    [EDBOX1] = "Embedded DRAM controller 1",
    [EDBOX2] = "Embedded DRAM controller 2",
    [EDBOX3] = "Embedded DRAM controller 3",
    [EDBOX4] = "Embedded DRAM controller 4",
    [EDBOX5] = "Embedded DRAM controller 5",
    [EDBOX6] = "Embedded DRAM controller 6",
    [EDBOX7] = "Embedded DRAM controller 7",
    [EDBOX0FIX] = "Embedded DRAM controller 0 fixed counter",
    [EDBOX1FIX] = "Embedded DRAM controller 1 fixed counter",
    [EDBOX2FIX] = "Embedded DRAM controller 2 fixed counter",
    [EDBOX3FIX] = "Embedded DRAM controller 3 fixed counter",
    [EDBOX4FIX] = "Embedded DRAM controller 4 fixed counter",
    [EDBOX5FIX] = "Embedded DRAM controller 5 fixed counter",
    [EDBOX6FIX] = "Embedded DRAM controller 6 fixed counter",
    [EDBOX7FIX] = "Embedded DRAM controller 7 fixed counter",
    [NUM_UNITS] = "Maximally usable register types",
    [NOTYPE] = "No Type, used for skipping unavailable counters"
};

#define REG_TYPE_MASK(type) (type < NUM_UNITS ? ((1ULL)<<(type)) : 0x0ULL)

#define TESTTYPE(eventset, type) \
        (((type) >= 0 && (type) <= 63 ? eventset->regTypeMask1 & (1ULL<<(type)) : \
        ((type) >= 64 && (type) <= 127 ? eventset->regTypeMask2 & (1ULL<<((type)-64)) : \
        ((type) >= 128 && (type) <= 191 ? eventset->regTypeMask3 & (1ULL<<((type)-128)) : \
        ((type) >= 192 && (type) <= 255 ? eventset->regTypeMask4 & (1ULL<<((type)-192)) : 0x0ULL)))))

#define SETTYPE(eventset, type) \
        if ((type) >= 0 && (type) <= 63) \
        { \
            eventset->regTypeMask1 |= (1ULL<<(type)); \
        } \
        else if ((type) >= 64 && (type) <= 127) \
        { \
            eventset->regTypeMask2 |= (1ULL<<((type)-64)); \
        } \
        else if ((type) >= 128 && (type) <= 191) \
        { \
            eventset->regTypeMask3 |= (1ULL<<((type)-128)); \
        } \
        else if ((type) >= 192 && (type) <= 255) \
        { \
            eventset->regTypeMask4 |= (1ULL<<((type)-192)); \
        } \
        else \
        { \
            ERROR_PRINT(Cannot set out-of-bounds type %d, (type)); \
        }
#define MEASURE_CORE(eventset) \
        (eventset->regTypeMask1 & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))

#define MEASURE_UNCORE(eventset) \
        (eventset->regTypeMask1 & ~(REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(THERMAL)|REG_TYPE_MASK(POWER)) || eventset->regTypeMask2 || eventset->regTypeMask3 || eventset->regTypeMask4)


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
    int       ovflOffset;
    uint8_t   isPci;
    PciDeviceIndex device;
    uint32_t  regWidth;
    uint32_t  filterRegister1;
    uint32_t  filterRegister2;
} BoxMap;

#endif /* REGISTERS_TYPES_H */
