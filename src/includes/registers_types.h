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
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
    PMC349, PMC350, PMC351, PMC352, PMC353, PMC354,
    PMC355, PMC356, PMC357, PMC358, PMC359,
    PMC360, PMC361, PMC362, PMC363, PMC364, PMC365, PMC366, PMC367, PMC368, PMC369,
    PMC370, PMC371, PMC372, PMC373, PMC374, PMC375, PMC376, PMC377, PMC378, PMC379,
    PMC380, PMC381, PMC382, PMC383, PMC384, PMC385, PMC386, PMC387, PMC388, PMC389,
    PMC390, PMC391, PMC392, PMC393, PMC394, PMC395, PMC396, PMC397, PMC398, PMC399,
    PMC400, PMC401, PMC402, PMC403, PMC404, PMC405, PMC406, PMC407, PMC408, PMC409,
    PMC410, PMC411, PMC412, PMC413, PMC414, PMC415, PMC416, PMC417, PMC418, PMC419,
    PMC420, PMC421, PMC422, PMC423, PMC424, PMC425, PMC426, PMC427, PMC428, PMC429,
    PMC430, PMC431, PMC432, PMC433, PMC434, PMC435, PMC436, PMC437, PMC438, PMC439,
    PMC440, PMC441, PMC442, PMC443, PMC444, PMC445, PMC446, PMC447, PMC448, PMC449,
    PMC450, PMC451, PMC452, PMC453, PMC454, PMC455, PMC456, PMC457, PMC458, PMC459,
    PMC460, PMC461, PMC462, PMC463, PMC464, PMC465, PMC466, PMC467, PMC468, PMC469,
    PMC470, PMC471, PMC472, PMC473, PMC474, PMC475, PMC476, PMC477, PMC478, PMC479,
    PMC480, PMC481, PMC482, PMC483, PMC484, PMC485, PMC486, PMC487, PMC488, PMC489,
    PMC490, PMC491, PMC492, PMC493, PMC494, PMC495, PMC496, PMC497, PMC498, PMC499,
    PMC500, PMC501, PMC502, PMC503, PMC504, PMC505, PMC506, PMC507, PMC508, PMC509,
    PMC510, PMC511, PMC512, PMC513, PMC514, PMC515, PMC516, PMC517, PMC518, PMC519,
    PMC520, PMC521, PMC522, PMC523, PMC524, PMC525, PMC526, PMC527, PMC528, PMC529,
    PMC530, PMC531, PMC532, PMC533, PMC534, PMC535, PMC536, PMC537, PMC538, PMC539,
    PMC540, PMC541, PMC542, PMC543, PMC544, PMC545, PMC546, PMC547, PMC548, PMC549,
    PMC550, PMC551, PMC552, PMC553, PMC554, PMC555, PMC556, PMC557, PMC558, PMC559,
    PMC560, PMC561, PMC562, PMC563, PMC564, PMC565, PMC566, PMC567, PMC568, PMC569,
    PMC570, PMC571, PMC572, PMC573, PMC574, PMC575, PMC576, PMC577, PMC578, PMC579,
    PMC580, PMC581, PMC582, PMC583, PMC584, PMC585, PMC586, PMC587, PMC588, PMC589,
    PMC590, PMC591, PMC592, PMC593, PMC594, PMC595, PMC596, PMC597, PMC598, PMC599,
    PMC600, PMC601, PMC602, PMC603, PMC604, PMC605, PMC606, PMC607, PMC608, PMC609,
    PMC610, PMC611, PMC612, PMC613, PMC614, PMC615, PMC616, PMC617, PMC618, PMC619,
    PMC620, PMC621, PMC622, PMC623, PMC624, PMC625, PMC626, PMC627, PMC628, PMC629,
    PMC630, PMC631, PMC632, PMC633, PMC634, PMC635, PMC636, PMC637, PMC638, PMC639,
    PMC640, PMC641, PMC642, PMC643, PMC644, PMC645, PMC646, PMC647, PMC648, PMC649,
    PMC650, PMC651, PMC652, PMC653, PMC654, PMC655, PMC656, PMC657, PMC658, PMC659,
    PMC660, PMC661, PMC662, PMC663, PMC664, PMC665, PMC666, PMC667, PMC668, PMC669,
    PMC670, PMC671, PMC672, PMC673, PMC674, PMC675, PMC676, PMC677, PMC678, PMC679,
    PMC680, PMC681, PMC682, PMC683, PMC684, PMC685, PMC686, PMC687, PMC688, PMC689,
    PMC690, PMC691, PMC692, PMC693, PMC694, PMC695, PMC696, PMC697, PMC698, PMC699,
    PMC700, PMC701, PMC702, PMC703, PMC704, PMC705, PMC706, PMC707, PMC708, PMC709,
    PMC710, PMC711, PMC712, PMC713, PMC714, PMC715, PMC716, PMC717, PMC718, PMC719,
    PMC720, PMC721, PMC722, PMC723, PMC724, PMC725, PMC726, PMC727, PMC728, PMC729,
    PMC730, PMC731, PMC732, PMC733, PMC734, PMC735, PMC736, PMC737, PMC738, PMC739,
    PMC740, PMC741, PMC742, PMC743, PMC744, PMC745, PMC746, PMC747, PMC748, PMC749,
    PMC750, PMC751, PMC752, PMC753, PMC754, PMC755, PMC756, PMC757, PMC758, PMC759,
    PMC760, PMC761, PMC762, PMC763, PMC764, PMC765, PMC766, PMC767, PMC768, PMC769,
    PMC770, PMC771, PMC772, PMC773, PMC774, PMC775, PMC776, PMC777, PMC778, PMC779,
    PMC780, PMC781, PMC782, PMC783, PMC784, PMC785, PMC786, PMC787, PMC788, PMC789,
    PMC790, PMC791, PMC792, PMC793, PMC794, PMC795, PMC796, PMC797, PMC798, PMC799,
    PMC800, PMC801, PMC802, PMC803, PMC804, PMC805, PMC806, PMC807, PMC808, PMC809,
    PMC810, PMC811, PMC812, PMC813, PMC814, PMC815, PMC816, PMC817, PMC818, PMC819,
    PMC820, PMC821, PMC822, PMC823, PMC824, PMC825, PMC826, PMC827, PMC828, PMC829,
    PMC830, PMC831, PMC832, PMC833, PMC834, PMC835, PMC836, PMC837, PMC838, PMC839,
    PMC840, PMC841, PMC842, PMC843, PMC844, PMC845, PMC846, PMC847, PMC848, PMC849,
    PMC850, PMC851, PMC852, PMC853, PMC854, PMC855, PMC856, PMC857, PMC858, PMC859,
    PMC860, PMC861, PMC862, PMC863, PMC864, PMC865, PMC866, PMC867, PMC868, PMC869,
    PMC870, PMC871, PMC872, PMC873, PMC874, PMC875, PMC876, PMC877, PMC878, PMC879,
    PMC880, PMC881, PMC882, PMC883, PMC884, PMC885, PMC886, PMC887, PMC888, PMC889,
    PMC890, PMC891, PMC892, PMC893, PMC894, PMC895, PMC896, PMC897, PMC898, PMC899,
    PMC900, PMC901, PMC902, PMC903, PMC904, PMC905, PMC906, PMC907, PMC908, PMC909,
    PMC910, PMC911, PMC912, PMC913, PMC914, PMC915, PMC916, PMC917, PMC918, PMC919,
    PMC920, PMC921, PMC922, PMC923, PMC924, PMC925, PMC926, PMC927, PMC928, PMC929,
    PMC930, PMC931, PMC932, PMC933, PMC934, PMC935, PMC936, PMC937, PMC938, PMC939,
    PMC940, PMC941, PMC942, PMC943, PMC944, PMC945, PMC946, PMC947, PMC948, PMC949,
    PMC950, PMC951, PMC952, PMC953, PMC954, PMC955, PMC956, PMC957, PMC958, PMC959,
    PMC960, PMC961, PMC962, PMC963, PMC964, PMC965, PMC966, PMC967, PMC968, PMC969,
    PMC970, PMC971, PMC972, PMC973, PMC974, PMC975, PMC976, PMC977, PMC978, PMC979,
    PMC980, PMC981, PMC982, PMC983, PMC984, PMC985, PMC986, PMC987, PMC988, PMC989,
    PMC990, PMC991, PMC992, PMC993, PMC994, PMC995, PMC996, PMC997, PMC998, PMC999,
    NUM_PMC
} RegisterIndex;

typedef enum {
    PMC = 0, FIXED, PERF, THERMAL, VOLTAGE, METRICS,
    POWER, UNCORE, MBOX0,
    MBOX1, MBOX2, MBOX3,
    MBOX4, MBOX5, MBOX6, MBOX7,
    MBOX8, MBOX9, MBOX10, MBOX11,
    MBOX12, MBOX13, MBOX14, MBOX15,
    MBOX0FIX, MBOX1FIX, MBOX2FIX,
    MBOX3FIX, MBOX4FIX, MBOX5FIX,
    MBOX6FIX, MBOX7FIX, MBOX8FIX,
    MBOX9FIX, MBOX10FIX, MBOX11FIX,
    MBOX12FIX, MBOX13FIX, MBOX14FIX, MBOX15FIX,
    MDEV0, MDEV1, MDEV2, MDEV3,
    MBOX0TMP,
    BBOX0, BBOX1, BBOX2, BBOX3,
    BBOX4, BBOX5, BBOX6, BBOX7,
    BBOX8, BBOX9, BBOX10, BBOX11,
    BBOX12, BBOX13, BBOX14, BBOX15,
    BBOX16, BBOX17, BBOX18, BBOX19,
    BBOX20, BBOX21, BBOX22, BBOX23,
    BBOX24, BBOX25, BBOX26, BBOX27,
    BBOX28, BBOX29, BBOX30, BBOX31,
    RBOX0, RBOX1, RBOX2, RBOX3,
    WBOX,
    WBOX0FIX, WBOX1FIX, WBOX2FIX, WBOX3FIX,
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
    CBOX36, CBOX37, CBOX38, CBOX39,
    CBOX40, CBOX41, CBOX42,
    CBOX43, CBOX44, CBOX45,
    CBOX46, CBOX47, CBOX48, CBOX49,
    CBOX50, CBOX51, CBOX52,
    CBOX53, CBOX54, CBOX55,
    CBOX56, CBOX57, CBOX58, CBOX59,
    PBOX, PBOX1, PBOX2, PBOX3, PBOX4, PBOX5, PBOX6, PBOX7, PBOX8, PBOX9, PBOX10, PBOX11, PBOX12, PBOX13, PBOX14, PBOX15,
    UBOX,
    UBOXFIX,
    IBOX0, IBOX1, IBOX2, IBOX3, IBOX4, IBOX5, IBOX6, IBOX7, IBOX8, IBOX9, IBOX10, IBOX11, IBOX12, IBOX13, IBOX14, IBOX15,
    IBOX0FIX, IBOX1FIX, IBOX2FIX, IBOX3FIX, IBOX4FIX, IBOX5FIX, IBOX6FIX, IBOX7FIX, IBOX8FIX, IBOX9FIX, IBOX10FIX, IBOX11FIX, IBOX12FIX, IBOX13FIX, IBOX14FIX, IBOX15FIX,
    IRP0, IRP1, IRP2, IRP3, IRP4, IRP5, IRP6, IRP7, IRP8, IRP9, IRP10, IRP11, IRP12, IRP13, IRP14, IRP15,
    QBOX0, QBOX1, QBOX2, QBOX3,
    QBOX0FIX, QBOX1FIX, QBOX2FIX, QBOX3FIX,
    EUBOX0, EUBOX0FIX, EUBOX1, EUBOX1FIX,
    EUBOX2, EUBOX2FIX, EUBOX3, EUBOX3FIX,
    EUBOX4, EUBOX4FIX, EUBOX5, EUBOX5FIX,
    EUBOX6, EUBOX6FIX, EUBOX7, EUBOX7FIX,
    EDBOX0, EDBOX0FIX, EDBOX1, EDBOX1FIX,
    EDBOX2, EDBOX2FIX, EDBOX3, EDBOX3FIX,
    EDBOX4, EDBOX4FIX, EDBOX5, EDBOX5FIX,
    EDBOX6, EDBOX6FIX, EDBOX7, EDBOX7FIX,
    MDF0, MDF1, MDF2, MDF3,
    MDF4, MDF5, MDF6, MDF7,
    MDF8, MDF9, MDF10, MDF11,
    MDF12, MDF13, MDF14, MDF15,
    MDF16, MDF17, MDF18, MDF19,
    MDF20, MDF21, MDF22, MDF23,
    MDF24, MDF25, MDF26, MDF27,
    MDF28, MDF29, MDF30, MDF31,
    MDF32, MDF33, MDF34, MDF35,
    MDF36, MDF37, MDF38, MDF39,
    MDF40, MDF41, MDF42, MDF43,
    MDF44, MDF45, MDF46, MDF47,
    MDF48, MDF49, MDF50, MDF51,
    HBM0, HBM0FIX, HBM1, HBM1FIX, HBM2, HBM2FIX, HBM3, HBM3FIX,
    HBM4, HBM4FIX, HBM5, HBM5FIX, HBM6, HBM6FIX, HBM7, HBM7FIX,
    HBM8, HBM8FIX, HBM9, HBM9FIX, HBM10, HBM10FIX, HBM11, HBM11FIX,
    HBM12, HBM12FIX, HBM13, HBM13FIX, HBM14, HBM14FIX, HBM15, HBM15FIX,
    HBM16, HBM16FIX, HBM17, HBM17FIX, HBM18, HBM18FIX, HBM19, HBM19FIX,
    HBM20, HBM20FIX, HBM21, HBM21FIX, HBM22, HBM22FIX, HBM23, HBM23FIX,
    HBM24, HBM24FIX, HBM25, HBM25FIX, HBM26, HBM26FIX, HBM27, HBM27FIX,
    HBM28, HBM28FIX, HBM29, HBM29FIX, HBM30, HBM30FIX, HBM31, HBM31FIX,
    NUM_UNITS, NOTYPE, MAX_UNITS
} RegisterType;

#define PBOX0 PBOX

static char* RegisterTypeNames[MAX_UNITS] = {
    [PMC] = "Core-local general purpose counters",
    [FIXED] = "Fixed counters",
    [PERF] = "Perf counters",
    [THERMAL] = "Thermal",
    [VOLTAGE] = "Voltage of hardware thread",
    [METRICS] = "Performance metrics provided by Intel systems starting with Intel Icelake",
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
    [MBOX0TMP] = "PP0/PP1 Temperature Sensor",
#ifdef _ARCH_PPC
    [BBOX0] = "Memory controller synchronous (port 0 & 1)",
    [BBOX1] = "Memory controller synchronous (port 2 & 3)",
#else
    [BBOX0] = "Home Agent box 0",
    [BBOX1] = "Home Agent box 1",
#endif
    [RBOX0] = "Routing box 0",
    [RBOX1] = "Routing box 1",
    [RBOX2] = "Routing box 2",
    [WBOX] = "Power control box",
    [WBOX0FIX] = "Power control box fixed counter 0",
    [WBOX1FIX] = "Power control box fixed counter 1",
    [WBOX2FIX] = "Power control box fixed counter 2",
    [WBOX3FIX] = "Power control box fixed counter 3",
#ifdef _ARCH_PPC
    [SBOX0] = "PowerBus",
#else
    [SBOX0] = "QPI Link Layer box 0",
#endif
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
    [CBOX38] = "Caching Agent box 38",
    [CBOX39] = "Caching Agent box 39",
    [CBOX40] = "Caching Agent box 40",
    [CBOX41] = "Caching Agent box 41",
    [CBOX42] = "Caching Agent box 42",
    [CBOX43] = "Caching Agent box 43",
    [CBOX44] = "Caching Agent box 44",
    [CBOX45] = "Caching Agent box 45",
    [CBOX46] = "Caching Agent box 46",
    [CBOX47] = "Caching Agent box 47",
    [CBOX48] = "Caching Agent box 48",
    [CBOX49] = "Caching Agent box 49",
    [CBOX50] = "Caching Agent box 50",
    [CBOX51] = "Caching Agent box 51",
    [CBOX52] = "Caching Agent box 52",
    [CBOX53] = "Caching Agent box 53",
    [CBOX54] = "Caching Agent box 54",
    [CBOX55] = "Caching Agent box 55",
    [CBOX56] = "Caching Agent box 56",
    [CBOX57] = "Caching Agent box 57",
    [CBOX58] = "Caching Agent box 58",
    [CBOX59] = "Caching Agent box 59",
    [PBOX] = "Physical Layer box 0",
    [PBOX1] = "Physical Layer box 1",
    [PBOX2] = "Physical Layer box 2",
    [PBOX3] = "Physical Layer box 3",
    [PBOX4] = "Physical Layer box 4",
    [PBOX5] = "Physical Layer box 5",
    [UBOX] = "System Configuration box",
    [UBOXFIX] = "System Configuration box fixed counter",
    [IBOX0] = "Coherency Maintainer for IIO traffic",
    [IBOX1] = "Coherency Maintainer for IIO traffic",
    [IBOX2] = "Coherency Maintainer for IIO traffic",
    [IBOX3] = "Coherency Maintainer for IIO traffic",
    [IBOX4] = "Coherency Maintainer for IIO traffic",
    [IBOX5] = "Coherency Maintainer for IIO traffic",
#ifdef _ARCH_PPC
    [QBOX0] = "Xlink 0",
    [QBOX1] = "Xlink 1",
    [QBOX2] = "Xlink 2",
#else
    [QBOX0] = "QPI Link Layer 0",
    [QBOX1] = "QPI Link Layer 1",
    [QBOX2] = "QPI Link Layer 2",
    [QBOX3] = "QPI Link Layer 3",
#endif
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
        ((type) >= 192 && (type) <= 255 ? eventset->regTypeMask4 & (1ULL<<((type)-192)) : \
        ((type) >= 256 && (type) <= 319 ? eventset->regTypeMask5 & (1ULL<<((type)-256)) : \
        ((type) >= 320 && (type) <= 383 ? eventset->regTypeMask6 & (1ULL<<((type)-320)) : 0x0ULL)))))))

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
        else if ((type) >= 256 && (type) <= 319) \
        { \
            eventset->regTypeMask5 |= (1ULL<<((type)-256)); \
        } \
        else if ((type) >= 320 && (type) <= 383) \
        { \
            eventset->regTypeMask6 |= (1ULL<<((type)-320)); \
        } \
        else \
        { \
            ERROR_PRINT(Cannot set out-of-bounds type %d, (type)); \
        }
#define MEASURE_CORE(eventset) \
        (eventset->regTypeMask1 & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(METRICS)))

#define MEASURE_METRICS(eventset) ((eventset)->regTypeMask1 & (REG_TYPE_MASK(METRICS))

#define MEASURE_UNCORE(eventset) \
        (eventset->regTypeMask1 & ~(REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(THERMAL)|REG_TYPE_MASK(VOLTAGE)|REG_TYPE_MASK(PERF)|REG_TYPE_MASK(POWER)|REG_TYPE_MASK(METRICS)) || eventset->regTypeMask2 || eventset->regTypeMask3 || eventset->regTypeMask4 || eventset->regTypeMask5 || eventset->regTypeMask6 )


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
