#include <pci_types.h>

// TODO We already have TOSTRING, is that not enough?
#define _MYSTRING(x) #x
#define MYSTRING(x) _MYSTRING((x))

const PciType pci_types[MAX_NUM_PCI_TYPES] = {
    [R3QPI] = {"R3QPI", "R3QPI is the interface between the Intel QPI Link Layer and the Ring."},
    [R2PCIE] = {"R2PCIE", "R2PCIe represents the interface between the Ring and IIO traffic to/from PCIe."},
    [IMC] = {"IMC", "The integrated Memory Controller provides the interface to DRAM and communicates to the rest of the uncore through the Home Agent."},
    [HA] = {"HA", "The HA is responsible for the protocol side of memory interactions."},
    [QPI] = {"QPI", "The Intel QPI Link Layer is responsible for packetizing requests from the caching agent on the way out to the system interface."},
    [IRP] = {"IRP", "IRP is responsible for maintaining coherency for IIO traffic e.g. crosssocket P2P."},
    [EDC] = {"EDC", "The Embedded DRAM controller is used for high bandwidth memory on the Xeon Phi (KNL)."},
};

const char *pci_device_names[MAX_NUM_PCI_DEVICES] = {
    [MSR_DEV] = MYSTRING(MSR_DEV),
    [MMIO_IMC_DEVICE_0_CH_0] = MYSTRING(MMIO_IMC_DEVICE_0_CH_0),
    [MMIO_IMC_DEVICE_0_CH_1] = MYSTRING(MMIO_IMC_DEVICE_0_CH_1),
    [MMIO_IMC_DEVICE_0_CH_2] = MYSTRING(MMIO_IMC_DEVICE_0_CH_2),
    [MMIO_IMC_DEVICE_0_CH_3] = MYSTRING(MMIO_IMC_DEVICE_0_CH_3),
    [MMIO_IMC_DEVICE_0_CH_4] = MYSTRING(MMIO_IMC_DEVICE_0_CH_4),
    [MMIO_IMC_DEVICE_0_CH_5] = MYSTRING(MMIO_IMC_DEVICE_0_CH_5),
    [MMIO_IMC_DEVICE_0_CH_6] = MYSTRING(MMIO_IMC_DEVICE_0_CH_6),
    [MMIO_IMC_DEVICE_0_CH_7] = MYSTRING(MMIO_IMC_DEVICE_0_CH_7),
    [MMIO_IMC_DEVICE_1_CH_0] = MYSTRING(MMIO_IMC_DEVICE_1_CH_0),
    [MMIO_IMC_DEVICE_1_CH_1] = MYSTRING(MMIO_IMC_DEVICE_1_CH_1),
    [MMIO_IMC_DEVICE_1_CH_2] = MYSTRING(MMIO_IMC_DEVICE_1_CH_2),
    [MMIO_IMC_DEVICE_1_CH_3] = MYSTRING(MMIO_IMC_DEVICE_1_CH_3),
    [MMIO_IMC_DEVICE_1_CH_4] = MYSTRING(MMIO_IMC_DEVICE_1_CH_4),
    [MMIO_IMC_DEVICE_1_CH_5] = MYSTRING(MMIO_IMC_DEVICE_1_CH_5),
    [MMIO_IMC_DEVICE_1_CH_6] = MYSTRING(MMIO_IMC_DEVICE_1_CH_6),
    [MMIO_IMC_DEVICE_1_CH_7] = MYSTRING(MMIO_IMC_DEVICE_1_CH_7),
    [MMIO_HBM_DEVICE_0] = MYSTRING(MMIO_HBM_DEVICE_0),
    [MMIO_HBM_DEVICE_1] = MYSTRING(MMIO_HBM_DEVICE_1),
    [MMIO_HBM_DEVICE_2] = MYSTRING(MMIO_HBM_DEVICE_2),
    [MMIO_HBM_DEVICE_3] = MYSTRING(MMIO_HBM_DEVICE_3),
    [MMIO_HBM_DEVICE_4] = MYSTRING(MMIO_HBM_DEVICE_4),
    [MMIO_HBM_DEVICE_5] = MYSTRING(MMIO_HBM_DEVICE_5),
    [MMIO_HBM_DEVICE_6] = MYSTRING(MMIO_HBM_DEVICE_6),
    [MMIO_HBM_DEVICE_7] = MYSTRING(MMIO_HBM_DEVICE_7),
    [MMIO_HBM_DEVICE_8] = MYSTRING(MMIO_HBM_DEVICE_8),
    [MMIO_HBM_DEVICE_9] = MYSTRING(MMIO_HBM_DEVICE_9),
    [MMIO_HBM_DEVICE_10] = MYSTRING(MMIO_HBM_DEVICE_10),
    [MMIO_HBM_DEVICE_11] = MYSTRING(MMIO_HBM_DEVICE_11),
    [MMIO_HBM_DEVICE_12] = MYSTRING(MMIO_HBM_DEVICE_12),
    [MMIO_HBM_DEVICE_13] = MYSTRING(MMIO_HBM_DEVICE_13),
    [MMIO_HBM_DEVICE_14] = MYSTRING(MMIO_HBM_DEVICE_14),
    [MMIO_HBM_DEVICE_15] = MYSTRING(MMIO_HBM_DEVICE_15),
    [MMIO_HBM_DEVICE_16] = MYSTRING(MMIO_HBM_DEVICE_16),
    [MMIO_HBM_DEVICE_17] = MYSTRING(MMIO_HBM_DEVICE_17),
    [MMIO_HBM_DEVICE_18] = MYSTRING(MMIO_HBM_DEVICE_18),
    [MMIO_HBM_DEVICE_19] = MYSTRING(MMIO_HBM_DEVICE_19),
    [MMIO_HBM_DEVICE_20] = MYSTRING(MMIO_HBM_DEVICE_20),
    [MMIO_HBM_DEVICE_21] = MYSTRING(MMIO_HBM_DEVICE_21),
    [MMIO_HBM_DEVICE_22] = MYSTRING(MMIO_HBM_DEVICE_22),
    [MMIO_HBM_DEVICE_23] = MYSTRING(MMIO_HBM_DEVICE_23),
    [MMIO_HBM_DEVICE_24] = MYSTRING(MMIO_HBM_DEVICE_24),
    [MMIO_HBM_DEVICE_25] = MYSTRING(MMIO_HBM_DEVICE_25),
    [MMIO_HBM_DEVICE_26] = MYSTRING(MMIO_HBM_DEVICE_26),
    [MMIO_HBM_DEVICE_27] = MYSTRING(MMIO_HBM_DEVICE_27),
    [MMIO_HBM_DEVICE_28] = MYSTRING(MMIO_HBM_DEVICE_28),
    [MMIO_HBM_DEVICE_29] = MYSTRING(MMIO_HBM_DEVICE_29),
    [MMIO_HBM_DEVICE_30] = MYSTRING(MMIO_HBM_DEVICE_30),
    [MMIO_HBM_DEVICE_31] = MYSTRING(MMIO_HBM_DEVICE_31),
    [MSR_CBOX_DEVICE_C0] = MYSTRING(MSR_CBOX_DEVICE_C0),
    [MSR_CBOX_DEVICE_C1] = MYSTRING(MSR_CBOX_DEVICE_C1),
    [MSR_CBOX_DEVICE_C2] = MYSTRING(MSR_CBOX_DEVICE_C2),
    [MSR_CBOX_DEVICE_C3] = MYSTRING(MSR_CBOX_DEVICE_C3),
    [MSR_CBOX_DEVICE_C4] = MYSTRING(MSR_CBOX_DEVICE_C4),
    [MSR_CBOX_DEVICE_C5] = MYSTRING(MSR_CBOX_DEVICE_C5),
    [MSR_CBOX_DEVICE_C6] = MYSTRING(MSR_CBOX_DEVICE_C6),
    [MSR_CBOX_DEVICE_C7] = MYSTRING(MSR_CBOX_DEVICE_C7),
    [MSR_CBOX_DEVICE_C8] = MYSTRING(MSR_CBOX_DEVICE_C8),
    [MSR_CBOX_DEVICE_C9] = MYSTRING(MSR_CBOX_DEVICE_C9),
    [MSR_CBOX_DEVICE_C10] = MYSTRING(MSR_CBOX_DEVICE_C10),
    [MSR_CBOX_DEVICE_C11] = MYSTRING(MSR_CBOX_DEVICE_C11),
    [MSR_CBOX_DEVICE_C12] = MYSTRING(MSR_CBOX_DEVICE_C12),
    [MSR_CBOX_DEVICE_C13] = MYSTRING(MSR_CBOX_DEVICE_C13),
    [MSR_CBOX_DEVICE_C14] = MYSTRING(MSR_CBOX_DEVICE_C14),
    [MSR_CBOX_DEVICE_C15] = MYSTRING(MSR_CBOX_DEVICE_C15),
    [MSR_CBOX_DEVICE_C16] = MYSTRING(MSR_CBOX_DEVICE_C16),
    [MSR_CBOX_DEVICE_C17] = MYSTRING(MSR_CBOX_DEVICE_C17),
    [MSR_CBOX_DEVICE_C18] = MYSTRING(MSR_CBOX_DEVICE_C18),
    [MSR_CBOX_DEVICE_C19] = MYSTRING(MSR_CBOX_DEVICE_C19),
    [MSR_CBOX_DEVICE_C20] = MYSTRING(MSR_CBOX_DEVICE_C20),
    [MSR_CBOX_DEVICE_C21] = MYSTRING(MSR_CBOX_DEVICE_C21),
    [MSR_CBOX_DEVICE_C22] = MYSTRING(MSR_CBOX_DEVICE_C22),
    [MSR_CBOX_DEVICE_C23] = MYSTRING(MSR_CBOX_DEVICE_C23),
    [MSR_CBOX_DEVICE_C24] = MYSTRING(MSR_CBOX_DEVICE_C24),
    [MSR_CBOX_DEVICE_C25] = MYSTRING(MSR_CBOX_DEVICE_C25),
    [MSR_CBOX_DEVICE_C26] = MYSTRING(MSR_CBOX_DEVICE_C26),
    [MSR_CBOX_DEVICE_C27] = MYSTRING(MSR_CBOX_DEVICE_C27),
    [MSR_CBOX_DEVICE_C28] = MYSTRING(MSR_CBOX_DEVICE_C28),
    [MSR_CBOX_DEVICE_C29] = MYSTRING(MSR_CBOX_DEVICE_C29),
    [MSR_CBOX_DEVICE_C30] = MYSTRING(MSR_CBOX_DEVICE_C30),
    [MSR_CBOX_DEVICE_C31] = MYSTRING(MSR_CBOX_DEVICE_C31),
    [MSR_CBOX_DEVICE_C32] = MYSTRING(MSR_CBOX_DEVICE_C32),
    [MSR_CBOX_DEVICE_C33] = MYSTRING(MSR_CBOX_DEVICE_C33),
    [MSR_CBOX_DEVICE_C34] = MYSTRING(MSR_CBOX_DEVICE_C34),
    [MSR_CBOX_DEVICE_C35] = MYSTRING(MSR_CBOX_DEVICE_C35),
    [MSR_CBOX_DEVICE_C36] = MYSTRING(MSR_CBOX_DEVICE_C36),
    [MSR_CBOX_DEVICE_C37] = MYSTRING(MSR_CBOX_DEVICE_C37),
    [MSR_CBOX_DEVICE_C38] = MYSTRING(MSR_CBOX_DEVICE_C38),
    [MSR_CBOX_DEVICE_C39] = MYSTRING(MSR_CBOX_DEVICE_C39),
    [MSR_CBOX_DEVICE_C40] = MYSTRING(MSR_CBOX_DEVICE_C40),
    [MSR_CBOX_DEVICE_C41] = MYSTRING(MSR_CBOX_DEVICE_C41),
    [MSR_CBOX_DEVICE_C42] = MYSTRING(MSR_CBOX_DEVICE_C42),
    [MSR_CBOX_DEVICE_C43] = MYSTRING(MSR_CBOX_DEVICE_C43),
    [MSR_CBOX_DEVICE_C44] = MYSTRING(MSR_CBOX_DEVICE_C44),
    [MSR_CBOX_DEVICE_C45] = MYSTRING(MSR_CBOX_DEVICE_C45),
    [MSR_CBOX_DEVICE_C46] = MYSTRING(MSR_CBOX_DEVICE_C46),
    [MSR_CBOX_DEVICE_C47] = MYSTRING(MSR_CBOX_DEVICE_C47),
    [MSR_CBOX_DEVICE_C48] = MYSTRING(MSR_CBOX_DEVICE_C48),
    [MSR_CBOX_DEVICE_C49] = MYSTRING(MSR_CBOX_DEVICE_C49),
    [MSR_CBOX_DEVICE_C50] = MYSTRING(MSR_CBOX_DEVICE_C50),
    [MSR_CBOX_DEVICE_C51] = MYSTRING(MSR_CBOX_DEVICE_C51),
    [MSR_CBOX_DEVICE_C52] = MYSTRING(MSR_CBOX_DEVICE_C52),
    [MSR_CBOX_DEVICE_C53] = MYSTRING(MSR_CBOX_DEVICE_C53),
    [MSR_CBOX_DEVICE_C54] = MYSTRING(MSR_CBOX_DEVICE_C54),
    [MSR_CBOX_DEVICE_C55] = MYSTRING(MSR_CBOX_DEVICE_C55),
    [MSR_CBOX_DEVICE_C56] = MYSTRING(MSR_CBOX_DEVICE_C56),
    [MSR_CBOX_DEVICE_C57] = MYSTRING(MSR_CBOX_DEVICE_C57),
    [MSR_CBOX_DEVICE_C58] = MYSTRING(MSR_CBOX_DEVICE_C58),
    [MSR_CBOX_DEVICE_C59] = MYSTRING(MSR_CBOX_DEVICE_C59),
    [MSR_CBOX_DEVICE_C60] = MYSTRING(MSR_CBOX_DEVICE_C60),
    [MSR_CBOX_DEVICE_C61] = MYSTRING(MSR_CBOX_DEVICE_C61),
    [MSR_CBOX_DEVICE_C62] = MYSTRING(MSR_CBOX_DEVICE_C62),
    [MSR_CBOX_DEVICE_C63] = MYSTRING(MSR_CBOX_DEVICE_C63),
    [MSR_CBOX_DEVICE_C64] = MYSTRING(MSR_CBOX_DEVICE_C64),
    [MSR_CBOX_DEVICE_C65] = MYSTRING(MSR_CBOX_DEVICE_C65),
    [MSR_CBOX_DEVICE_C66] = MYSTRING(MSR_CBOX_DEVICE_C66),
    [MSR_CBOX_DEVICE_C67] = MYSTRING(MSR_CBOX_DEVICE_C67),
    [MSR_CBOX_DEVICE_C68] = MYSTRING(MSR_CBOX_DEVICE_C68),
    [MSR_CBOX_DEVICE_C69] = MYSTRING(MSR_CBOX_DEVICE_C69),
    [MSR_CBOX_DEVICE_C70] = MYSTRING(MSR_CBOX_DEVICE_C70),
    [MSR_CBOX_DEVICE_C71] = MYSTRING(MSR_CBOX_DEVICE_C71),
    [MSR_CBOX_DEVICE_C72] = MYSTRING(MSR_CBOX_DEVICE_C72),
    [MSR_CBOX_DEVICE_C73] = MYSTRING(MSR_CBOX_DEVICE_C73),
    [MSR_CBOX_DEVICE_C74] = MYSTRING(MSR_CBOX_DEVICE_C74),
    [MSR_CBOX_DEVICE_C75] = MYSTRING(MSR_CBOX_DEVICE_C75),
    [MSR_CBOX_DEVICE_C76] = MYSTRING(MSR_CBOX_DEVICE_C76),
    [MSR_CBOX_DEVICE_C77] = MYSTRING(MSR_CBOX_DEVICE_C77),
    [MSR_CBOX_DEVICE_C78] = MYSTRING(MSR_CBOX_DEVICE_C78),
    [MSR_CBOX_DEVICE_C79] = MYSTRING(MSR_CBOX_DEVICE_C79),
    [MSR_CBOX_DEVICE_C80] = MYSTRING(MSR_CBOX_DEVICE_C80),
    [MSR_CBOX_DEVICE_C81] = MYSTRING(MSR_CBOX_DEVICE_C81),
    [MSR_CBOX_DEVICE_C82] = MYSTRING(MSR_CBOX_DEVICE_C82),
    [MSR_CBOX_DEVICE_C83] = MYSTRING(MSR_CBOX_DEVICE_C83),
    [MSR_CBOX_DEVICE_C84] = MYSTRING(MSR_CBOX_DEVICE_C84),
    [MSR_CBOX_DEVICE_C85] = MYSTRING(MSR_CBOX_DEVICE_C85),
    [MSR_CBOX_DEVICE_C86] = MYSTRING(MSR_CBOX_DEVICE_C86),
    [MSR_CBOX_DEVICE_C87] = MYSTRING(MSR_CBOX_DEVICE_C87),
    [MSR_CBOX_DEVICE_C88] = MYSTRING(MSR_CBOX_DEVICE_C88),
    [MSR_CBOX_DEVICE_C89] = MYSTRING(MSR_CBOX_DEVICE_C89),
    [MSR_CBOX_DEVICE_C90] = MYSTRING(MSR_CBOX_DEVICE_C90),
    [MSR_CBOX_DEVICE_C91] = MYSTRING(MSR_CBOX_DEVICE_C91),
    [MSR_CBOX_DEVICE_C92] = MYSTRING(MSR_CBOX_DEVICE_C92),
    [MSR_CBOX_DEVICE_C93] = MYSTRING(MSR_CBOX_DEVICE_C93),
    [MSR_CBOX_DEVICE_C94] = MYSTRING(MSR_CBOX_DEVICE_C94),
    [MSR_CBOX_DEVICE_C95] = MYSTRING(MSR_CBOX_DEVICE_C95),
    [MSR_CBOX_DEVICE_C96] = MYSTRING(MSR_CBOX_DEVICE_C96),
    [MSR_CBOX_DEVICE_C97] = MYSTRING(MSR_CBOX_DEVICE_C97),
    [MSR_CBOX_DEVICE_C98] = MYSTRING(MSR_CBOX_DEVICE_C98),
    [MSR_CBOX_DEVICE_C99] = MYSTRING(MSR_CBOX_DEVICE_C99),
    [MSR_CBOX_DEVICE_C100] = MYSTRING(MSR_CBOX_DEVICE_C100),
    [MSR_CBOX_DEVICE_C101] = MYSTRING(MSR_CBOX_DEVICE_C101),
    [MSR_CBOX_DEVICE_C102] = MYSTRING(MSR_CBOX_DEVICE_C102),
    [MSR_CBOX_DEVICE_C103] = MYSTRING(MSR_CBOX_DEVICE_C103),
    [MSR_CBOX_DEVICE_C104] = MYSTRING(MSR_CBOX_DEVICE_C104),
    [MSR_CBOX_DEVICE_C105] = MYSTRING(MSR_CBOX_DEVICE_C105),
    [MSR_CBOX_DEVICE_C106] = MYSTRING(MSR_CBOX_DEVICE_C106),
    [MSR_CBOX_DEVICE_C107] = MYSTRING(MSR_CBOX_DEVICE_C107),
    [MSR_CBOX_DEVICE_C108] = MYSTRING(MSR_CBOX_DEVICE_C108),
    [MSR_CBOX_DEVICE_C109] = MYSTRING(MSR_CBOX_DEVICE_C109),
    [MSR_CBOX_DEVICE_C110] = MYSTRING(MSR_CBOX_DEVICE_C110),
    [MSR_CBOX_DEVICE_C111] = MYSTRING(MSR_CBOX_DEVICE_C111),
    [MSR_CBOX_DEVICE_C112] = MYSTRING(MSR_CBOX_DEVICE_C112),
    [MSR_CBOX_DEVICE_C113] = MYSTRING(MSR_CBOX_DEVICE_C113),
    [MSR_CBOX_DEVICE_C114] = MYSTRING(MSR_CBOX_DEVICE_C114),
    [MSR_CBOX_DEVICE_C115] = MYSTRING(MSR_CBOX_DEVICE_C115),
    [MSR_CBOX_DEVICE_C116] = MYSTRING(MSR_CBOX_DEVICE_C116),
    [MSR_CBOX_DEVICE_C117] = MYSTRING(MSR_CBOX_DEVICE_C117),
    [MSR_CBOX_DEVICE_C118] = MYSTRING(MSR_CBOX_DEVICE_C118),
    [MSR_CBOX_DEVICE_C119] = MYSTRING(MSR_CBOX_DEVICE_C119),
    [MSR_CBOX_DEVICE_C120] = MYSTRING(MSR_CBOX_DEVICE_C120),
    [MSR_CBOX_DEVICE_C121] = MYSTRING(MSR_CBOX_DEVICE_C121),
    [MSR_CBOX_DEVICE_C122] = MYSTRING(MSR_CBOX_DEVICE_C122),
    [MSR_CBOX_DEVICE_C123] = MYSTRING(MSR_CBOX_DEVICE_C123),
    [MSR_CBOX_DEVICE_C124] = MYSTRING(MSR_CBOX_DEVICE_C124),
    [MSR_CBOX_DEVICE_C125] = MYSTRING(MSR_CBOX_DEVICE_C125),
    [MSR_CBOX_DEVICE_C126] = MYSTRING(MSR_CBOX_DEVICE_C126),
    [MSR_CBOX_DEVICE_C127] = MYSTRING(MSR_CBOX_DEVICE_C127),
    [MSR_CBOX_DEVICE_C128] = MYSTRING(MSR_CBOX_DEVICE_C128),
    [MSR_CBOX_DEVICE_C129] = MYSTRING(MSR_CBOX_DEVICE_C129),
    [MSR_UBOX_DEVICE] = MYSTRING(MSR_UBOX_DEVICE),
    [MSR_MDF_DEVICE_0] = MYSTRING(MSR_MDF_DEVICE_0),
    [MSR_MDF_DEVICE_1] = MYSTRING(MSR_MDF_DEVICE_1),
    [MSR_MDF_DEVICE_2] = MYSTRING(MSR_MDF_DEVICE_2),
    [MSR_MDF_DEVICE_3] = MYSTRING(MSR_MDF_DEVICE_3),
    [MSR_MDF_DEVICE_4] = MYSTRING(MSR_MDF_DEVICE_4),
    [MSR_MDF_DEVICE_5] = MYSTRING(MSR_MDF_DEVICE_5),
    [MSR_MDF_DEVICE_6] = MYSTRING(MSR_MDF_DEVICE_6),
    [MSR_MDF_DEVICE_7] = MYSTRING(MSR_MDF_DEVICE_7),
    [MSR_MDF_DEVICE_8] = MYSTRING(MSR_MDF_DEVICE_8),
    [MSR_MDF_DEVICE_9] = MYSTRING(MSR_MDF_DEVICE_9),
    [MSR_MDF_DEVICE_10] = MYSTRING(MSR_MDF_DEVICE_10),
    [MSR_MDF_DEVICE_11] = MYSTRING(MSR_MDF_DEVICE_11),
    [MSR_MDF_DEVICE_12] = MYSTRING(MSR_MDF_DEVICE_12),
    [MSR_MDF_DEVICE_13] = MYSTRING(MSR_MDF_DEVICE_13),
    [MSR_MDF_DEVICE_14] = MYSTRING(MSR_MDF_DEVICE_14),
    [MSR_MDF_DEVICE_15] = MYSTRING(MSR_MDF_DEVICE_15),
    [MSR_MDF_DEVICE_16] = MYSTRING(MSR_MDF_DEVICE_16),
    [MSR_MDF_DEVICE_17] = MYSTRING(MSR_MDF_DEVICE_17),
    [MSR_MDF_DEVICE_18] = MYSTRING(MSR_MDF_DEVICE_18),
    [MSR_MDF_DEVICE_19] = MYSTRING(MSR_MDF_DEVICE_19),
    [MSR_MDF_DEVICE_20] = MYSTRING(MSR_MDF_DEVICE_20),
    [MSR_MDF_DEVICE_21] = MYSTRING(MSR_MDF_DEVICE_21),
    [MSR_MDF_DEVICE_22] = MYSTRING(MSR_MDF_DEVICE_22),
    [MSR_MDF_DEVICE_23] = MYSTRING(MSR_MDF_DEVICE_23),
    [MSR_MDF_DEVICE_24] = MYSTRING(MSR_MDF_DEVICE_24),
    [MSR_MDF_DEVICE_25] = MYSTRING(MSR_MDF_DEVICE_25),
    [MSR_MDF_DEVICE_26] = MYSTRING(MSR_MDF_DEVICE_26),
    [MSR_MDF_DEVICE_27] = MYSTRING(MSR_MDF_DEVICE_27),
    [MSR_MDF_DEVICE_28] = MYSTRING(MSR_MDF_DEVICE_28),
    [MSR_MDF_DEVICE_29] = MYSTRING(MSR_MDF_DEVICE_29),
    [MSR_MDF_DEVICE_30] = MYSTRING(MSR_MDF_DEVICE_30),
    [MSR_MDF_DEVICE_31] = MYSTRING(MSR_MDF_DEVICE_31),
    [MSR_MDF_DEVICE_32] = MYSTRING(MSR_MDF_DEVICE_32),
    [MSR_MDF_DEVICE_33] = MYSTRING(MSR_MDF_DEVICE_33),
    [MSR_MDF_DEVICE_34] = MYSTRING(MSR_MDF_DEVICE_34),
    [MSR_MDF_DEVICE_35] = MYSTRING(MSR_MDF_DEVICE_35),
    [MSR_MDF_DEVICE_36] = MYSTRING(MSR_MDF_DEVICE_36),
    [MSR_MDF_DEVICE_37] = MYSTRING(MSR_MDF_DEVICE_37),
    [MSR_MDF_DEVICE_38] = MYSTRING(MSR_MDF_DEVICE_38),
    [MSR_MDF_DEVICE_39] = MYSTRING(MSR_MDF_DEVICE_39),
    [MSR_MDF_DEVICE_40] = MYSTRING(MSR_MDF_DEVICE_40),
    [MSR_MDF_DEVICE_41] = MYSTRING(MSR_MDF_DEVICE_41),
    [MSR_MDF_DEVICE_42] = MYSTRING(MSR_MDF_DEVICE_42),
    [MSR_MDF_DEVICE_43] = MYSTRING(MSR_MDF_DEVICE_43),
    [MSR_MDF_DEVICE_44] = MYSTRING(MSR_MDF_DEVICE_44),
    [MSR_MDF_DEVICE_45] = MYSTRING(MSR_MDF_DEVICE_45),
    [MSR_MDF_DEVICE_46] = MYSTRING(MSR_MDF_DEVICE_46),
    [MSR_MDF_DEVICE_47] = MYSTRING(MSR_MDF_DEVICE_47),
    [MSR_MDF_DEVICE_48] = MYSTRING(MSR_MDF_DEVICE_48),
    [MSR_MDF_DEVICE_49] = MYSTRING(MSR_MDF_DEVICE_49),
    [MSR_MDF_DEVICE_50] = MYSTRING(MSR_MDF_DEVICE_50),
    [MSR_MDF_DEVICE_51] = MYSTRING(MSR_MDF_DEVICE_51),
    [MSR_MDF_DEVICE_52] = MYSTRING(MSR_MDF_DEVICE_52),
    [MSR_MDF_DEVICE_53] = MYSTRING(MSR_MDF_DEVICE_53),
    [MSR_MDF_DEVICE_54] = MYSTRING(MSR_MDF_DEVICE_54),
    [MSR_MDF_DEVICE_55] = MYSTRING(MSR_MDF_DEVICE_55),
    [MSR_MDF_DEVICE_56] = MYSTRING(MSR_MDF_DEVICE_56),
    [MSR_MDF_DEVICE_57] = MYSTRING(MSR_MDF_DEVICE_57),
    [MSR_MDF_DEVICE_58] = MYSTRING(MSR_MDF_DEVICE_58),
    [MSR_MDF_DEVICE_59] = MYSTRING(MSR_MDF_DEVICE_59),
    [MSR_MDF_DEVICE_60] = MYSTRING(MSR_MDF_DEVICE_60),
    [MSR_MDF_DEVICE_61] = MYSTRING(MSR_MDF_DEVICE_61),
    [MSR_MDF_DEVICE_62] = MYSTRING(MSR_MDF_DEVICE_62),
    [MSR_MDF_DEVICE_63] = MYSTRING(MSR_MDF_DEVICE_63),
    [MSR_MDF_DEVICE_64] = MYSTRING(MSR_MDF_DEVICE_64),
    [MSR_MDF_DEVICE_65] = MYSTRING(MSR_MDF_DEVICE_65),
    [MSR_MDF_DEVICE_66] = MYSTRING(MSR_MDF_DEVICE_66),
    [MSR_MDF_DEVICE_67] = MYSTRING(MSR_MDF_DEVICE_67),
    [MSR_MDF_DEVICE_68] = MYSTRING(MSR_MDF_DEVICE_68),
    [MSR_MDF_DEVICE_69] = MYSTRING(MSR_MDF_DEVICE_69),
    [MSR_MDF_DEVICE_70] = MYSTRING(MSR_MDF_DEVICE_70),
    [MSR_MDF_DEVICE_71] = MYSTRING(MSR_MDF_DEVICE_71),
    [MSR_MDF_DEVICE_72] = MYSTRING(MSR_MDF_DEVICE_72),
    [MSR_MDF_DEVICE_73] = MYSTRING(MSR_MDF_DEVICE_73),
    [MSR_MDF_DEVICE_74] = MYSTRING(MSR_MDF_DEVICE_74),
    [MSR_MDF_DEVICE_75] = MYSTRING(MSR_MDF_DEVICE_75),
    [MSR_MDF_DEVICE_76] = MYSTRING(MSR_MDF_DEVICE_76),
    [MSR_MDF_DEVICE_77] = MYSTRING(MSR_MDF_DEVICE_77),
    [MSR_MDF_DEVICE_78] = MYSTRING(MSR_MDF_DEVICE_78),
    [MSR_MDF_DEVICE_79] = MYSTRING(MSR_MDF_DEVICE_79),
    [MSR_MDF_DEVICE_80] = MYSTRING(MSR_MDF_DEVICE_80),
    [MSR_MDF_DEVICE_81] = MYSTRING(MSR_MDF_DEVICE_81),
    [MSR_MDF_DEVICE_82] = MYSTRING(MSR_MDF_DEVICE_82),
    [MSR_MDF_DEVICE_83] = MYSTRING(MSR_MDF_DEVICE_83),
    [MSR_MDF_DEVICE_84] = MYSTRING(MSR_MDF_DEVICE_84),
    [MSR_MDF_DEVICE_85] = MYSTRING(MSR_MDF_DEVICE_85),
    [MSR_MDF_DEVICE_86] = MYSTRING(MSR_MDF_DEVICE_86),
    [MSR_MDF_DEVICE_87] = MYSTRING(MSR_MDF_DEVICE_87),
    [MSR_MDF_DEVICE_88] = MYSTRING(MSR_MDF_DEVICE_88),
    [MSR_MDF_DEVICE_89] = MYSTRING(MSR_MDF_DEVICE_89),
    [PCI_HA_DEVICE_0] = MYSTRING(PCI_HA_DEVICE_0),
    [PCI_HA_DEVICE_1] = MYSTRING(PCI_HA_DEVICE_1),
    [PCI_HA_DEVICE_2] = MYSTRING(PCI_HA_DEVICE_2),
    [PCI_HA_DEVICE_3] = MYSTRING(PCI_HA_DEVICE_3),
    [PCI_HA_DEVICE_4] = MYSTRING(PCI_HA_DEVICE_4),
    [PCI_HA_DEVICE_5] = MYSTRING(PCI_HA_DEVICE_5),
    [PCI_HA_DEVICE_6] = MYSTRING(PCI_HA_DEVICE_6),
    [PCI_HA_DEVICE_7] = MYSTRING(PCI_HA_DEVICE_7),
    [PCI_HA_DEVICE_8] = MYSTRING(PCI_HA_DEVICE_8),
    [PCI_HA_DEVICE_9] = MYSTRING(PCI_HA_DEVICE_9),
    [PCI_HA_DEVICE_10] = MYSTRING(PCI_HA_DEVICE_10),
    [PCI_HA_DEVICE_11] = MYSTRING(PCI_HA_DEVICE_11),
    [PCI_HA_DEVICE_12] = MYSTRING(PCI_HA_DEVICE_12),
    [PCI_HA_DEVICE_13] = MYSTRING(PCI_HA_DEVICE_13),
    [PCI_HA_DEVICE_14] = MYSTRING(PCI_HA_DEVICE_14),
    [PCI_HA_DEVICE_15] = MYSTRING(PCI_HA_DEVICE_15),
    [PCI_HA_DEVICE_16] = MYSTRING(PCI_HA_DEVICE_16),
    [PCI_HA_DEVICE_17] = MYSTRING(PCI_HA_DEVICE_17),
    [PCI_HA_DEVICE_18] = MYSTRING(PCI_HA_DEVICE_18),
    [PCI_HA_DEVICE_19] = MYSTRING(PCI_HA_DEVICE_19),
    [PCI_HA_DEVICE_20] = MYSTRING(PCI_HA_DEVICE_20),
    [PCI_HA_DEVICE_21] = MYSTRING(PCI_HA_DEVICE_21),
    [PCI_HA_DEVICE_22] = MYSTRING(PCI_HA_DEVICE_22),
    [PCI_HA_DEVICE_23] = MYSTRING(PCI_HA_DEVICE_23),
    [PCI_HA_DEVICE_24] = MYSTRING(PCI_HA_DEVICE_24),
    [PCI_HA_DEVICE_25] = MYSTRING(PCI_HA_DEVICE_25),
    [PCI_HA_DEVICE_26] = MYSTRING(PCI_HA_DEVICE_26),
    [PCI_HA_DEVICE_27] = MYSTRING(PCI_HA_DEVICE_27),
    [PCI_HA_DEVICE_28] = MYSTRING(PCI_HA_DEVICE_28),
    [PCI_HA_DEVICE_29] = MYSTRING(PCI_HA_DEVICE_29),
    [PCI_HA_DEVICE_30] = MYSTRING(PCI_HA_DEVICE_30),
    [PCI_HA_DEVICE_31] = MYSTRING(PCI_HA_DEVICE_31),
    [PCI_QPI_DEVICE_PORT_0] = MYSTRING(PCI_QPI_DEVICE_PORT_0),
    [PCI_QPI_DEVICE_PORT_1] = MYSTRING(PCI_QPI_DEVICE_PORT_1),
    [PCI_QPI_DEVICE_PORT_2] = MYSTRING(PCI_QPI_DEVICE_PORT_2),
    [PCI_QPI_DEVICE_PORT_3] = MYSTRING(PCI_QPI_DEVICE_PORT_3),
    [PCI_QPI_DEVICE_PORT_4] = MYSTRING(PCI_QPI_DEVICE_PORT_4),
    [PCI_QPI_DEVICE_PORT_5] = MYSTRING(PCI_QPI_DEVICE_PORT_5),
    [PCI_R3QPI_DEVICE_LINK_0] = MYSTRING(PCI_R3QPI_DEVICE_LINK_0),
    [PCI_R3QPI_DEVICE_LINK_1] = MYSTRING(PCI_R3QPI_DEVICE_LINK_1),
    [PCI_R3QPI_DEVICE_LINK_2] = MYSTRING(PCI_R3QPI_DEVICE_LINK_2),
    [PCI_R3QPI_DEVICE_LINK_3] = MYSTRING(PCI_R3QPI_DEVICE_LINK_3),
    [PCI_R3QPI_DEVICE_LINK_4] = MYSTRING(PCI_R3QPI_DEVICE_LINK_4),
    [PCI_R3QPI_DEVICE_LINK_5] = MYSTRING(PCI_R3QPI_DEVICE_LINK_5),
    [MSR_PCU_DEVICE_0] = MYSTRING(MSR_PCU_DEVICE_0),
    [MSR_PCU_DEVICE_1] = MYSTRING(MSR_PCU_DEVICE_1),
    [MSR_PCU_DEVICE_2] = MYSTRING(MSR_PCU_DEVICE_2),
    [MSR_PCU_DEVICE_3] = MYSTRING(MSR_PCU_DEVICE_3),
    [MSR_PCU_DEVICE_4] = MYSTRING(MSR_PCU_DEVICE_4),
    [MSR_PCU_DEVICE_5] = MYSTRING(MSR_PCU_DEVICE_5),
    [MSR_IRP_DEVICE_0] = MYSTRING(MSR_IRP_DEVICE_0),
    [MSR_IRP_DEVICE_1] = MYSTRING(MSR_IRP_DEVICE_1),
    [MSR_IRP_DEVICE_2] = MYSTRING(MSR_IRP_DEVICE_2),
    [MSR_IRP_DEVICE_3] = MYSTRING(MSR_IRP_DEVICE_3),
    [MSR_IRP_DEVICE_4] = MYSTRING(MSR_IRP_DEVICE_4),
    [MSR_IRP_DEVICE_5] = MYSTRING(MSR_IRP_DEVICE_5),
    [MSR_IRP_DEVICE_6] = MYSTRING(MSR_IRP_DEVICE_6),
    [MSR_IRP_DEVICE_7] = MYSTRING(MSR_IRP_DEVICE_7),
    [MSR_IRP_DEVICE_8] = MYSTRING(MSR_IRP_DEVICE_8),
    [MSR_IRP_DEVICE_9] = MYSTRING(MSR_IRP_DEVICE_9),
    [MSR_IRP_DEVICE_10] = MYSTRING(MSR_IRP_DEVICE_10),
    [MSR_IRP_DEVICE_11] = MYSTRING(MSR_IRP_DEVICE_11),
    [MSR_IRP_DEVICE_12] = MYSTRING(MSR_IRP_DEVICE_12),
    [MSR_IRP_DEVICE_13] = MYSTRING(MSR_IRP_DEVICE_13),
    [MSR_IRP_DEVICE_14] = MYSTRING(MSR_IRP_DEVICE_14),
    [MSR_IRP_DEVICE_15] = MYSTRING(MSR_IRP_DEVICE_15),
    [MSR_IIO_DEVICE_0] = MYSTRING(MSR_IIO_DEVICE_0),
    [MSR_IIO_DEVICE_1] = MYSTRING(MSR_IIO_DEVICE_1),
    [MSR_IIO_DEVICE_2] = MYSTRING(MSR_IIO_DEVICE_2),
    [MSR_IIO_DEVICE_3] = MYSTRING(MSR_IIO_DEVICE_3),
    [MSR_IIO_DEVICE_4] = MYSTRING(MSR_IIO_DEVICE_4),
    [MSR_IIO_DEVICE_5] = MYSTRING(MSR_IIO_DEVICE_5),
    [MSR_IIO_DEVICE_6] = MYSTRING(MSR_IIO_DEVICE_6),
    [MSR_IIO_DEVICE_7] = MYSTRING(MSR_IIO_DEVICE_7),
    [MSR_IIO_DEVICE_8] = MYSTRING(MSR_IIO_DEVICE_8),
    [MSR_IIO_DEVICE_9] = MYSTRING(MSR_IIO_DEVICE_9),
    [MSR_IIO_DEVICE_10] = MYSTRING(MSR_IIO_DEVICE_10),
    [MSR_IIO_DEVICE_11] = MYSTRING(MSR_IIO_DEVICE_11),
    [MSR_IIO_DEVICE_12] = MYSTRING(MSR_IIO_DEVICE_12),
    [MSR_IIO_DEVICE_13] = MYSTRING(MSR_IIO_DEVICE_13),
    [MSR_IIO_DEVICE_14] = MYSTRING(MSR_IIO_DEVICE_14),
    [MSR_IIO_DEVICE_15] = MYSTRING(MSR_IIO_DEVICE_15),
    [PCI_R2PCIE_DEVICE0] = MYSTRING(PCI_R2PCIE_DEVICE0),
    [PCI_R2PCIE_DEVICE1] = MYSTRING(PCI_R2PCIE_DEVICE1),
    [PCI_R2PCIE_DEVICE2] = MYSTRING(PCI_R2PCIE_DEVICE2),
    [PCI_R2PCIE_DEVICE3] = MYSTRING(PCI_R2PCIE_DEVICE3),
    [PCI_R2PCIE_DEVICE4] = MYSTRING(PCI_R2PCIE_DEVICE4),
    [PCI_R2PCIE_DEVICE5] = MYSTRING(PCI_R2PCIE_DEVICE5),
    [PCI_R2PCIE_DEVICE6] = MYSTRING(PCI_R2PCIE_DEVICE6),
    [PCI_R2PCIE_DEVICE7] = MYSTRING(PCI_R2PCIE_DEVICE7),
    [PCI_R2PCIE_DEVICE8] = MYSTRING(PCI_R2PCIE_DEVICE8),
    [PCI_R2PCIE_DEVICE9] = MYSTRING(PCI_R2PCIE_DEVICE9),
    [PCI_R2PCIE_DEVICE10] = MYSTRING(PCI_R2PCIE_DEVICE10),
    [PCI_R2PCIE_DEVICE11] = MYSTRING(PCI_R2PCIE_DEVICE11),
    [PCI_R2PCIE_DEVICE12] = MYSTRING(PCI_R2PCIE_DEVICE12),
    [PCI_R2PCIE_DEVICE13] = MYSTRING(PCI_R2PCIE_DEVICE13),
    [PCI_R2PCIE_DEVICE14] = MYSTRING(PCI_R2PCIE_DEVICE14),
    [PCI_R2PCIE_DEVICE15] = MYSTRING(PCI_R2PCIE_DEVICE15),
    [PCI_R2PCIE_DEVICE16] = MYSTRING(PCI_R2PCIE_DEVICE16),
    [PCI_R2PCIE_DEVICE17] = MYSTRING(PCI_R2PCIE_DEVICE17),
    [PCI_R2PCIE_DEVICE18] = MYSTRING(PCI_R2PCIE_DEVICE18),
    [PCI_R2PCIE_DEVICE19] = MYSTRING(PCI_R2PCIE_DEVICE19),
    [PCI_R2PCIE_DEVICE20] = MYSTRING(PCI_R2PCIE_DEVICE20),
    [PCI_R2PCIE_DEVICE21] = MYSTRING(PCI_R2PCIE_DEVICE21),
    [PCI_R2PCIE_DEVICE22] = MYSTRING(PCI_R2PCIE_DEVICE22),
    [PCI_R2PCIE_DEVICE23] = MYSTRING(PCI_R2PCIE_DEVICE23),
    [PCI_R2PCIE_DEVICE24] = MYSTRING(PCI_R2PCIE_DEVICE24),
    [PCI_R2PCIE_DEVICE25] = MYSTRING(PCI_R2PCIE_DEVICE25),
    [PCI_R2PCIE_DEVICE26] = MYSTRING(PCI_R2PCIE_DEVICE26),
    [PCI_R2PCIE_DEVICE27] = MYSTRING(PCI_R2PCIE_DEVICE27),
    [PCI_R2PCIE_DEVICE28] = MYSTRING(PCI_R2PCIE_DEVICE28),
    [PCI_R2PCIE_DEVICE29] = MYSTRING(PCI_R2PCIE_DEVICE29),
    [PCI_R2PCIE_DEVICE30] = MYSTRING(PCI_R2PCIE_DEVICE30),
    [PCI_R2PCIE_DEVICE31] = MYSTRING(PCI_R2PCIE_DEVICE31),
    [PCI_R2PCIE_DEVICE32] = MYSTRING(PCI_R2PCIE_DEVICE32),
    [PCI_R2PCIE_DEVICE33] = MYSTRING(PCI_R2PCIE_DEVICE33),
    [PCI_R2PCIE_DEVICE34] = MYSTRING(PCI_R2PCIE_DEVICE34),
    [PCI_R2PCIE_DEVICE35] = MYSTRING(PCI_R2PCIE_DEVICE35),
    [PCI_R2PCIE_DEVICE36] = MYSTRING(PCI_R2PCIE_DEVICE36),
    [PCI_R2PCIE_DEVICE37] = MYSTRING(PCI_R2PCIE_DEVICE37),
    [PCI_R2PCIE_DEVICE38] = MYSTRING(PCI_R2PCIE_DEVICE38),
    [PCI_R2PCIE_DEVICE39] = MYSTRING(PCI_R2PCIE_DEVICE39),
};
