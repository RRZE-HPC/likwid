#ifndef HWFEATURES_X86_INTEL_RAPL_H
#define HWFEATURES_X86_INTEL_RAPL_H

#include <sysFeatures_types.h>

int intel_rapl_pkg_test(void);
int intel_rapl_dram_test(void);
int intel_rapl_pp0_test(void);
int intel_rapl_pp1_test(void);
int intel_rapl_psys_test(void);
int sysFeatures_init_intel_rapl(_SysFeatureList* out);

#endif /* HWFEATURES_X86_INTEL_UNCOREFREQ_H */
