#ifndef HWFEATURES_X86_AMD_RAPL_H
#define HWFEATURES_X86_AMD_RAPL_H


int amd_rapl_pkg_test(void);
int amd_rapl_core_test(void);
int amd_rapl_l3_test(void);
int sysFeatures_init_amd_rapl(_SysFeatureList* out);


#endif /* HWFEATURES_X86_AMD_RAPL_H */
