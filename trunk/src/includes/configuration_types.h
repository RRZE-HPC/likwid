#ifndef CONFIGURATION_TYPES_H
#define CONFIGURATION_TYPES_H

typedef struct {
    char* topologyCfgFileName;
    char* daemonPath;
    AccessMode daemonMode;
    int maxNumThreads;
    int maxNumNodes;
    int maxHashTableSize;
} Configuration;

typedef Configuration* Configuration_t;


#endif
