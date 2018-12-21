#ifndef TOPOLOGY_CAVTX2_H
#define TOPOLOGY_CAVTX2_H

CacheLevel caviumTX2_caches[3] = {
    {1, DATACACHE, 32, 4, 64, 32768, 2, 1},
    {2, DATACACHE, 1, 8, 64, 262144, 2, 1},
    {3, DATACACHE, 0, 8, 64, 29360128, 112, 1},
};




#endif
