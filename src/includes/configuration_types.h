#ifndef CONFIGURATION_TYPES_H
#define CONFIGURATION_TYPES_H


/** \addtogroup Config Config file module
*  @{
*/
/*! \brief Structure holding values of the configuration file

LIKWID supports the definition of runtime values in a configuration file. The 
most important configurations in most cases are the path the access daemon and 
the corresponding access mode. In order to avoid reading in the system topology
at each start, a path to a topology file can be set. The other values are mostly
used internally.
*/
typedef struct {
    char* topologyCfgFileName; /*!< \brief Path to the topology file */
    char* daemonPath; /*!< \brief Path of the access daemon */
    AccessMode daemonMode; /*!< \brief Access mode to the MSR and PCI registers */
    int maxNumThreads; /*!< \brief Maximum number of HW threads */
    int maxNumNodes; /*!< \brief Maximum number of NUMA nodes */
    int maxHashTableSize; /*!< \brief Maximal hash table size used for the marker API */
} Configuration;

/** \brief Pointer for exporting the Configuration data structure */
typedef Configuration* Configuration_t;
/** @}*/

#endif
