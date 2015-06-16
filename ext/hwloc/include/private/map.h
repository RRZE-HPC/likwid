/*
 * Copyright © 2013 Inria.  All rights reserved.
 * Copyright © 2013 Cisco Systems, Inc.  All rights reserved.
 * Copyright © 2013-2014 University of Wisconsin-La Crosse.
 *                         All rights reserved.
 *
 * See COPYING in top-level directory.
 *
 * $HEADER$
 */

#ifndef _PRIVATE_NETLOC_MAP_H_
#define _PRIVATE_NETLOC_MAP_H_

#include <hwloc.h>
#include <netloc.h>


struct netloc_map__subnet;
struct netloc_map__server;

struct netloc_map__port {
  struct netloc_map__subnet * subnet;
  struct netloc_map__server * server;

  netloc_edge_t * edge;

  unsigned hwloc_obj_depth;
  unsigned hwloc_obj_index;
  hwloc_obj_t hwloc_obj; /* cached from depth/index above,
			  * only non-NULL if the topology hasn't been compressed in the meantime.
			  */

  struct netloc_map__port *prev, *next;

  char id[0];
};

struct netloc_map__subnet {
  netloc_topology_t topology;
  netloc_network_type_t type;

  int port_by_id_ready;
  struct netloc_dt_lookup_table port_by_id;

  struct netloc_map__subnet *prev, *next;

  struct netloc_map__port *port_first, *port_last;
  unsigned ports_nr;

  char id[0];
};

struct netloc_map__server {
  hwloc_topology_t topology; /* NULL if compressed */
#if HWLOC_API_VERSION >= 0x00010800
  hwloc_topology_diff_t topology_diff;
  struct netloc_map__server *topology_diff_refserver;
#endif

  int usecount; /* references from the application,
		 * or from topology diff for other servers.
		 * no compression when > 0
		 */

  unsigned nr_ports;
  unsigned nr_ports_allocated;
  struct netloc_map__port ** ports;

  struct netloc_map__server *prev, *next;
  struct netloc_map *map;

  char name[0];
};

enum netloc_map_verbose_flags_e {
  NETLOC_MAP_VERBOSE_FLAG_COMPRESS = (1<<0)
};

struct netloc_map {
  unsigned long flags;
  unsigned long verbose_flags;

  unsigned server_ports_nr; /* needed during build, to create large-enough hash tables */

  char *hwloc_xml_path;
  struct netloc_dt_lookup_table server_by_name;
  struct netloc_map__server *server_first, *server_last;
  unsigned servers_nr;

  char *netloc_data_path;
  struct netloc_dt_lookup_table subnet_by_id[NETLOC_NETWORK_TYPE_INVALID]; /* enough room for existing types */
  struct netloc_map__subnet *subnet_first, *subnet_last;
  unsigned subnets_nr;

  int merged;
};

struct netloc_map__paths {
  struct netloc_map *map;
  unsigned long flags;
  unsigned nr_paths;
  struct netloc_map__path {
    /* FIXME: cache the subnet */
    unsigned nr_edges;
    struct netloc_map_edge_s *edges;
  } * paths;
};

#endif /* _PRIVATE_NETLOC_MAP_H_ */
