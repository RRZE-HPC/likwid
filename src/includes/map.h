#ifndef MAP_H
#define MAP_H

#include <ghash.h>

typedef void* mpointer;
typedef void (*map_value_destroy_func)(mpointer data);
typedef void (*map_foreach_func)(mpointer key, mpointer value, mpointer user_data);

typedef enum {
    MAP_KEY_TYPE_STR = 0,
    MAP_KEY_TYPE_INT,
    MAP_KEY_TYPE_BSTR,
    MAX_MAP_KEY_TYPE
} MapKeyType;

typedef struct {
    mpointer key;
    mpointer value;
    mpointer iptr;
} MapValue;

typedef struct {
    int num_values;
    int size;
    int max_size;
    int id;
    GHashTable *ghash;
    MapKeyType key_type;
    MapValue *values;
    map_value_destroy_func value_func;
} Map;

typedef Map* Map_t;

int init_smap(Map_t* map);
int init_map(Map_t* map, MapKeyType type, int max_size, map_value_destroy_func value_func);
int add_smap(Map_t map, char* key, void* val);
int get_smap_by_key(Map_t map, char* key, void** val);
int get_smap_by_idx(Map_t map, int idx, void** val);
void foreach_in_smap(Map_t map, map_foreach_func func, mpointer user_data);
int del_smap(Map_t map, char* key);
void destroy_smap(Map_t map);
int get_map_size(Map_t map);

#endif
