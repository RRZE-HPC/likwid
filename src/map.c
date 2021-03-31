/*
 * =======================================================================================
 *
 *      Filename:  map.c
 *
 *      Description:  Implementation a hashmap in C using ghash as backend
 *
 *      Version:   5.1.1
 *      Released:  31.03.2021
 *
 *      Author:   Thoams Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
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

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>

#include <map.h>

#ifdef WITH_BSTRING
#include <bstrlib.h>
#endif

static int* int_dup(int val)
{
    int* valptr = malloc(sizeof(int));
    if (valptr)
    {
        *valptr = val;
    }
    return valptr;
}

#ifdef WITH_BSTRING
gboolean
g_bstr_equal (gconstpointer v1,
              gconstpointer v2)
{
  const_bstring string1 = v1;
  const_bstring string2 = v2;

  return bstrcmp (string1, string2) == BSTR_OK;
}

guint
g_bstr_hash (gconstpointer v)
{
    uint32_t h = 5381;
/*  const signed char *p;*/
/*  if (v == NULL)*/
/*    printf("NULL hash\n");*/
/*  bstring b = (bstring)v;*/
/*  char * base = bstr2cstr(b, '\0');*/

/*  for (p = base; *p != '\0'; p++)*/
/*    h = (h << 5) + h + *p;*/
/*  bcstrfree(base);*/
/*  return h;*/
    const signed char *p;
    bstring b = (bstring)v;
    int i = 0;
    for (; i < blength(b); ++i)
    {
        p = bdataofs(b, i);
        h = (h << 5) + h + *p;
    }
    return h;
}

void g_bstr_destroy(void* v)
{
    bstring b = (bstring)v;
    bdestroy(b);
}
#endif



int init_map(Map_t* map, MapKeyType type, int max_size, map_value_destroy_func value_func)
{
    int err = 0;
    Map* m = malloc(sizeof(Map));
    if (m)
    {
        switch(type)
        {
            case MAP_KEY_TYPE_STR:
                m->ghash = g_hash_table_new_full(g_str_hash, g_str_equal, free, free);
                if (m->ghash)
                {
                    err = 0;
                }
                break;
            case MAP_KEY_TYPE_INT:
                m->ghash = g_hash_table_new_full(g_direct_hash, g_direct_equal, free, free);
                if (m->ghash)
                {
                    err = 0;
                }
                break;
#ifdef WITH_BSTRING
            case MAP_KEY_TYPE_BSTR:
                m->ghash = g_hash_table_new_full(g_bstr_hash, g_bstr_equal, g_bstr_destroy, free);
                if (m->ghash)
                {
                    err = 0;
                }
                break;
#endif
            default:
                printf("Unknown hash type\n");
                free(m);
                err = -ENODEV;
                break;
        }
    }
    else
    {
        err = -ENOMEM;
    }
    if (!err && m)
    {
        m->num_values = 0;
        m->size = 0;
        m->max_size = max_size;
        m->values = NULL;
        m->key_type = type;
        m->value_func = value_func;
        *map = m;
    }
    return err;
}

int init_smap(Map_t* map)
{
    return init_map(map, MAP_KEY_TYPE_STR, 0, NULL);
}

int add_smap(Map_t map, char* key, void* val)
{
    MapValue *mval = NULL;
#ifndef WITH_BSTRING
    gpointer gval = g_hash_table_lookup(map->ghash, key);
#else
    bstring bkey = bfromcstr(key);
    gpointer gval = g_hash_table_lookup(map->ghash, bkey);
#endif
    if (gval)
    {
        return -EEXIST;
    }
    if (map->num_values == map->size)
    {
        if (map->max_size > 0 && map->size == map->max_size)
        {
/*            printf("Map is full\n");*/
            return -ENOSPC;
        }
/*        printf("Realloc to size %d\n", (map->size+1));*/
        MapValue *vals = realloc(map->values, (map->size+1)*sizeof(MapValue));
        if (!vals)
        {
/*            printf("Failed to enlarge values\n");*/
            return -ENOMEM;
        }
        map->values = vals;
        map->values[map->size].key = NULL;
        map->values[map->size].value = NULL;
        map->values[map->size].iptr = NULL;
        map->size++;
/*        printf("Realloc done to size %d\n",map->size);*/
    }
    if (map->num_values < map->size)
    {
        int idx = map->size-1;
/*        printf("Startidx %d\n", idx);*/
        while (idx >= 0 && map->values[idx].value != NULL)
        {
            idx--;
        }
/*        printf("Adding value at index %d\n", idx);*/
#ifndef WITH_BSTRING
        map->values[idx].key = g_strdup(key);
#else
        map->values[idx].key = bkey;
#endif
        map->values[idx].value = val;
        map->values[idx].iptr = int_dup(idx);
/*        printf("Adding %s -> %d\n", key, idx);*/
        g_hash_table_insert(map->ghash, map->values[idx].key, map->values[idx].iptr);
        map->num_values++;
        return idx;
    }

    return -1;
}

int get_smap_by_key(Map_t map, char* key, void** val)
{
#ifndef WITH_BSTRING
    gpointer gval = g_hash_table_lookup(map->ghash, key);
#else
    bstring bkey = bfromcstr(key);
    gpointer gval = g_hash_table_lookup(map->ghash, bkey);
    bdestroy(bkey);
#endif
    if (gval)
    {
        if (val)
        {
            int* ival = (int*)gval;
            MapValue *mval = &(map->values[*ival]);
            *val = (void*)mval->value;
        }
        return 0;
    }
    return -ENOENT;
}

int get_smap_by_idx(Map_t map, int idx, void** val)
{
    if (idx >= 0 && idx < map->size)
    {
        MapValue *mval = &(map->values[idx]);
        *val = (void*)mval->value;
        return 0;
    }
    return -ENOENT;
}

int get_map_size(Map_t map)
{
    if (map)
    {
        return map->num_values;
    }
    return -1;
}

void foreach_in_smap(Map_t map, map_foreach_func func, mpointer user_data)
{
    if (map && func)
    {
        g_hash_table_foreach(map->ghash, func, user_data);
    }
}

int del_smap(Map_t map, char* key)
{
#ifndef WITH_BSTRING
    gpointer gval = g_hash_table_lookup(map->ghash, key);
#else
    bstring bkey = bfromcstr(key);
    gpointer gval = g_hash_table_lookup(map->ghash, bkey);
#endif
    if (gval)
    {
        int* ival = (int*)gval;
/*        printf("Remove %s at index %d\n", key, *ival);*/
        map->values[*ival].key = NULL;
        if (map->value_func != NULL)
        {
            map->value_func(map->values[*ival].value);
        }
        map->values[*ival].value = NULL;
        map->values[*ival].iptr = NULL;
#ifndef WITH_BSTRING
        g_hash_table_remove(map->ghash, key);
#else
        g_hash_table_remove(map->ghash, bkey);
        bdestroy(bkey);
#endif
        map->num_values--;
/*        printf("num_values %d size %d\n", map->num_values, map->size);*/
        return 0;
    }
    return -ENOENT;
}

void destroy_smap(Map_t map)
{
    if (map)
    {
        g_hash_table_destroy(map->ghash);
        map->ghash = NULL;
        if (map->values)
        {
            if (map->value_func)
            {
                for (int i = 0; i < map->size; i++)
                {
                    if (map->values[i].value != NULL)
                    {
/*                        printf("Calling free function for map value %d\n", i);*/
                        map->value_func(map->values[i].value);
                    }
                }
            }
            free(map->values);
            map->values = NULL;
        }
        map->num_values = 0;
        map->size = 0;
        free(map);
    }
}
