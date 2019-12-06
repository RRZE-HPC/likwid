/*
This file is part of GOTCHA.  For copyright information see the COPYRIGHT
file in the top level directory, or at
https://github.com/LLNL/gotcha/blob/master/COPYRIGHT
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.  This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU Lesser General Public License
for more details.  You should have received a copy of the GNU Lesser General
Public License along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#if !defined(HASH_H_)
#define HASH_H_

#include <stdlib.h>
#include <stdint.h>

typedef void* hash_key_t;
typedef void* hash_data_t;
typedef int hash_hashvalue_t;
typedef hash_hashvalue_t (*hash_func_t)(hash_data_t data);
typedef int (*hash_cmp_t)(hash_key_t a, hash_key_t b);

struct hash_entry_t;

typedef struct 
{
   size_t table_size;
   size_t entry_count;
   hash_func_t hashfunc;
   hash_cmp_t keycmp;
   struct hash_entry_t *table;
   struct hash_entry_t *head;
} hash_table_t;

int create_hashtable(hash_table_t *table, size_t initial_size, hash_func_t func, 
                     hash_cmp_t keycmp);
int grow_hashtable(hash_table_t *table, size_t new_size);
int destroy_hashtable(hash_table_t *table);

int lookup_hashtable(hash_table_t *table, hash_key_t key, hash_data_t *data);
int addto_hashtable(hash_table_t *table, hash_key_t key, hash_data_t data);
int removefrom_hashtable(hash_table_t *table, hash_key_t key);
int foreach_hash_entry(hash_table_t *table, void *opaque, int (*cb)(hash_key_t key, hash_data_t data, void *opaque));

hash_hashvalue_t strhash(const char *str);

#endif
