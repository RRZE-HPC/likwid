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

#include "libc_wrappers.h"
#include "hash.h"

#define EMPTY 0
#define TOMBSTONE 1
#define INUSE 2

struct hash_entry_t {
   hash_key_t key;
   hash_data_t data;
   hash_hashvalue_t hash_value;
   struct hash_entry_t *next;
   struct hash_entry_t *prev;
   uint32_t status;
};

typedef struct hash_entry_t hash_entry_t;

int create_hashtable(hash_table_t *table, size_t initial_size, hash_func_t hashfunc, 
                     hash_cmp_t keycmp)
{
   hash_entry_t *newtable;
   int entries_per_page;

   entries_per_page = gotcha_getpagesize() / sizeof(hash_entry_t);
   if (initial_size % entries_per_page)
      initial_size += entries_per_page - (initial_size % entries_per_page);

   newtable = (hash_entry_t *) gotcha_malloc(initial_size * sizeof(hash_entry_t));
   if (!newtable)
      return -1;
   gotcha_memset(newtable, 0, initial_size * sizeof(hash_entry_t));

   table->table_size = initial_size;
   table->entry_count = 0;
   table->hashfunc = hashfunc;
   table->keycmp = keycmp;
   table->table = newtable;
   table->head = NULL;
   
   return 0;
}

static hash_entry_t *insert(hash_table_t *table, hash_key_t key, hash_data_t data, hash_hashvalue_t value)
{
   unsigned long index = (unsigned long)value % table->table_size;
   unsigned long startindex = index;

   hash_entry_t *entry = NULL;
   do {
      entry = table->table + index;
      if (entry->status == EMPTY || entry->status == TOMBSTONE) {
         entry->key = key;
         entry->data = data;
         entry->hash_value = value;
         entry->status = INUSE;
         break;
      }
      index++;
      if (index == table->table_size)
         index = 0;
   } while (index != startindex);

   if (!entry)
      return NULL;

   entry->next = table->head;
   entry->prev = NULL;
   if (table->head)
      table->head->prev = entry;
   table->head = entry;
   table->entry_count++;         

   return entry;
}

int grow_hashtable(hash_table_t *table, size_t new_size)
{
   hash_table_t newtable;
   hash_entry_t *result;
   size_t i;

   newtable.table_size = new_size;
   newtable.entry_count = 0;
   newtable.hashfunc = table->hashfunc;
   newtable.keycmp = table->keycmp;
   newtable.table = (hash_entry_t *) gotcha_malloc(new_size * sizeof(hash_entry_t));
   newtable.head = NULL;
   gotcha_memset(newtable.table, 0, new_size * sizeof(hash_entry_t));

   for (i = 0; i < table->table_size; i++) {
      if (table->table[i].status == EMPTY || table->table[i].status == TOMBSTONE)
         continue;
      result = insert(&newtable, table->table[i].key, table->table[i].data,
                      table->table[i].hash_value);
      if (!result) {
         return -1;
      }
   }

   destroy_hashtable(table);
   *table = newtable;
   return 0;
}

int destroy_hashtable(hash_table_t *table)
{
   gotcha_free(table->table);
   table->table_size = 0;
   table->entry_count = 0;
   table->hashfunc = NULL;
   table->keycmp = NULL;
   table->table = NULL;
   table->head = NULL;
   return 0;
}

static int lookup(hash_table_t *table, hash_key_t key, hash_entry_t **entry)
{
   size_t index, startindex;
   hash_hashvalue_t hashval;

   hashval = table->hashfunc(key);
   index = hashval % table->table_size;
   startindex = index;
   
   for (;;) {
      hash_entry_t *cur = table->table + index;
      if ((cur->status == INUSE) && 
          (cur->hash_value == hashval) && 
          (table->keycmp(cur->key, key) == 0)) {
         *entry = cur;
         return 0;
      }

      if (cur->status == EMPTY)
         return -1;
      index++;
      if (index == table->table_size)
         index = 0;
      if (index == startindex)
         return -1;
   }
}

int lookup_hashtable(hash_table_t *table, hash_key_t key, hash_data_t *data)
{
   hash_entry_t *entry;
   int result;

   result = lookup(table, key, &entry);
   if (result == -1)
      return -1;
   *data = entry->data;
   return 0;
}

int addto_hashtable(hash_table_t *table, hash_key_t key, hash_data_t data)
{
   size_t newsize;
   int result;
   hash_hashvalue_t val;
   hash_entry_t *entry;

   newsize = table->table_size;
   while (table->entry_count > newsize/2)
      newsize *= 2;
   if (newsize != table->table_size) {
      result = grow_hashtable(table, newsize);
      if (result == -1)
         return -1;
   }

   val = table->hashfunc(key);
   entry = insert(table, key, data, val);
   if (!entry)
      return -1;

   return 0;
}

int removefrom_hashtable(hash_table_t *table, hash_key_t key)
{
   hash_entry_t *entry;
   int result;

   result = lookup(table, key, &entry);
   if (result == -1)
      return -1;

   entry->key = NULL;
   entry->data = NULL;
   entry->hash_value = 0;
   entry->status = TOMBSTONE;
   if (entry->next)
      entry->next->prev = entry->prev;
   if (entry->prev)
      entry->prev->next = entry->next;
   if (table->head == entry)
      table->head = entry->next;
   //Do not set entry->next to NULL, which would break the iterate & delete
   //idiom used under dlopen_wrapper.
   
   table->entry_count--;
   return 0;
}

int foreach_hash_entry(hash_table_t *table, void *opaque, int (*cb)(hash_key_t key, hash_data_t data, void *opaque))
{
   int result;
   struct hash_entry_t *i;
   for (i = table->head; i != NULL; i = i->next) {
      result = cb(i->key, i->data, opaque);
      if (result != 0)
         return result;
   }
   return 0;
}

hash_hashvalue_t strhash(const char *str)
{
   unsigned long hash = 5381;
   int c;

   while ((c = *str++))
      hash = hash * 33 + c;

   return (hash_hashvalue_t) hash;
}
