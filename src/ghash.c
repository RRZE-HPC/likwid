/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/.
 */

/*
 * MT safe
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <ghash.h>

#define  g_slice_new(type)      ((type*) malloc (sizeof (type)))

#define HASH_TABLE_MIN_SHIFT 3  /* 1 << 3 == 8 buckets */
#define UNUSED_HASH_VALUE 0
#define TOMBSTONE_HASH_VALUE 1
#define HASH_IS_UNUSED(h_) ((h_) == UNUSED_HASH_VALUE)
#define HASH_IS_TOMBSTONE(h_) ((h_) == TOMBSTONE_HASH_VALUE)
#define HASH_IS_REAL(h_) ((h_) >= 2)

#ifndef    FALSE
#define    FALSE    (0)
#endif

#ifndef    TRUE
#define    TRUE    (!FALSE)
#endif

#undef    MAX
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

#undef    MIN
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))

#undef    ABS
#define ABS(a)       (((a) < 0) ? -(a) : (a))
#define G_LIKELY(expr) (expr)
#define G_UNLIKELY(expr) (expr)

#define _G_NEW(struct_type, n_structs, func) \
        ((struct_type *) g_##func##_n ((n_structs), sizeof (struct_type)))

#define g_new(struct_type, n_structs)            _G_NEW (struct_type, n_structs, malloc)
#define g_new0(struct_type, n_structs)            _G_NEW (struct_type, n_structs, malloc0)

struct _GHashTable
{
  gint             size;
  gint             mod;
  guint            mask;
  gint             nnodes;
  gint             noccupied;  /* nnodes + tombstones */

  gpointer        *keys;
  guint           *hashes;
  gpointer        *values;

  GHashFunc        hash_func;
  GEqualFunc       key_equal_func;
  gint             ref_count;
  GDestroyNotify   key_destroy_func;
  GDestroyNotify   value_destroy_func;
};

typedef struct
{
  GHashTable  *hash_table;
  gpointer     dummy1;
  gpointer     dummy2;
  int          position;
  gboolean     dummy3;
  int          version;
} RealIter;

typedef size_t gsize;

/* Each table size has an associated prime modulo (the first prime
 * lower than the table size) used to find the initial bucket. Probing
 * then works modulo 2^n. The prime modulo is necessary to get a
 * good distribution with poor hash functions.
 */
static const gint prime_mod [] =
{
  1,          /* For 1 << 0 */
  2,
  3,
  7,
  13,
  31,
  61,
  127,
  251,
  509,
  1021,
  2039,
  4093,
  8191,
  16381,
  32749,
  65521,      /* For 1 << 16 */
  131071,
  262139,
  524287,
  1048573,
  2097143,
  4194301,
  8388593,
  16777213,
  33554393,
  67108859,
  134217689,
  268435399,
  536870909,
  1073741789,
  2147483647  /* For 1 << 31 */
};

static gpointer
g_malloc (gsize n_bytes)
{
    if (G_LIKELY (n_bytes))
    {
        gpointer mem;

        mem = malloc (n_bytes);
        if (mem)
            return mem;
    }
    return NULL;
}

gpointer
g_malloc_n (gsize n_blocks, gsize n_block_bytes)
{
  return g_malloc (n_blocks * n_block_bytes);
}


static gpointer
g_malloc0 (gsize n_bytes)
{
    if (G_LIKELY (n_bytes))
    {
        gpointer mem;

        mem = calloc (1, n_bytes);
        if (mem)
            return mem;

    }

    return NULL;
}

static gpointer
g_malloc0_n (gsize n_blocks, gsize n_block_bytes)
{
  return g_malloc0 (n_blocks * n_block_bytes);
}

static void
g_free (gpointer mem)
{
  if (G_LIKELY (mem))
    free (mem);
}


static gpointer
g_memdup (gconstpointer mem,
          guint         byte_size)
{
  gpointer new_mem;

  if (mem)
    {
      new_mem = malloc (byte_size);
      memcpy (new_mem, mem, byte_size);
    }
  else
    new_mem = NULL;

  return new_mem;
}


static void
g_hash_table_set_shift (GHashTable *hash_table, gint shift)
{
  gint i;
  guint mask = 0;

  hash_table->size = 1 << shift;
  hash_table->mod  = prime_mod [shift];

  for (i = 0; i < shift; i++)
    {
      mask <<= 1;
      mask |= 1;
    }

  hash_table->mask = mask;
}

static gint
g_hash_table_find_closest_shift (gint n)
{
  gint i;

  for (i = 0; n; i++)
    n >>= 1;

  return i;
}

static void
g_hash_table_set_shift_from_size (GHashTable *hash_table, gint size)
{
  gint shift;

  shift = g_hash_table_find_closest_shift (size);
  shift = MAX (shift, HASH_TABLE_MIN_SHIFT);

  g_hash_table_set_shift (hash_table, shift);
}

static inline guint
g_hash_table_lookup_node (GHashTable    *hash_table,
                          gconstpointer  key,
                          guint         *hash_return)
{
  guint node_index;
  guint node_hash;
  guint hash_value;
  guint first_tombstone = 0;
  gboolean have_tombstone = FALSE;
  guint step = 0;

  hash_value = hash_table->hash_func (key);
  if (G_UNLIKELY (!HASH_IS_REAL (hash_value)))
    hash_value = 2;

  *hash_return = hash_value;

  node_index = hash_value % hash_table->mod;
  node_hash = hash_table->hashes[node_index];

  while (!HASH_IS_UNUSED (node_hash))
    {
      if (node_hash == hash_value)
        {
          gpointer node_key = hash_table->keys[node_index];

          if (hash_table->key_equal_func)
            {
              if (hash_table->key_equal_func (node_key, key))
                return node_index;
            }
          else if (node_key == key)
            {
              return node_index;
            }
        }
      else if (HASH_IS_TOMBSTONE (node_hash) && !have_tombstone)
        {
          first_tombstone = node_index;
          have_tombstone = TRUE;
        }

      step++;
      node_index += step;
      node_index &= hash_table->mask;
      node_hash = hash_table->hashes[node_index];
    }

  if (have_tombstone)
    return first_tombstone;

  return node_index;
}

static void
g_hash_table_remove_node (GHashTable   *hash_table,
                          gint          i)
{
  gpointer key;
  gpointer value;

  key = hash_table->keys[i];
  value = hash_table->values[i];

  hash_table->hashes[i] = TOMBSTONE_HASH_VALUE;
  hash_table->keys[i] = NULL;
  hash_table->values[i] = NULL;
  hash_table->nnodes--;

  if (hash_table->key_destroy_func != NULL)
    hash_table->key_destroy_func (key);

  if (hash_table->value_destroy_func != NULL)
    hash_table->value_destroy_func (value);

}

int
g_hash_table_remove(GHashTable *hash_table, gpointer key)
{
    int hashret = 0;
    if (key)
    {
        int* key_ptr = (int*)key;
        int keyint = *key_ptr;
        int idx = g_hash_table_lookup_node(hash_table, key_ptr, (guint*)&hashret);
        g_hash_table_remove_node(hash_table, idx);
    }
    return 0;
}

static void
g_hash_table_remove_all_nodes (GHashTable *hash_table,
                               gboolean    notify)
{
  int i;
  gpointer key;
  gpointer value;

  if (!notify ||
      (hash_table->key_destroy_func == NULL &&
       hash_table->value_destroy_func == NULL))
    {
      memset (hash_table->hashes, 0, hash_table->size * sizeof (guint));
      memset (hash_table->keys, 0, hash_table->size * sizeof (gpointer));
      memset (hash_table->values, 0, hash_table->size * sizeof (gpointer));

      return;
    }

  for (i = 0; i < hash_table->size; i++)
    {
      if (HASH_IS_REAL (hash_table->hashes[i]))
      {
        g_hash_table_remove_node(hash_table, i);
      }
/*        {*/
/*          key = hash_table->keys[i];*/
/*          value = hash_table->values[i];*/

/*          hash_table->hashes[i] = UNUSED_HASH_VALUE;*/
/*          hash_table->keys[i] = NULL;*/
/*          hash_table->values[i] = NULL;*/

/*          if (hash_table->key_destroy_func != NULL)*/
/*            hash_table->key_destroy_func (key);*/

/*          if (hash_table->value_destroy_func != NULL)*/
/*            hash_table->value_destroy_func (value);*/
/*        }*/
      else if (HASH_IS_TOMBSTONE (hash_table->hashes[i]))
        {
          hash_table->hashes[i] = UNUSED_HASH_VALUE;
        }
    }

  hash_table->nnodes = 0;
  hash_table->noccupied = 0;
}

static void
g_hash_table_resize (GHashTable *hash_table)
{
  gpointer *new_keys;
  gpointer *new_values;
  guint *new_hashes;
  gint old_size;
  gint i;

  old_size = hash_table->size;
  g_hash_table_set_shift_from_size (hash_table, hash_table->nnodes * 2);

  new_keys = g_new0 (gpointer, hash_table->size);
  if (hash_table->keys == hash_table->values)
    new_values = new_keys;
  else
    new_values = g_new0 (gpointer, hash_table->size);
  new_hashes = g_new0 (guint, hash_table->size);

  for (i = 0; i < old_size; i++)
    {
      guint node_hash = hash_table->hashes[i];
      guint hash_val;
      guint step = 0;

      if (!HASH_IS_REAL (node_hash))
        continue;

      hash_val = node_hash % hash_table->mod;

      while (!HASH_IS_UNUSED (new_hashes[hash_val]))
        {
          step++;
          hash_val += step;
          hash_val &= hash_table->mask;
        }

      new_hashes[hash_val] = hash_table->hashes[i];
      new_keys[hash_val] = hash_table->keys[i];
      new_values[hash_val] = hash_table->values[i];
    }

  if (hash_table->keys != hash_table->values)
    g_free (hash_table->values);

  g_free (hash_table->keys);
  g_free (hash_table->hashes);

  hash_table->keys = new_keys;
  hash_table->values = new_values;
  hash_table->hashes = new_hashes;

  hash_table->noccupied = hash_table->nnodes;
}

static inline void
g_hash_table_maybe_resize (GHashTable *hash_table)
{
  gint noccupied = hash_table->noccupied;
  gint size = hash_table->size;

  if ((size > hash_table->nnodes * 4 && size > 1 << HASH_TABLE_MIN_SHIFT) ||
      (size <= noccupied + (noccupied / 16)))
    g_hash_table_resize (hash_table);
}

GHashTable *
g_hash_table_new_full (GHashFunc      hash_func,
                       GEqualFunc     key_equal_func,
                       GDestroyNotify key_destroy_func,
                       GDestroyNotify value_destroy_func)
{
  GHashTable *hash_table;

  hash_table = g_slice_new (GHashTable);
  g_hash_table_set_shift (hash_table, HASH_TABLE_MIN_SHIFT);
  hash_table->nnodes             = 0;
  hash_table->noccupied          = 0;
  hash_table->hash_func          = hash_func;
  hash_table->key_equal_func     = key_equal_func;
  hash_table->ref_count          = 1;
  hash_table->key_destroy_func   = key_destroy_func;
  hash_table->value_destroy_func = value_destroy_func;
  hash_table->keys               = g_new0 (gpointer, hash_table->size);
  hash_table->values             = hash_table->keys;
  hash_table->hashes             = g_new0 (guint, hash_table->size);

  return hash_table;
}


GHashTable *
g_hash_table_new (GHashFunc  hash_func,
                  GEqualFunc key_equal_func)
{
  /* Thomas Gruber added g_free as destructor of hash table keys. This reduces
   * memory leaks since we know that all key strings are duplicated.
   */
  return g_hash_table_new_full (hash_func, key_equal_func, g_free, NULL);
}


void
g_hash_table_iter_init (GHashTableIter *iter,
                        GHashTable     *hash_table)
{
  RealIter *ri = (RealIter *) iter;
  ri->hash_table = hash_table;
  ri->position = -1;
}

gboolean
g_hash_table_iter_next (GHashTableIter *iter,
                        gpointer       *key,
                        gpointer       *value)
{
  RealIter *ri = (RealIter *) iter;
  gint position;
  position = ri->position;

  do
    {
      position++;
      if (position >= ri->hash_table->size)
        {
          ri->position = position;
          return FALSE;
        }
    }
  while (!HASH_IS_REAL (ri->hash_table->hashes[position]));

  if (key != NULL)
    *key = ri->hash_table->keys[position];
  if (value != NULL)
    *value = ri->hash_table->values[position];

  ri->position = position;
  return TRUE;
}

static gboolean
g_hash_table_insert_node (GHashTable *hash_table,
                          guint       node_index,
                          guint       key_hash,
                          gpointer    new_key,
                          gpointer    new_value,
                          gboolean    keep_new_key,
                          gboolean    reusing_key)
{
  gboolean already_exists;
  guint old_hash;
  gpointer key_to_free = NULL;
  gpointer value_to_free = NULL;

  old_hash = hash_table->hashes[node_index];
  already_exists = HASH_IS_REAL (old_hash);

  if (already_exists)
    {
      value_to_free = hash_table->values[node_index];

      if (keep_new_key)
        {
          key_to_free = hash_table->keys[node_index];
          hash_table->keys[node_index] = new_key;
        }
      else
        key_to_free = new_key;
    }
  else
    {
      hash_table->hashes[node_index] = key_hash;
      hash_table->keys[node_index] = new_key;
    }

  if (G_UNLIKELY (hash_table->keys == hash_table->values && hash_table->keys[node_index] != new_value))
    hash_table->values = g_memdup (hash_table->keys, sizeof (gpointer) * hash_table->size);

  hash_table->values[node_index] = new_value;

  if (!already_exists)
    {
      hash_table->nnodes++;

      if (HASH_IS_UNUSED (old_hash))
        {
          hash_table->noccupied++;
          g_hash_table_maybe_resize (hash_table);
        }

    }

  if (already_exists)
    {
      if (hash_table->key_destroy_func && !reusing_key)
        (* hash_table->key_destroy_func) (key_to_free);
      if (hash_table->value_destroy_func)
        (* hash_table->value_destroy_func) (value_to_free);
    }

  return !already_exists;
}

void
g_hash_table_remove_all (GHashTable *hash_table)
{
  g_hash_table_remove_all_nodes (hash_table, TRUE);
  g_hash_table_maybe_resize (hash_table);
}


void
g_hash_table_unref (GHashTable *hash_table)
{

    if (hash_table->values && hash_table->keys != hash_table->values)
    {
        free(hash_table->values);
        hash_table->values = NULL;
    }
    if (hash_table->keys)
    {
        free(hash_table->keys);
        hash_table->keys = NULL;
    }
    if (hash_table->hashes)
    {
        free(hash_table->hashes);
        hash_table->hashes = NULL;
    }
    free(hash_table);
}

void
g_hash_table_destroy (GHashTable *hash_table)
{
  g_hash_table_remove_all (hash_table);
  g_hash_table_unref (hash_table);
}

gpointer
g_hash_table_lookup (GHashTable    *hash_table,
                     gconstpointer  key)
{
  guint node_index;
  guint node_hash;
  node_index = g_hash_table_lookup_node (hash_table, key, &node_hash);

  return HASH_IS_REAL (hash_table->hashes[node_index])
    ? hash_table->values[node_index]
    : NULL;
}


static gboolean
g_hash_table_insert_internal (GHashTable *hash_table,
                              gpointer    key,
                              gpointer    value,
                              gboolean    keep_new_key)
{
  guint key_hash;
  guint node_index;
  node_index = g_hash_table_lookup_node (hash_table, key, &key_hash);

  return g_hash_table_insert_node (hash_table, node_index, key_hash, key, value, keep_new_key, FALSE);
}

gboolean
g_hash_table_insert (GHashTable *hash_table,
                     gpointer    key,
                     gpointer    value)
{
  return g_hash_table_insert_internal (hash_table, key, value, FALSE);
}

void
g_hash_table_foreach (GHashTable *hash_table,
                      GHFunc      func,
                      gpointer    user_data)
{
  gint i;

  for (i = 0; i < hash_table->size; i++)
    {
      guint node_hash = hash_table->hashes[i];
      gpointer node_key = hash_table->keys[i];
      gpointer node_value = hash_table->values[i];

      if (HASH_IS_REAL (node_hash))
        (* func) (node_key, node_value, user_data);

    }
}

gpointer
g_hash_table_find (GHashTable *hash_table,
                   GHRFunc     predicate,
                   gpointer    user_data)
{
  gint i;
  gboolean match;
  match = FALSE;

  for (i = 0; i < hash_table->size; i++)
    {
      guint node_hash = hash_table->hashes[i];
      gpointer node_key = hash_table->keys[i];
      gpointer node_value = hash_table->values[i];

      if (HASH_IS_REAL (node_hash))
        match = predicate (node_key, node_value, user_data);


      if (match)
        return node_value;
    }

  return NULL;
}

guint
g_hash_table_size (GHashTable *hash_table)
{
  return hash_table->nnodes;
}

gchar*
g_strdup (const gchar *str)
{
  gchar *new_str;
  gsize length;

  if (str)
    {
      length = strlen (str) + 1;
      new_str = g_new (char, length);
      memcpy (new_str, str, length);
    }
  else
    new_str = NULL;

  return new_str;
}


/* Hash functions.
 */

gboolean
g_str_equal (gconstpointer v1,
             gconstpointer v2)
{
  const gchar *string1 = v1;
  const gchar *string2 = v2;

  return strcmp (string1, string2) == 0;
}

guint
g_str_hash (gconstpointer v)
{
  const signed char *p;
  uint32_t h = 5381;

  for (p = v; *p != '\0'; p++)
    h = (h << 5) + h + *p;

  return h;
}

gboolean
g_int_equal (gconstpointer v1,
             gconstpointer v2)
{
  return *((const gint*) v1) == *((const gint*) v2);
}

guint
g_int_hash (gconstpointer v)
{
  return *(const gint*) v;
}

gboolean
g_int64_equal (gconstpointer v1,
               gconstpointer v2)
{
  return *((const int64_t*) v1) == *((const int64_t*) v2);
}

guint
g_int64_hash (gconstpointer v)
{
  return (guint) *(const int64_t*) v;
}

guint
g_direct_hash (gconstpointer v)
{
  return GPOINTER_TO_UINT (v);
}

gboolean
g_direct_equal (gconstpointer v1,
                gconstpointer v2)
{
  return v1 == v2;
}
