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

#ifndef __G_HASH_H__
#define __G_HASH_H__

typedef char   gchar;
typedef short  gshort;
typedef long   glong;
typedef int    gint;
typedef gint   gboolean;

typedef unsigned char   guchar;
typedef unsigned short  gushort;
typedef unsigned long   gulong;
typedef unsigned int    guint;

typedef void* gpointer;
typedef const void *gconstpointer;

typedef struct _GHashTable  GHashTable;

typedef guint  (*GHashFunc)(gconstpointer  key);
typedef void            (*GDestroyNotify)       (gpointer       data);
typedef gboolean        (*GEqualFunc)           (gconstpointer  a,
                                                 gconstpointer  b);
typedef void            (*GHFunc)               (gpointer       key,
                                                 gpointer       value,
                                                 gpointer       user_data);

typedef gboolean  (*GHRFunc)  (gpointer  key,
                               gpointer  value,
                               gpointer  user_data);

typedef struct _GHashTableIter GHashTableIter;

struct _GHashTableIter
{
  /*< private >*/
  gpointer      dummy1;
  gpointer      dummy2;
  gpointer      dummy3;
  int           dummy4;
  gboolean      dummy5;
  gpointer      dummy6;
};

char* g_strdup (const char *str);

extern GHashTable* g_hash_table_new(GHashFunc hash_func, GEqualFunc key_equal_func);

extern void        g_hash_table_destroy(GHashTable  *hash_table);

extern gboolean    g_hash_table_insert(GHashTable  *hash_table,
                                            gpointer        key,
                                            gpointer        value);

extern gpointer    g_hash_table_lookup(GHashTable *hash_table,
                                            gconstpointer   key);

extern void        g_hash_table_foreach(GHashTable *hash_table,
                                            GHFunc          func,
                                            gpointer        user_data);

extern gpointer    g_hash_table_find(GHashTable *hash_table,
                                            GHRFunc         predicate,
                                            gpointer        user_data);

extern guint       g_hash_table_size(GHashTable *hash_table);

extern void        g_hash_table_iter_init(GHashTableIter *iter,
                                            GHashTable     *hash_table);

extern gboolean    g_hash_table_iter_next(GHashTableIter *iter,
                                            gpointer       *key,
                                            gpointer       *value);


/* Hash Functions
 */
extern gboolean g_str_equal    (gconstpointer  v1, gconstpointer  v2);
extern guint    g_str_hash     (gconstpointer  v);

extern gboolean g_int_equal    (gconstpointer  v1, gconstpointer  v2);
extern guint    g_int_hash     (gconstpointer  v);

extern gboolean g_int64_equal  (gconstpointer  v1, gconstpointer  v2);
extern guint    g_int64_hash   (gconstpointer  v);


#endif /* __G_HASH_H__ */
