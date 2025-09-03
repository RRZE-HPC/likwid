#ifndef INCLUDE_XML_H
#define INCLUDE_XML_H

#include <stddef.h>

typedef enum {
    XML_ELEM_TAG,
    XML_ELEM_EMPTYTAG,
    XML_ELEM_XMLDECL,
    XML_ELEM_DECL,
    XML_ELEM_COMMENT,
    XML_ELEM_TEXT,
    XML_ELEM_ROOT,
} xml_elem_type_t;

struct xml_attr_t {
    char *key;
    char *value;
};

struct xml_elem_t {
    xml_elem_type_t type;

    char *name;

    struct xml_elem_t **child_elem_arr;
    size_t child_elem_count;

    struct xml_attr_t **attr_arr;
    size_t attr_count;

    char *text;
};

int xml_create(struct xml_elem_t **doc);
int xml_create_from_file(struct xml_elem_t **doc, const char *filepath);
int xml_create_from_string(struct xml_elem_t **doc, const char *buf);
void xml_destroy(struct xml_elem_t *doc);
int xml_to_file(struct xml_elem_t *doc, const char *filepath);
int xml_to_string(struct xml_elem_t *doc, char **xmlstr);

int xml_elem_create(struct xml_elem_t **elem, xml_elem_type_t type);
int xml_elem_attach(struct xml_elem_t *parent, struct xml_elem_t *elem);
int xml_elem_create_and_attach(struct xml_elem_t *parent, struct xml_elem_t **elem, xml_elem_type_t type);
void xml_elem_destroy(struct xml_elem_t *elem);
int xml_elem_detach(struct xml_elem_t *parent, struct xml_elem_t *elem);
int xml_elem_destroy_and_detach(struct xml_elem_t *parent, struct xml_elem_t *elem);
int xml_elem_copy(const struct xml_elem_t *src, struct xml_elem_t **dst);

int xml_attr_set(struct xml_elem_t *elem, const char *key, const char *value);
int xml_attr_clear(struct xml_elem_t *elem, const char *key);

#endif //INCLUDE_XML_H
