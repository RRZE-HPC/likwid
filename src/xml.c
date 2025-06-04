#include "xml.h"

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int xml_parse(struct xml_elem_t *elem, const char *buf, size_t *buf_pos);

static int attr_create(struct xml_attr_t **attr, const char *key, const char *value) {
    char *key_copy = NULL;
    char *value_copy = NULL;
    struct xml_attr_t *new_attr = malloc(sizeof(*new_attr));
    if (!new_attr)
        goto error;

    key_copy = strdup(key);
    if (!key_copy)
        goto error;

    value_copy = strdup(value);
    if (!value_copy)
        goto error;

    new_attr->key = key_copy;
    new_attr->value = value_copy;
    *attr = new_attr;
    return 0;

error:
    free(key_copy);
    free(value_copy);
    free(new_attr);
    return -errno;
}

static void attr_destroy(struct xml_attr_t *attr) {
    if (!attr)
        return;
    free(attr->key);
    free(attr->value);
    free(attr);
}

int xml_attr_set(struct xml_elem_t *elem, const char *key, const char *value) {
    // Search if the key already exists and replace it if so.
    for (size_t i = 0; i < elem->attr_count; i++) {
        if (strcmp(elem->attr_arr[i]->key, key) == 0) {
            char *value_copy = strdup(value);
            if (!value_copy)
                return -errno;
            free(elem->attr_arr[i]->value);
            elem->attr_arr[i]->value = value_copy;
            return 0;
        }
    }

    struct xml_attr_t *new_attr;
    int err = attr_create(&new_attr, key, value);
    if (err < 0)
        return err;

    size_t new_attr_count = elem->attr_count + 1;
    struct xml_attr_t **new_attr_arr = realloc(elem->attr_arr, sizeof(*new_attr_arr) * new_attr_count);
    if (!new_attr_arr) {
        err = -errno;
        attr_destroy(new_attr);
        return err;
    }

    elem->attr_arr = new_attr_arr;
    elem->attr_arr[elem->attr_count] = new_attr;
    elem->attr_count = new_attr_count;
    return 0;
}

int xml_attr_clear(struct xml_elem_t *elem, const char *key) {
    size_t index_to_delete;
    bool found = false;

    for (size_t i = 0; i < elem->attr_count; i++) {
        if (strcmp(elem->attr_arr[i]->key, key) == 0) {
            index_to_delete = i;
            found = true;
            break;
        }
    }

    if (!found)
        return -EINVAL;

    const size_t new_attr_count = elem->attr_count - 1;
    struct xml_attr_t **new_attr_arr = calloc(new_attr_count, sizeof(*new_attr_arr));
    if (!new_attr_arr)
        return -errno;

    attr_destroy(elem->attr_arr[index_to_delete]);

    for (size_t i = index_to_delete; i < new_attr_count; i++)
        new_attr_arr[i] = elem->attr_arr[i+1];

    free(elem->attr_arr);
    elem->attr_arr = new_attr_arr;
    elem->attr_count = new_attr_count;
    return 0;
}

static int attr_copy(const struct xml_attr_t *src, struct xml_attr_t **dst) {
    if (!src || !dst)
        return -EINVAL;

    struct xml_attr_t *new_attr = calloc(1, sizeof(*new_attr));
    if (!new_attr)
        return -errno;

    int err;
    new_attr->key = strdup(src->key);
    if (!new_attr->key) {
        err = -errno;
        goto error;
    }

    new_attr->value = strdup(src->value);
    if (!new_attr->value) {
        err = -errno;
        goto error;
    }

    *dst = new_attr;
    return 0;

error:
    attr_destroy(new_attr);
    return err;
}

int xml_elem_create(struct xml_elem_t **elem, xml_elem_type_t type) {
    struct xml_elem_t *new_elem = malloc(sizeof(*new_elem));
    if (!new_elem)
        return -errno;

    new_elem->type = type;
    new_elem->name = NULL;
    new_elem->child_elem_arr = NULL;
    new_elem->child_elem_count = 0;
    new_elem->attr_arr = NULL;
    new_elem->attr_count = 0;
    new_elem->text = NULL;

    *elem = new_elem;
    return 0;
}

int xml_elem_attach(struct xml_elem_t *parent, struct xml_elem_t *elem) {
    // alloc new parent's child array
    const size_t new_child_elem_count = parent->child_elem_count + 1;
    struct xml_elem_t **new_child_elem_arr = calloc(new_child_elem_count, sizeof(*new_child_elem_arr));
    if (!new_child_elem_arr)
        return -errno;

    // copy child array to new array
    for (size_t i = 0; i < parent->child_elem_count; i++)
        new_child_elem_arr[i] = parent->child_elem_arr[i];
    new_child_elem_arr[parent->child_elem_count] = elem;
    free(parent->child_elem_arr);
    parent->child_elem_arr = new_child_elem_arr;
    parent->child_elem_count = new_child_elem_count;
    return 0;
}

int xml_elem_create_and_attach(struct xml_elem_t *parent, struct xml_elem_t **elem, xml_elem_type_t type) {
    // alloc new element
    struct xml_elem_t *new_elem;
    int err = xml_elem_create(&new_elem, type);
    if (err < 0)
        return err;

    err = xml_elem_attach(parent, new_elem);
    if (err < 0) {
        xml_elem_destroy(new_elem);
        return err;
    }

    // return new element
    *elem = new_elem;
    return 0;
}

int xml_elem_detach(struct xml_elem_t *parent, struct xml_elem_t *elem) {
    // find existing element
    bool found = false;
    size_t index_to_delete = 0;

    for (size_t i = 0; i < parent->child_elem_count; i++) {
        if (parent->child_elem_arr[i] == elem) {
            found = true;
            index_to_delete = i;
            break;
        }
    }

    if (!found)
        return -EINVAL;

    // alloc new parent's child array
    const size_t new_child_elem_count = parent->child_elem_count - 1;
    struct xml_elem_t **new_child_elem_arr = calloc(new_child_elem_count, sizeof(*new_child_elem_arr));
    if (!new_child_elem_arr)
        return -errno;

    // copy from old array without the element to delete
    for (size_t i = index_to_delete; i < new_child_elem_count; i++)
        new_child_elem_arr[i] = parent->child_elem_arr[i+1];

    free(parent->child_elem_arr);
    parent->child_elem_arr = new_child_elem_arr;
    parent->child_elem_count = new_child_elem_count;
    return 0;
}

int xml_elem_destroy_and_detach(struct xml_elem_t *parent, struct xml_elem_t *elem) {
    int err = xml_elem_detach(parent, elem);
    if (err < 0)
        return err;

    xml_elem_destroy(elem);
    return 0;
}

int xml_elem_copy(const struct xml_elem_t *src, struct xml_elem_t **dst) {
    if (!src || !dst)
        return -EINVAL;

    struct xml_elem_t *new_elem;
    int err = xml_elem_create(&new_elem, src->type);
    if (err < 0)
        return err;

    // copy name
    if (src->name) {
        new_elem->name = strdup(src->name);
        if (!new_elem->name) {
            err = -errno;
            goto error;
        }
    }

    // copy children
    if (src->child_elem_arr) {
        new_elem->child_elem_arr = calloc(src->child_elem_count, sizeof(*new_elem->child_elem_arr));
        if (!new_elem->child_elem_arr) {
            err = -errno;
            goto error;
        }
        new_elem->child_elem_count = src->child_elem_count;
        for (size_t i = 0; i < new_elem->child_elem_count; i++) {
            err = xml_elem_copy(src->child_elem_arr[i], &new_elem->child_elem_arr[i]);
            if (err < 0)
                goto error;
        }
    }

    // copy attributes
    if (src->attr_arr) {
        new_elem->attr_arr = calloc(src->attr_count, sizeof(*new_elem->attr_arr));
        if (!new_elem->attr_arr) {
            err = -errno;
            goto error;
        }
        new_elem->attr_count = src->attr_count;
        for (size_t i = 0; i < new_elem->attr_count; i++) {
            err = attr_copy(src->attr_arr[i], &new_elem->attr_arr[i]);
            if (err < 0)
                goto error;
        }
    }

    // copy text
    if (src->text) {
        new_elem->text = strdup(src->text);
        if (!new_elem->text) {
            err = -errno;
            goto error;
        }
    }

    *dst = new_elem;
    return 0;

error:
    xml_elem_destroy(new_elem);
    return err;
}

void xml_elem_destroy(struct xml_elem_t *elem) {
    if (!elem)
        return;

    free(elem->name);

    for (size_t i = 0; i < elem->child_elem_count; i++)
        xml_elem_destroy(elem->child_elem_arr[i]);
    free(elem->child_elem_arr);

    for (size_t i = 0; i < elem->attr_count; i++)
        attr_destroy(elem->attr_arr[i]);
    free(elem->attr_arr);

    free(elem->text);
    free(elem);
}

int xml_create(struct xml_elem_t **doc) {
    return xml_elem_create(doc, XML_ELEM_ROOT);
}

int xml_create_from_file(struct xml_elem_t **doc, const char *filepath) {
    FILE *file = fopen(filepath, "r");
    if (!file)
        return -errno;

    char *buf = NULL;
    if (fseek(file, 0, SEEK_END))
        goto error;

    long tmp = ftell(file);
    if (tmp < 0)
        goto error;

    if (fseek(file, 0, SEEK_SET))
        goto error;

    const size_t buf_size = (size_t)tmp;
    buf = malloc(buf_size + 1);
    if (!buf)
        goto error;

    const size_t buf_read_size = fread(buf, 1, buf_size, file);
    buf[buf_read_size] = '\0';

    fclose(file);

    int err = xml_create_from_string(doc, buf);
    free(buf);
    return err;

error:
    if (file)
        fclose(file);
    free(buf);
    return -errno;
}

int xml_create_from_string(struct xml_elem_t **doc, const char *buf) {
    struct xml_elem_t *root_elem = NULL;
    int err = xml_create(&root_elem);
    if (err < 0)
        goto error;

    size_t buf_pos = 0;
    err = xml_parse(root_elem, buf, &buf_pos);
    if (err < 0)
        goto error;

    *doc = root_elem;
    return 0;

error:
    xml_destroy(root_elem);
    return err;
}

void xml_destroy(struct xml_elem_t *doc) {
    xml_elem_destroy(doc);
}

int xml_to_file(struct xml_elem_t *doc, const char *filepath) {
    char *xmlstr;
    int err = xml_to_string(doc, &xmlstr);
    if (err < 0)
        return err;

    FILE *file = fopen(filepath, "w");
    if (!file) {
        err = -errno;
        goto error;
    }

    const size_t bytes_to_write = strlen(xmlstr);
    const size_t bytes_written = fwrite(xmlstr, 1, bytes_to_write, file);
    if (bytes_to_write != bytes_written) {
        err = -EPERM;
        goto error;
    }

    if (fclose(file)) {
        err = -errno;
        goto error;
    }

    return 0;

error:
    if (file)
        fclose(file);
    free(xmlstr);
    return err;
}

struct strbuf_t {
    char *buf;
    size_t chars_used;
    size_t bytes_used;
    size_t cap;
};

static int strbuf_fmt_append(struct strbuf_t *strbuf, const char *fmt, ...) {
    size_t bytes_avail;
    size_t bytes_req;

    while (true) {
        bytes_avail = strbuf->cap - strbuf->bytes_used;

        va_list args;
        va_start(args, fmt);
        int err = vsnprintf(&strbuf->buf[strbuf->chars_used], bytes_avail, fmt, args);
        if (err < 0)
            return -EINVAL;
        bytes_req = (size_t)err;
        va_end(args);

        // Was there enough space available for appending?
        if (bytes_avail > bytes_req)
            break;

        assert(strbuf->cap != 0);
        const size_t newcap = strbuf->cap * 2;
        char *newbuf = realloc(strbuf->buf, newcap);
        if (!newbuf)
            return -errno;

        strbuf->buf = newbuf;
        strbuf->cap = newcap;
    }

    strbuf->bytes_used += bytes_req;
    strbuf->chars_used += bytes_req;

    return 0;
}

static int xml_elem_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem);

static int xml_attrlist_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    for (size_t i = 0; i < elem->attr_count; i++) {
        /* WARNING: This is wrong. We should properly escape the strings,
         * which we currently don't do. */
        struct xml_attr_t *a = elem->attr_arr[i];
        int err = strbuf_fmt_append(strbuf, " %s=\"%s\"", a->key, a->value);
        if (err < 0)
            return err;
    }
    return 0;
}

static int xml_tag_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    // output tag begin
    if (elem->type != XML_ELEM_ROOT) {
        assert(elem->name);
        int err = strbuf_fmt_append(strbuf, "<%s", elem->name);
        if (err < 0)
            return err;

        err = xml_attrlist_to_string(strbuf, elem);
        if (err < 0)
            return err;

        // quit early for empty tags, since they don't have children and no closing tag
        if (elem->type == XML_ELEM_EMPTYTAG)
            return strbuf_fmt_append(strbuf, "/>");
        
        err = strbuf_fmt_append(strbuf, ">");
        if (err < 0)
            return err;
    }

    // output children
    for (size_t i = 0; i < elem->child_elem_count; i++) {
        int err = xml_elem_to_string(strbuf, elem->child_elem_arr[i]);
        if (err < 0)
            return err;
    }
    
    // output tag end
    if (elem->type != XML_ELEM_ROOT) {
        int err = strbuf_fmt_append(strbuf, "</%s>", elem->name);
        if (err < 0)
            return err;
    }

    return 0;
}

static int xml_xmldecl_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    assert(elem->type == XML_ELEM_XMLDECL);
    int err = strbuf_fmt_append(strbuf, "<?%s", elem->name);
    if (err < 0)
        return err;

    err = xml_attrlist_to_string(strbuf, elem);
    if (err < 0)
        return err;

    return strbuf_fmt_append(strbuf, "?>");
}

static int xml_decl_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    assert(elem->type == XML_ELEM_DECL);
    return strbuf_fmt_append(strbuf, "<!%s>", elem->text);
}

static int xml_comment_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    assert(elem->type == XML_ELEM_COMMENT);
    return strbuf_fmt_append(strbuf, "<!--%s-->", elem->text);
}

static int xml_text_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    assert(elem->type == XML_ELEM_TEXT);
    return strbuf_fmt_append(strbuf, "%s", elem->text);
}

static int xml_elem_to_string(struct strbuf_t *strbuf, struct xml_elem_t *elem) {
    switch (elem->type) {
    case XML_ELEM_ROOT:
    case XML_ELEM_TAG:
    case XML_ELEM_EMPTYTAG:
        return xml_tag_to_string(strbuf, elem);
    case XML_ELEM_XMLDECL:
        return xml_xmldecl_to_string(strbuf, elem);
    case XML_ELEM_DECL:
        return xml_decl_to_string(strbuf, elem);
    case XML_ELEM_COMMENT:
        return xml_comment_to_string(strbuf, elem);
    case XML_ELEM_TEXT:
        return xml_text_to_string(strbuf, elem);
    default:
        return -EINVAL;
    }
}

int xml_to_string(struct xml_elem_t *doc, char **xmlstr) {
    struct strbuf_t strbuf = {
        .cap = 4096,
        .bytes_used = 1,
        .chars_used = 0,
    };

    strbuf.buf = malloc(strbuf.cap);
    if (!strbuf.buf)
        return -errno;

    assert(strbuf.cap >= strbuf.bytes_used);

    strbuf.buf[0] = '\0';

    int err = xml_elem_to_string(&strbuf, doc);
    if (err < 0) {
        free(strbuf.buf);
        return err;
    }

    *xmlstr = strbuf.buf;
    return 0;
}

static bool is_word_char(char c) {
    if (c >= '0' && c <= '9')
        return true;
    if (c >= 'a' && c <= 'z')
        return true;
    if (c >= 'A' && c <= 'Z')
        return true;
    if (c == '_')
        return true;
    return false;
}

static bool is_singlequote(char c) {
    return c == '\'';
}

static bool is_doublequote(char c) {
    return c == '\"';
}

static bool is_whitespace_char(char c) {
    switch (c) {
    case '\t':
    case '\n':
    case '\v':
    case '\f':
    case '\r':
    case ' ':
        return true;
    default:
        return false;
    }
}

static bool is_whitespace_str(const char *beg, const char *end) {
    for (const char *c = beg; c != end; c++) {
        if (!is_whitespace_char(*c))
            return false;
    }
    return true;
}

static int skip_none_past_string(const char *buf, size_t *buf_pos, const char *match) {
    if (strncmp(&buf[*buf_pos], match, strlen(match)) != 0)
        return -EINVAL;
    *buf_pos += strlen(match);
    return 0;
}

static int skip_ws_past_string(const char *buf, size_t *buf_pos, const char *match) {
    const char *beg = &buf[*buf_pos];
    const char *end = strstr(beg, match);
    if (!end)
        return -EINVAL;

    if (!is_whitespace_str(beg, end))
        return -EINVAL;

    *buf_pos += (size_t)(end - beg) + strlen(match);
    return 0;
}

static int skip_any_past_string(const char *buf, size_t *buf_pos, const char *match) {
    const char *beg = &buf[*buf_pos];
    const char *end = strstr(beg, match);
    if (!end)
        return -EINVAL;

    *buf_pos += (size_t)(end - beg) + strlen(match);
    return 0;
}

static int xml_parse_word(const char *buf, size_t *buf_pos, char **word) {
    size_t new_buf_pos = *buf_pos;

    // skip initial whitespace
    while (true) {
        const char c = buf[new_buf_pos];
        if (c == 0)
            return -EINVAL;
        if (!is_whitespace_char(c))
            break;
        new_buf_pos++;
    }

    const char *beg = &buf[new_buf_pos];
    while (true) {
        const char c = buf[new_buf_pos];
        if (c == 0)
            return -EINVAL;
        if (!is_word_char(c))
            break;
        new_buf_pos++;
    }

    const char *end = &buf[new_buf_pos];
    if (beg == end)
        return -EINVAL;

    char *new_word = strndup(beg, (size_t)(end - beg));
    if (!new_word)
        return -errno;

    *word = new_word;
    *buf_pos = new_buf_pos;
    return 0;
}

static int xml_parse_string(const char *buf, size_t *buf_pos, char **string) {
    // mystring
    // "my string"
    // 'my string'
    size_t new_buf_pos = *buf_pos;

    // skip initial whitespace
    while (true) {
        const char c = buf[new_buf_pos];
        if (c == 0)
            return -EINVAL;
        if (!is_whitespace_char(c))
            break;
        new_buf_pos++;
    }

    bool has_quotes = false;
    bool (*term_cond)(char c);
    if (buf[new_buf_pos] == '\'') {
        term_cond = &is_singlequote;
        new_buf_pos++;
        has_quotes = true;
    } else if (buf[new_buf_pos] == '\"') {
        term_cond = &is_doublequote;
        new_buf_pos++;
        has_quotes = true;
    } else {
        term_cond = &is_whitespace_char;
    }

    const char *beg = &buf[new_buf_pos];
    while (true) {
        const char c = buf[new_buf_pos];
        if (c == 0)
            return -EINVAL;
        if (c == '>' || (!has_quotes && c == '/'))
            break;
        if (term_cond(c))
            break;
        new_buf_pos++;
    }

    const char *end = &buf[new_buf_pos];

    if (has_quotes)
        new_buf_pos++;

    char *new_string = strndup(beg, (size_t)(end - beg));
    if (!new_string)
        return -errno;

    *string = new_string;
    *buf_pos = new_buf_pos;
    return 0;
}

static int xml_parse_attrlist(struct xml_elem_t *elem, const char *buf, size_t *buf_pos) {
    // myattr="hello" myattr2=there myattr3 = "foobar"
    while (true) {
        char *key;
        int err = xml_parse_word(buf, buf_pos, &key);
        if (err == -EINVAL)
            break;

        err = skip_ws_past_string(buf, buf_pos, "=");
        if (err < 0) {
            free(key);
            return -EINVAL;
        }

        char *value;
        err = xml_parse_string(buf, buf_pos, &value);
        if (err < 0) {
            free(key);
            return -EINVAL;
        }

        err = xml_attr_set(elem, key, value);
        free(key);
        free(value);
        if (err < 0)
            return -EINVAL;
    }
    return 0;
}

static int xml_parse_xmldecl(struct xml_elem_t *parent, const char *buf, size_t *buf_pos) {
    // <?xml version="1.0" encoding="UTF-8"?>
    int err = skip_none_past_string(buf, buf_pos, "<?");
    assert(err == 0);

    struct xml_elem_t *elem;
    err = xml_elem_create_and_attach(parent, &elem, XML_ELEM_XMLDECL);
    if (err < 0)
        return err;

    err = xml_parse_word(buf, buf_pos, &elem->name);
    if (err < 0)
        return err;

    err = xml_parse_attrlist(elem, buf, buf_pos);
    if (err < 0)
        return err;

    return skip_ws_past_string(buf, buf_pos, "?>");
}

static int xml_parse_comment(struct xml_elem_t *parent, const char *buf, size_t *buf_pos) {
    // <!-- ....... -->
    int err = skip_none_past_string(buf, buf_pos, "<!--");
    assert(err == 0);

    const char *start = &buf[*buf_pos];
    const char *end_str = "-->";
    const char *end = strstr(start, end_str);
    if (!end)
        return -EINVAL;

    struct xml_elem_t *elem;
    err = xml_elem_create_and_attach(parent, &elem, XML_ELEM_COMMENT);
    if (err < 0)
        return err;

    char *text = strndup(start, (size_t)(end - start));
    if (!text)
        return -errno;
    elem->text = text;

    return skip_any_past_string(buf, buf_pos, end_str);
}

static int xml_parse_decl(struct xml_elem_t *parent, const char *buf, size_t *buf_pos) {
    // <!DOCTYPE topology SYSTEM "hwloc2.dtd">
    int err = skip_none_past_string(buf, buf_pos, "<!");
    assert(err == 0);

    /* WARNING: Skipping past the next '>' is "wrong", since it ignores the syntax
     * of XML declarations. However, because the syntax can be complex, we will
     * ignore this error for now. */
    const char *start = &buf[*buf_pos];
    const char *end_str = ">";
    const char *end = strstr(start, end_str);
    if (!end)
        return -EINVAL;

    struct xml_elem_t *elem;
    err = xml_elem_create_and_attach(parent, &elem, XML_ELEM_DECL);
    if (err < 0)
        return err;

    char *text = strndup(start, (size_t)(end - start));
    if (!text)
        return -errno;
    elem->text = text;

    return skip_any_past_string(buf, buf_pos, end_str);
}

static int xml_parse_tag(struct xml_elem_t *parent, const char *buf, size_t *buf_pos) {
    // <mytag myattr1="hello" myattr2="you there"> ... </mytag>
    // OR
    // <mytag myattr1="hello" myattr2="you there"/>
    int err = skip_none_past_string(buf, buf_pos, "<");
    assert(err == 0);

    struct xml_elem_t *elem;
    err = xml_elem_create_and_attach(parent, &elem, XML_ELEM_TAG);
    if (err < 0)
        return err;

    err = xml_parse_word(buf, buf_pos, &elem->name);
    if (err < 0)
        return err;

    err = xml_parse_attrlist(elem, buf, buf_pos);
    if (err < 0)
        return err;

    err = skip_ws_past_string(buf, buf_pos, "/>");
    if (err == 0) {
        elem->type = XML_ELEM_EMPTYTAG;
        return 0;
    }

    err = skip_ws_past_string(buf, buf_pos, ">");
    if (err < 0)
        return err;

    // If we have a normal tag, parse this sub tag
    err = xml_parse(elem, buf, buf_pos);
    if (err < 0)
        return err;

    // Parse the closing tag
    err = skip_none_past_string(buf, buf_pos, "</");
    if (err < 0)
        return err;

    err = skip_none_past_string(buf, buf_pos, elem->name);
    if (err < 0)
        return err;

    return skip_ws_past_string(buf, buf_pos, ">");
}

static int xml_parse_text(struct xml_elem_t *parent, const char *buf, size_t *buf_pos) {
    // Parse anything which may occur between elements, which is usually just blank
    // whitespace stuff or actual text.
    const char *beg = &buf[*buf_pos];

    while (true) {
        char c = buf[*buf_pos];
        if (c == '<' || c == '\0')
            break;

        if (c == '>' || c == '\'' || c == '\"')
            return -EINVAL;

        (*buf_pos)++;
    }

    const char *end = &buf[*buf_pos];
    //if (is_whitespace_str(beg, end))
    //    return 0;

    struct xml_elem_t *elem;
    int err = xml_elem_create_and_attach(parent, &elem, XML_ELEM_TEXT);
    if (err < 0)
        return err;

    const size_t num_characters = (size_t)(end - beg);
    elem->text = strndup(beg, num_characters);
    if (elem->text == NULL)
        return -errno;

    return 0;
}

static int xml_parse(struct xml_elem_t *elem, const char *buf, size_t *buf_pos) {
    int err = 0;

    while (buf[*buf_pos] && err == 0) {
        const char *buf_cur = &buf[*buf_pos];
        if (strncmp(buf_cur, "</", 2) == 0) {
            break;
        } else if (strncmp(buf_cur, "<?", 2) == 0) {
            err = xml_parse_xmldecl(elem, buf, buf_pos);
        } else if (strncmp(buf_cur, "<!--", 4) == 0) {
            err = xml_parse_comment(elem, buf, buf_pos);
        } else if (strncmp(buf_cur, "<!", 2) == 0) {
            err = xml_parse_decl(elem, buf, buf_pos);
        } else if (strncmp(buf_cur, "<", 1) == 0) {
            err = xml_parse_tag(elem, buf, buf_pos);
        } else {
            err = xml_parse_text(elem, buf, buf_pos);
        }
    }

    return err;
}
