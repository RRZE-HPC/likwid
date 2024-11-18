// calculator_exptree.h

struct exptree_node; // fwd declaration

// cannot fwd declare CounterList because it was an anonymous struct (changed)
// thus we named CounterList to avoid unnecessary cyclic inclusion dependency with:
// #include "perfgroup.h"
struct CounterList; // fwd declaration

// TODO: documentation of interfaces
// TODO: do we want "print_expression_tree"?

extern struct exptree_node *make_expression_tree(const char *expr);

extern void free_expression_tree(struct exptree_node *root);

extern double evaluate_expression_tree(const struct exptree_node *node, const struct CounterList *clist);

extern void print_expression_tree(const struct exptree_node *root);
