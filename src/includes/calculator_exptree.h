// calculator_exptree.h
// TODO: file description

#ifndef CALCULATOR_EXPTREE_H_INCLUDED
#define CALCULATOR_EXPTREE_H_INCLUDED

/**
 * @brief Forward declaration of struct exptree_node
 * 
 */
struct exptree_node;

// cannot fwd declare CounterList because it was an anonymous struct (changed)
// thus we named CounterList to avoid unnecessary inclusion dependency with:
// #include "perfgroup.h"
/**
 * @brief Forward declaration of struct CounterList
 * 
 */
struct CounterList;

// TODO: do we want "print_expression_tree"?

/**
 * @brief Builds a binary expression tree from expression expr
 * 
 * @param expr The expression
 * @return struct exptree_node* Expression tree node on success, NULL on error
 */
extern struct exptree_node *make_expression_tree(const char *expr);

/**
 * @brief Deallocates the binary expression tree
 * 
 * @param root The root node
 */
extern void free_expression_tree(struct exptree_node *root);

/**
 * @brief Evaluates the expression tree with values from counter list
 * 
 * @param root The root node
 * @param clist The counter list
 * @return double The value of the expression tree evaluation on success, NAN on error
 */
extern double evaluate_expression_tree(const struct exptree_node *root, const struct CounterList *clist);

/**
 * @brief Prints the expression tree in infix order
 * 
 * @param root The root node
 */
extern void print_expression_tree(const struct exptree_node *root);

#endif /* CALCULATOR_EXPTREE_H_INCLUDED */
