/*
 * ===========================================================================
 *
 *      Filename:  tree.h
 *
 *      Description:  Header File tree Module. 
 *                    Implements a simple tree data structure.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */


#ifndef TREE_H
#define TREE_H

#include <types.h>

extern void tree_init(TreeNode** root, int id);
extern void tree_print(TreeNode* nodePtr);
extern void tree_insertNode(TreeNode* nodePtr, int id);
extern int tree_nodeExists(TreeNode* nodePtr, int id);
extern int tree_countChildren(TreeNode* nodePtr);
extern void tree_sort(TreeNode* root);
extern TreeNode* tree_getNode(TreeNode* nodePtr, int id);
extern TreeNode* tree_getChildNode(TreeNode* nodePtr);
extern TreeNode* tree_getNextNode(TreeNode* nodePtr);

#endif /*TREE_H*/
