/*
 * ===========================================================================
 *
 *      Filename:  tree_types.h
 *
 *      Description:  Types file for tree module.
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


#ifndef TREE_TYPES_H
#define TREE_TYPES_H

/* For arbitrary trees llink are the children and
 * rlink are the neighbours
 */
typedef struct treeNode {
    int id;
    struct treeNode* llink;
    struct treeNode* rlink;
} TreeNode;


#endif /*TREE_TYPES_H*/
