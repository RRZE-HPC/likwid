/*
 * =======================================================================================
 *
 *      Filename:  tree_types.h
 *
 *      Description:  Types file for tree module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
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
