/*
 * =======================================================================================
 *
 *      Filename:  tree_types.h
 *
 *      Description:  Types file for tree module.
 *
 *      Version:   4.0
 *      Released:  16.6.2015
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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


/** \addtogroup CPUTopology
*  @{
*/
/*! \brief Structure of a tree node

This structure is used to form the tree of the system topology. The information
describing each node is store in other places, therefore an ID is enough.
\extends CpuTopology
*/
struct treeNode {
    int id; /*!< \brief ID of the node */
    struct treeNode* llink; /*!< \brief List of children of the current node */
    struct treeNode* rlink; /*!< \brief List of neighbors of the current node */
};

/** \brief Shorter name for struct treeNode */
typedef struct treeNode TreeNode;
/** @}*/

#endif /*TREE_TYPES_H*/
