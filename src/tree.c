/*
 * =======================================================================================
 *
 *      Filename:  tree.c
 *
 *      Description:  Module implementing a tree data structure
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>

#include <error.h>
#include <tree.h>

/* #####   FUNCTION DEFINITIONS  -  INTERNAL FUNCTIONS   ################## */

void _tree_destroy(TreeNode* nodePtr)
{
    if (nodePtr == NULL)
        return;
    if (nodePtr->rlink)
    {
        _tree_destroy(nodePtr->rlink);
        free(nodePtr->rlink);
    }
    if (nodePtr->llink)
    {
        _tree_destroy(nodePtr->llink);
        free(nodePtr->llink);
    }
    return;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
tree_init(TreeNode** root, int id)
{
    *root = (TreeNode*) malloc(sizeof(TreeNode));
    if (!(*root))
    {
        *root = NULL;
        return;
    }
    (*root)->id = id;
    (*root)->llink = NULL;
    (*root)->rlink = NULL;
}

void
tree_print(TreeNode* nodePtr)
{
  int level = 0;

  if (nodePtr != NULL)
  {

    TreeNode* digger = NULL;
    TreeNode* walker = NULL;

    digger = nodePtr->llink;

    while (digger != NULL)
    {
      printf("\n Level %d:\n", level++);
      printf("%d ", digger->id);
      walker = digger->rlink;

      while (walker != NULL)
      {
        printf("%d ", walker->id);
        walker = walker->rlink;
      }

      digger = digger->llink;
    }

    printf("\n ");
  }
}

void
tree_destroy(TreeNode* nodePtr)
{

    if (nodePtr != NULL)
    {
        _tree_destroy(nodePtr);
        free(nodePtr);
    }
}

void
tree_insertNode(TreeNode* nodePtr, int id)
{
    TreeNode* currentNode = NULL;
    TreeNode* tmpNode = NULL;
    TreeNode* newNode = NULL;

    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
    }

    newNode = (TreeNode*) malloc(sizeof(TreeNode));
    if (!newNode)
    {
        return;
    }
    newNode->id = id;
    newNode->llink = NULL;
    newNode->rlink = NULL;

    if (nodePtr->llink == NULL)
    {
        nodePtr->llink = newNode;
    }
    else
    {
        currentNode = nodePtr->llink;

        while (currentNode->rlink != NULL)
        {
            if (id < currentNode->rlink->id)
            {
                tmpNode = currentNode->rlink;
                currentNode->rlink = newNode;
                currentNode->rlink->rlink = tmpNode;
                return;
            }
            currentNode = currentNode->rlink;
        }

        if (id > currentNode->id)
        {
            currentNode->rlink = newNode;
        }
        else
        {
            tmpNode = currentNode;
            nodePtr->llink = newNode;
            nodePtr->llink->rlink = tmpNode;
        }
    }
}

int
tree_nodeExists(TreeNode* nodePtr, int id)
{
    TreeNode* walker;

    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
        return 0;
    }

    walker = nodePtr->llink;

    while (walker != NULL)
    {
        if (walker->id == id)
        {
            return 1;
        }
        else
        {
            walker = walker->rlink;
        }
    }

    return 0;
}

int
tree_countChildren(TreeNode* nodePtr)
{
    TreeNode* walker;
    int count=0;

    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
        return 0;
    }
    if (nodePtr->llink == NULL)
    {
        return 0;
    }

    walker = nodePtr->llink;

    while (walker != NULL)
    {
        count++;
        walker = walker->rlink;
    }

    return count;
}

TreeNode*
tree_getNode(TreeNode* nodePtr, int id)
{
    TreeNode* walker;

    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
        return NULL;
    }
    if (nodePtr->llink == NULL)
    {
        return NULL;
    }

    walker = nodePtr->llink;

    while (walker != NULL)
    {
        if (walker->id == id)
        {
            return walker;
        }
        else
        {
            walker = walker->rlink;
        }
    }

    return NULL;
}

TreeNode*
tree_getChildNode(TreeNode* nodePtr)
{
    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
        return NULL;
    }
    if (nodePtr->llink == NULL)
    {
        return NULL;
    }

    return nodePtr->llink;
}

TreeNode*
tree_getNextNode(TreeNode* nodePtr)
{
    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
    }

    if (nodePtr->rlink == NULL)
    {
        return NULL;
    }

    return nodePtr->rlink;
}

