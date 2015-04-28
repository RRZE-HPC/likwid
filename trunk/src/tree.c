/*
 * =======================================================================================
 *
 *      Filename:  tree.c
 *
 *      Description:  Module implementing a tree data structure
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
tree_init(TreeNode** root, int id)
{
    *root = (TreeNode*) malloc(sizeof(TreeNode));
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

    TreeNode* digger;
    TreeNode* walker;

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

        TreeNode* digger;
        TreeNode* walker;
        TreeNode* tmp;

        digger = nodePtr->llink;

        while (digger != NULL)
        {
            walker = digger->rlink;

            while (walker != NULL)
            {
                tmp = walker;
                walker = walker->rlink;
                free(tmp);
            }
            tmp = digger;
            digger = digger->llink;
            free(tmp);
        }
    }
}

void
tree_insertNode(TreeNode* nodePtr, int id)
{
    TreeNode* currentNode;
    TreeNode* tmpNode;

    if (nodePtr == NULL)
    {
        ERROR_PLAIN_PRINT(Node invalid);
    }

    if (nodePtr->llink == NULL)
    {
        nodePtr->llink = (TreeNode*) malloc(sizeof(TreeNode));
        nodePtr->llink->id = id;
        nodePtr->llink->llink = NULL;
        nodePtr->llink->rlink = NULL;
    }
    else
    {
        currentNode = nodePtr->llink;

        while (currentNode->rlink != NULL)
        {
            if (id < currentNode->rlink->id)
            {
                tmpNode = currentNode->rlink;
                currentNode->rlink = (TreeNode*) malloc(sizeof(TreeNode));
                currentNode->rlink->id = id;
                currentNode->rlink->llink = NULL;
                currentNode->rlink->rlink = tmpNode;
                return;
            }
            currentNode = currentNode->rlink;
        }


        if (id > currentNode->id)
        {
            currentNode->rlink = (TreeNode*) malloc(sizeof(TreeNode));
            currentNode->rlink->id = id;
            currentNode->rlink->llink = NULL;
            currentNode->rlink->rlink = NULL;
        }
        else
        {
            tmpNode = currentNode;
            nodePtr->llink = (TreeNode*) malloc(sizeof(TreeNode));
            nodePtr->llink->id = id;
            nodePtr->llink->llink = NULL;
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



