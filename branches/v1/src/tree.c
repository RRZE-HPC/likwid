/*
 * ===========================================================================
 *
 *      Filename:  tree.c
 *
 *      Description:  Module implementing a tree data structure
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


#include <stdlib.h>
#include <stdio.h>

#include <tree.h>


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
    if (nodePtr != NULL) 
    {
        printf("%d ", nodePtr->id);
        TreeNode* walker;    
        walker = nodePtr->llink;

        while (walker != NULL)
        {
            tree_print(walker);
            walker = walker->rlink;
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
        printf("ERROR: insertNode Node invalid\n");
        return;
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
        printf("ERROR: Node invalid\n");
        exit(0);
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
        printf("ERROR: Node invalid\n");
        exit(0);
    }
    if (nodePtr->llink == NULL)
    {
        printf("WARNING: Node has no children\n");
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
        printf("ERROR: getNode Node invalid\n");
        exit(0);
    }
    if (nodePtr->llink == NULL)
    {
        printf("WARNING: getNode Node has no children\n");
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
        printf("ERROR: getNode Node invalid\n");
        exit(0);
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
        printf("ERROR: getNode Node invalid\n");
        exit(0);
    }
    if (nodePtr->rlink == NULL)
    {
        return NULL;
    }

    return nodePtr->rlink;
}



