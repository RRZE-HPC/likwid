

#include <cpuid-hwloc.h>

#ifdef LIKWID_USE_HWLOC
int hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list)
{
    int i;
    int count = 0;
    hwloc_obj_t walker;
    if (!obj->arity) return 0;
    for (i=0;i<obj->arity;i++)
    {
        walker = obj->children[i];
        if (walker->type == type) 
        {
            if (list && *list && index)
            {
	            (*list)[(*index)++] = walker->logical_index;
	        }
            count++;
        }
        count += hwloc_record_objs_of_type_below_obj(t, walker, type, index, list);
    }
    return count;
}

#endif
