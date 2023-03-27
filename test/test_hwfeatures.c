#include <stdlib.h>
#include <stdio.h>

#include <likwid.h>


int main(int argc, char* argv[])
{
	int err = 0;
	HWFeatureList list = {0, NULL};
	err = topology_init();
	if (err != 0)
	{
		printf("Error topology_init\n");
		return 1;
	}
	err = HPMinit();
	if (err != 0)
	{
		printf("Error HPMinit\n");
		return 1;
	}
	err = HPMaddThread(0);
	if (err != 0)
	{
		printf("Error HPMaddThread\n");
		return 1;
	}
	
	err = hwFeatures_init();
	if (err != 0)
	{
		printf("Error hwfeatures_init\n");
		return 1;
	}
	err = hwFeatures_list(&list);
	if (err != 0)
	{
		printf("Error hwfeatures_list\n");
		return 1;
	}
	printf("Feature list:\n");
	for (int i = 0; i < list.num_features; i++)
	{
	    printf("- %s.%s (Type: %s)\n", list.features[i].category, list.features[i].name, device_type_name(list.features[i].type));
	}
	printf("\n");
	LikwidDevice_t hwtdevice = NULL;
	err = likwid_device_create(DEVICE_TYPE_HWTHREAD, 0, &hwtdevice);
	if (err != 0)
	{
		printf("Error hwFeatures_create_device (hwt)\n");
		return 1;
	}
	LikwidDevice_t nodedevice = NULL;
	err = likwid_device_create(DEVICE_TYPE_NODE, 0, &nodedevice);
	if (err != 0)
	{
		printf("Error hwFeatures_create_device (node)\n");
		return 1;
	}
	LikwidDevice_t socketdevice = NULL;
	err = likwid_device_create(DEVICE_TYPE_SOCKET, 0, &socketdevice);
	if (err != 0)
	{
		printf("Error hwFeatures_create_device (socket)\n");
		return 1;
	}
	//perfmon_setVerbosity(3);
	for (int i = 0; i < list.num_features; i++)
	{
		char* val = NULL;
		if (list.features[i].type == DEVICE_TYPE_HWTHREAD)
		    hwFeatures_get(&list.features[i], hwtdevice, &val);
		else if (list.features[i].type == DEVICE_TYPE_NODE)
		    hwFeatures_get(&list.features[i], nodedevice, &val);
		else if (list.features[i].type == DEVICE_TYPE_SOCKET)
		    hwFeatures_get(&list.features[i], socketdevice, &val);
		printf("%s.%s : %s\n", list.features[i].category, list.features[i].name, val);
		/*uint64_t new = !val;
		uint64_t newread = 0;
		hwFeatures_modify(&list.features[i], 0, new);
		hwFeatures_get(&list.features[i], 0, &newread);
		printf("%s : %d\n", list.features[i].name, newread);
		hwFeatures_modify(&list.features[i], 0, val);*/

	}
	perfmon_setVerbosity(0);
	likwid_device_destroy(hwtdevice);
	likwid_device_destroy(nodedevice);
	likwid_device_destroy(socketdevice);
	hwFeatures_list_return(&list);
	hwFeatures_finalize();
	HPMfinalize();
	return 0;
}
