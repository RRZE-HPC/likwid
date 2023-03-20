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
	//perfmon_setVerbosity(3);
	for (int i = 0; i < list.num_features; i++)
	{
		uint64_t val = 0;
		hwFeatures_get(&list.features[i], 0, &val);
		printf("%s : %d\n", list.features[i].name, val);
		/*uint64_t new = !val;
		uint64_t newread = 0;
		hwFeatures_modify(&list.features[i], 0, new);
		hwFeatures_get(&list.features[i], 0, &newread);
		printf("%s : %d\n", list.features[i].name, newread);
		hwFeatures_modify(&list.features[i], 0, val);*/

	}
	//perfmon_setVerbosity(0);
	hwFeatures_list_return(&list);
	hwFeatures_finalize();
	HPMfinalize();
	return 0;
}
