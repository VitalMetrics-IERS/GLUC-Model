import collections
import logging
import os
import warnings
import numpy as np
from osgeo import gdal
import scipy
import hazelbean as hb
import hazelbean.pyramids
from hazelbean.ui import model, inputs

from collections import OrderedDict

import seals_utils

logging.basicConfig(level=logging.WARNING)
hb.ui.model.LOGGER.setLevel(logging.WARNING)
hb.ui.inputs.LOGGER.setLevel(logging.WARNING)

L = hb.get_logger('seals', logging_level='warning')
L.setLevel(logging.INFO)

logging.getLogger('Fiona').setLevel(logging.WARNING)
logging.getLogger('fiona.collection').setLevel(logging.WARNING)

np.seterr(divide='ignore', invalid='ignore')

p = hb.ProjectFlow()

# TASKS
def generate_batch_zones():
    global p
    p.layers_to_stitch = []

    if p.enable_batch_mode:
        if not p.use_existing_batch:
            hb.convert_shapefile_to_multiple_shapefiles_by_id(p.area_of_interest_path, p.batch_id, p.cur_dir)


        # HACK HERE
        list_of_aoi_paths = hb.list_filtered_paths_nonrecursively(p.cur_dir, include_extensions='.shp')
        # list_of_aoi_paths = hb.list_filtered_paths_nonrecursively(p.cur_dir, include_extensions='.shp', include_strings=['aoi_10.shp', 'aoi_11.shp'])

        # This is the one part of the model code that needs to be aware of the structure of project_flow
        # When a batch is designed, it first does something (like creating the shapefiles above), and then it defines
        # p.iterator_replacements, which is a dict of name, value pairs that will modify the p object when the iterated task is run
        # according to it's position among the iterating tasks.
        p.iterator_replacements = collections.OrderedDict()

        # Simple replacement of the aoi to use
        p.iterator_replacements['area_of_interest_path'] = list_of_aoi_paths

        # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location
        p.iterator_replacements['cur_dir_parent_dir'] = [os.path.splitext(i)[0] for i in list_of_aoi_paths]
    else:
        print ('Not running in batch mode.')

        # Because the seals model is enherently batchable, i chose to still define a single iteration into replacements:
        p.iterator_replacements = collections.OrderedDict()
        p.iterator_replacements['area_of_interest_path'] = [p.area_of_interest_path]
        p.iterator_replacements['cur_dir_parent_dir'] = [hb.file_root(p.area_of_interest_path)]

def process_coarse_change_maps():
    global p
    L.info('process_coarse_change_maps.')

    # Change maps are in this directory and must be of the format [CLASS_ID_INT]_[someting, but anything else].tif
    if not os.path.isdir(p.coarse_change_maps_dir):
        p.coarse_change_maps_dir = os.path.split(p.coarse_change_maps_dir)[0]
        if not os.path.isdir(p.coarse_change_maps_dir):
            raise NameError('Unable to parse coarse_change_maps_dir.')
    tifs_in_dir = hb.list_filtered_paths_nonrecursively(p.coarse_change_maps_dir, include_extensions='.tif')

    p.change_map_paths = []
    for path in tifs_in_dir:
        try:
            rendered_int = int(hb.file_root(path).split('_')[0])
        except:
            rendered_int = None
        if isinstance(rendered_int, int):
            p.change_map_paths.append(path)
    p.change_map_raster_infos = [hb.get_raster_info(i) for i in p.change_map_paths]

    # Test that all the change maps are the same properties.
    if len(set([i['geotransform'] for i in p.change_map_raster_infos])) != 1:
        for j in [i['geotransform'] for i in p.change_map_raster_infos]:
            L.critical('geotransform: ' + str(j))
        # raise NameError('The maps in coarse change maps dir are not all the same shape, projection, etc, or they have been improperly named/formatted.')

    # p.current_change_in_crop_extent_path = os.path.join(p.cur_dir, 'change_in_crop_extent.tif')
    p.current_change_map_paths = []
    p.float_ndv = None
    p.int_ndv = 255

    L.info('change_map_paths: ' + str(p.change_map_paths))
    p.zone_transition_sums = OrderedDict()
    p.classes_projected_to_change = []
    for path in p.change_map_paths:
        changing_class_id = int(os.path.split(path)[1].split('_')[0])
        p.classes_projected_to_change.append(changing_class_id)

        if not p.float_ndv:
            p.float_ndv = hb.get_nodata_from_uri(path)
            if p.float_ndv is None:
                p.float_ndv = -9999.0

        new_path = os.path.join(p.cur_dir, os.path.split(path)[1])
        p.current_change_map_paths.append(new_path)
        if p.run_this:  # NOTE NONSTANDARD placement of run_this
            hb.clip_raster_by_vector(str(path), str(new_path), str(p.area_of_interest_path),
                                     resample_method='nearest',
                                     all_touched=True, verbose=True,
                                     ensure_fits=True, gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

            # To make the model not run in zones with zero change, we collect these sums and prevent runing if all of them are zero
            current_coarse_array = hb.as_array(new_path)
            current_sum = np.sum(current_coarse_array[current_coarse_array != p.float_ndv])
            p.zone_transition_sums[changing_class_id] = current_sum
    p.run_this_zone = True
    if np.sum([float(i) for i in p.zone_transition_sums.values()]) <= 0:
        p.run_this_zone = False

    L.info('current_change_map_paths' + str(p.current_change_map_paths))

def create_lulc():
    global p
    L.info('Creating class-types lulc.')

    p.name_from_iterator_replacements = hb.file_root(p.area_of_interest_path)
    p.base_year_current_zone_lulc_path = os.path.join(p.cur_dir, 'base_year_' + p.name_from_iterator_replacements + '.tif')
    # Create match paths of both data types
    p.match_int_path = p.base_year_current_zone_lulc_path


    p.lulc_simplified_path = os.path.join(p.cur_dir, 'lulc_simplified.tif')
    # p.lulc_simplified_path = p.base_year_current_zone_lulc_path
    p.valid_mask_path = os.path.join(p.cur_dir, 'valid_mask_high_res.tif')

    p.proportion_valid_fine_per_coarse_cell_path = os.path.join(p.cur_dir, 'proportion_valid_fine_per_coarse_cell.tif')

    if p.run_this:

        hb.clip_while_aligning_to_coarser(p.base_year_lulc_path, p.base_year_current_zone_lulc_path, p.area_of_interest_path,
                                          p.current_change_map_paths[0], resample_method='nearest',
                                          output_data_type=1, nodata_target=255,
                                          all_touched=True, verbose=True,
                                          ensure_fits=True, gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

        # Set NDV masking based on AOI of current zone.

        hb.create_valid_mask_from_vector_path(p.area_of_interest_path, p.base_year_current_zone_lulc_path, p.valid_mask_path)
        p.valid_mask = hb.as_array(p.valid_mask_path)
        hb.set_ndv_by_mask_path(p.base_year_current_zone_lulc_path, p.valid_mask_path)

        p.proportion_valid_fine_per_coarse_cell = hazelbean.pyramids.calc_proportion_of_coarse_res_with_valid_fine_res(p.current_change_map_paths[0], p.valid_mask_path)
        hb.save_array_as_geotiff(p.proportion_valid_fine_per_coarse_cell, p.proportion_valid_fine_per_coarse_cell_path, p.current_change_map_paths[0])

        lulc_ds = gdal.Open(p.base_year_current_zone_lulc_path)
        lulc_band = lulc_ds.GetRasterBand(1)
        lulc_array = lulc_band.ReadAsArray().astype(np.int)

        p.scaled_proportion_to_allocate_paths = []
        for path in p.current_change_map_paths:
            unscaled = hb.as_array(path).astype(np.float64)
            scaled_proportion_to_allocate = p.proportion_valid_fine_per_coarse_cell * unscaled

            scaled_proportion_to_allocate_path = os.path.join(p.cur_dir, os.path.split(hb.suri(path, 'scaled'))[1])

            hb.save_array_as_geotiff(scaled_proportion_to_allocate, scaled_proportion_to_allocate_path, path, ndv=-9999.0, data_type=7)

            p.scaled_proportion_to_allocate_paths.append(scaled_proportion_to_allocate_path)

        if os.path.exists(p.lulc_class_types_path):
            # load the simplified class correspondnce as a nested dictionary.
            lulc_class_types_odict = hb.file_to_python_object(p.lulc_class_types_path, declare_type='DD')

            # For cythonization reasons, I need to ensure this comes in as ints
            lulc_class_types_ints_dict = dict()

            p.lulc_unsimplified_classes_list = []
            for row_name in lulc_class_types_odict.keys():
                lulc_class_types_ints_dict[int(row_name)] = int(lulc_class_types_odict[row_name]['lulc_class_type'])
                p.lulc_unsimplified_classes_list.append(int(row_name))

            p.max_unsimplified_lulc_classes = max(p.lulc_unsimplified_classes_list)
            p.new_unsimplified_lulc_addition_value = 10 ** (len(str(p.max_unsimplified_lulc_classes)) + 1) / 10  # DOCUMENTATION, new classes are defined here as adding 1 order



            # # 1 is agriculture, 2 is mixed ag/natural, 3 is natural, 4 is urban, 0 is no data
            lulc_simplified_array = hb.reclassify_int_array_by_dict_to_ints(lulc_array, lulc_class_types_ints_dict)
            no_data_value_override = hb.get_nodata_from_uri(p.base_year_current_zone_lulc_path)
            hb.save_array_as_geotiff(lulc_simplified_array, p.lulc_simplified_path, p.base_year_current_zone_lulc_path, data_type=1, set_inf_to_no_data_value=False, ndv=no_data_value_override, compress=True)
        else:
            L.warn('No lulc_class_types_path specified. Assuming you want to run every class uniquely.')

        # If we don't run this zone, we know we will need to use the unmodified lulc when stitching everything back together
        if p.run_this_zone is False:
            p.layers_to_stitch.append(p.base_year_current_zone_lulc_path)

    else:
        p.lulc_simplified_path = p.base_year_current_zone_lulc_path



def create_physical_suitability():
    global p
    L.info('Creating physical suitability layer from base data.')
    #  physical suitability calculations, though for speed it's included as a base datum.
    dem_unaligned_path = hb.temp('.tif', folder=p.workspace_dir, remove_at_exit=True) #hb.temp('.tif', remove_at_exit=True)
    stats_to_calculate = ['TRI']
    hb.clip_hydrosheds_dem_from_aoi(dem_unaligned_path, p.area_of_interest_path, p.match_float_path)
    hb.calculate_topographic_stats_from_dem(dem_unaligned_path, p.physical_suitability_dir, stats_to_calculate=stats_to_calculate, output_suffix='unaligned')
    dem_path = os.path.join(p.physical_suitability_dir, 'dem.tif')
    hb.align_dataset_to_match(dem_unaligned_path, p.match_float_path, dem_path, aoi_uri=p.area_of_interest_path)
    for stat in stats_to_calculate:
        stat_unaligned_path = os.path.join(p.physical_suitability_dir, stat + '_unaligned.tif')
        hb.delete_path_at_exit(stat_unaligned_path)
        stat_path = os.path.join(p.physical_suitability_dir, stat + '.tif')
        hb.align_dataset_to_match(stat_unaligned_path, p.match_float_path, stat_path, resample_method='bilinear',
                                  align_to_match=True, aoi_uri=p.area_of_interest_path)
    soc_path = os.path.join(p.physical_suitability_dir, 'soc.tif')
    hb.align_dataset_to_match(p.base_data_soc_path, p.match_int_path, soc_path, aoi_uri=p.area_of_interest_path, output_data_type=7)
    tri_path = os.path.join(p.physical_suitability_dir, 'tri.tif')
    hb.align_dataset_to_match(p.base_data_tri_path, p.match_int_path, tri_path, aoi_uri=p.area_of_interest_path, output_data_type=7)
    # TODOO Create cythonized array_sum_product()
    p.physical_suitability_path = os.path.join(p.physical_suitability_dir, 'physical_suitability.tif')
    soc_array = hb.as_array(soc_path)
    tri_array = hb.as_array(tri_path)
    physical_suitability_array = np.log(soc_array) - np.log(tri_array)

    # p.global_physical_suitability_path = os.path.join(p.model_base_data_dir, 'physical_suitability_compressed.tif')
    p.clipped_physical_suitability_path = os.path.join(p.cur_dir, 'physical_suitability.tif')

    if p.run_this and p.run_this_zone:
        # hb.clip_raster_by_vector(p.global_physical_suitability_path, p.physical_suitability_path, p.coarse_res_aoi_path, all_touched=True)
        hb.clip_while_aligning_to_coarser(p.physical_suitability_path, p.clipped_physical_suitability_path, p.area_of_interest_path,
                                          p.current_change_map_paths[0], resample_method='nearest',
                                          all_touched=True, verbose=True,
                                          ensure_fits=True, gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

        p.current_physical_suitability_path = p.clipped_physical_suitability_path # NOTE awkward naming
        # hb.clip_dataset_uri(p.global_physical_suitability_path, p.coarse_res_aoi_path, p.physical_suitability_path, False, all_touched=False)
        physical_suitability_array = hb.as_array(p.current_physical_suitability_path)
        p.match_float_path = p.current_physical_suitability_path

        np.seterr(divide='ignore', invalid='ignore')
        physical_suitability_array = np.where(physical_suitability_array > -1000, physical_suitability_array, 0)
        physical_suitability_array = np.where(physical_suitability_array < 100000000, physical_suitability_array, 0)

        hb.save_array_as_geotiff(physical_suitability_array, p.current_physical_suitability_path, p.match_float_path, compress=True)

def create_convolution_inputs():
    global p
    p.convolution_inputs_dir = p.cur_dir
    if p.run_this and p.run_this_zone:
        lulc_array = hb.as_array(p.lulc_simplified_path)

        ndv = hb.get_nodata_from_uri(p.lulc_simplified_path)

        # Get which values exist in simplified_lulc
        unique_values = list(hb.enumerate_array_as_odict(lulc_array).keys())
        unique_values = [int(i) for i in unique_values]

        try:
            p.classes_to_ignore = [int(i) for i in p.classes_to_ignore.split(' ')]
        except:
            p.classes_to_ignore = []
        # TODOO Better approach than ignoring classes would be to encode ALL such information into the different CSVs. This would allow more grandular control over how, e.g. water DOES have attraction effect but does not necessarily expand.
        ignore_values = [ndv] + p.classes_to_ignore
        p.simplified_lulc_classes = [i for i in unique_values if i not in ignore_values]

        # HACK
        p.classes_to_ignore = [0]

        p.classes_with_effect = [i for i in p.simplified_lulc_classes if i not in p.classes_to_ignore]
        L.info('Creating binaries for classes ' + str(p.classes_with_effect))

        try:
            p.max_simplified_lulc_classes = max(p.simplified_lulc_classes)
        except:
            p.max_simplified_lulc_classes = 20
        p.new_simplified_lulc_addition_value = 10 ** (len(str(
            p.max_simplified_lulc_classes)) + 1) / 10  # DOCUMENTATION, new classes are defined here as adding 1 order of magnitude larger value (2 becomes 12 if the max is 5. 2 becomes 102 if the max is 15.

        p.classes_with_change = [int(os.path.split(i)[1].split('_')[0]) for i in p.current_change_map_paths]

        binary_paths = []
        for unique_value in p.classes_with_effect:
            # binary_array = np.zeros(lulc_array.shape)
            binary_array = np.where(lulc_array == unique_value, 1, 0).astype(np.uint8)
            binary_path = os.path.join(p.convolution_inputs_dir, 'class_' + str(unique_value) + '_binary.tif')
            binary_paths.append(binary_path)
            hb.save_array_as_geotiff(binary_array, binary_path, p.lulc_simplified_path, compress=True)

        convolution_params = hb.file_to_python_object(p.class_proximity_parameters_path, declare_type='DD', output_key_data_type=str, output_value_data_type=float)
        convolution_paths = []

        for i, v in enumerate(p.classes_with_effect):
            L.info('Calculating convolution for class ' + str(v))
            binary_array = hb.as_array(binary_paths[i])
            convolution_metric = seals_utils.distance_from_blurred_threshold(binary_array, convolution_params[str(v)]['clustering'], 0.5, convolution_params[str(v)]['decay'])
            convolution_path = os.path.join(p.convolution_inputs_dir, 'class_' + str(p.classes_with_effect[i]) + '_convolution.tif')
            convolution_paths.append(convolution_path)
            hb.save_array_as_geotiff(convolution_metric, convolution_path, p.match_float_path, compress=True)

        pairwise_params = hb.file_to_python_object(p.pairwise_class_relationships_path, declare_type='DD', output_key_data_type=str, output_value_data_type=float)
        for i in p.classes_with_effect:
            i_convolution_path = os.path.join(p.convolution_inputs_dir, 'class_' + str(i) + '_convolution.tif')
            i_convolution_array = hb.as_array(i_convolution_path)
            for j in p.classes_with_change:
                L.info('Processing effect of ' + str(i) + ' on ' + str(j))
                adjacency_effect_path = os.path.join(p.convolution_inputs_dir, 'adjacency_effect_of_' + str(i) + '_on_' + str(j) + '.tif')
                adjacency_effect_array = i_convolution_array * pairwise_params[str(i)][str(j)]
                hb.save_array_as_geotiff(adjacency_effect_array, adjacency_effect_path, p.match_float_path, compress=True)

        for i in p.classes_with_change:
            L.info('Combining adjacency effects for class ' + str(i))
            combined_adjacency_effect_array = np.ones(lulc_array.shape)
            combined_adjacency_effect_path = os.path.join(p.convolution_inputs_dir, 'combined_adjacency_effect_' + str(i) + '.tif')
            for j in p.classes_with_effect:
                current_uri = os.path.join(p.convolution_inputs_dir, 'adjacency_effect_of_' + str(j) + '_on_' + str(i) + '.tif')  # NOTICE SWITCHED I and J
                current_effect = hb.as_array(current_uri)
                combined_adjacency_effect_array *= current_effect + 1.0  # Center on 1 so that 0.0 has no effect
            hb.save_array_as_geotiff(combined_adjacency_effect_array, combined_adjacency_effect_path, p.match_float_path, compress=True)

def create_conversion_eligibility():
    global p
    p.conversion_eligibility_dir = p.cur_dir

    if p.run_this and p.run_this_zone:
        # Prevent illogical conversion eg new ag onto existing ag, or new ag onto urban
        conversion_eligibility_params = hb.file_to_python_object(p.conversion_eligibility_path, declare_type='DD', output_key_data_type=str, output_value_data_type=int)
        simplified_lulc_array = hb.as_array(p.lulc_simplified_path)
        for i in p.classes_with_change:

            conversion_eligibility_raster_path = os.path.join(p.conversion_eligibility_dir, str(i) + '_conversion_eligibility.tif')
            conversion_eligibility_array = np.zeros(simplified_lulc_array.shape).astype(np.float64)
            for j in p.classes_with_effect:
                conversion_eligibility_array = np.where(simplified_lulc_array == j, conversion_eligibility_params[str(j)][str(i)], conversion_eligibility_array)
            hb.save_array_as_geotiff(conversion_eligibility_array, conversion_eligibility_raster_path, p.match_int_path, compress=True)

def create_overall_suitability():
    global p
    p.overall_suitability_dir = p.cur_dir
    hb.create_directories(p.overall_suitability_dir)
    p.overall_suitability_paths = []

    if p.run_this and p.run_this_zone:
        # NOTE, here the methods assume ONLY crop will be changing insofar as the physical suitability is defined wrt crops; 0 is 1 becasue already  got rid of 0 in unique values
        physical_suitability_array = hb.as_array(p.current_physical_suitability_path)

        for i in p.classes_with_change:
            suitability_path = hb.ruri(os.path.join(p.overall_suitability_dir, 'overall_suitability_' + str(i) + '.tif'))
            p.overall_suitability_paths.append(suitability_path)
            combined_adjacency_effect_path = os.path.join(p.convolution_inputs_dir, 'combined_adjacency_effect_' + str(i) + '.tif')

            adjacency_effect_array = hb.as_array(combined_adjacency_effect_path)
            adjacency_effect_array = seals_utils.normalize_array(adjacency_effect_array)  # Didn't put this in HB because didn't want to redo the 0.4.0 release.
            conversion_eligibility_raster_path = os.path.join(p.create_conversion_eligibility_dir, str(i) + '_conversion_eligibility.tif')
            conversion_eligibility_array = hb.as_array(conversion_eligibility_raster_path)
            try:
                physical_suitability_importance = float(p.physical_suitability_importance)
            except:
                physical_suitability_importance = 0.5
                L.warning('Could not interpret physical suitability importance. Using default of 0.5')

            physical_suitability_array = seals_utils.normalize_array(physical_suitability_array)

            overall_suitability_array = (adjacency_effect_array + (physical_suitability_importance * physical_suitability_array)) * conversion_eligibility_array
            overall_suitability_array = np.where(np.isnan(overall_suitability_array), 0, overall_suitability_array)
            overall_suitability_array = np.where(overall_suitability_array < 0, 0, overall_suitability_array)
            hb.save_array_as_geotiff(overall_suitability_array, suitability_path, p.match_float_path, compress=True)

def create_allocation_from_change_map():
    global p
    p.projected_lulc_simplified_path = hb.ruri(os.path.join(p.cur_dir, 'projected_lulc_simplified.tif'))

    # AGROSERVE shortcut note: assumed that it happens in SEQUENCE first cropland then pasture.
    if p.run_this and p.run_this_zone:

        lulc_array = hb.as_array(p.lulc_simplified_path)
        new_lulc_array = np.copy(lulc_array)

        p.change_array_paths = []
        for change_map_index, change_map_path in enumerate(p.scaled_proportion_to_allocate_paths):

            change_to_allocate_array = hb.as_array(change_map_path)

            # Often it is the case that the number of cells that will be allocated is greater than the amount of high-res cells actually available for conversion. This happens only if the
            # conversion_elligibility.csv rules out cells (it will not happen if only adjacency and physical suitability is done, as there will be SOME places allbethem terrible.
            num_cells_skipped = np.zeros(change_to_allocate_array.shape)

            class_to_allocate = int(os.path.split(change_map_path)[1].split('_')[0])
            current_overall_suitability_path = p.overall_suitability_paths[change_map_index]
            overall_suitability_array = hb.as_array(current_overall_suitability_path)

            # Test that map resolutions are workable multiples of each other
            aspect_ratio_test_result = int(round(overall_suitability_array.shape[0] / change_to_allocate_array.shape[0])) == int(
                round(overall_suitability_array.shape[1] / change_to_allocate_array.shape[1]))

            if not aspect_ratio_test_result:
                warnings.warn('aspect_ratio_test_value FAILED.')
            aspect_ratio = int(round(overall_suitability_array.shape[0] / change_to_allocate_array.shape[0]))

            L.info('Beginning allocation using allocation ratio of ' + str(aspect_ratio))
            L.info('Sizes involved: overall_suitability_array, ' + str(overall_suitability_array.shape) + ' change_to_allocate_array, ' + str(change_to_allocate_array.shape))

            ha_per_source_cell = 300 ** 2 / 100 ** 2
            change_array = np.zeros(lulc_array.shape)
            combined_rank_array = np.zeros(lulc_array.shape).astype(np.int64)

            # TODOO Note that i ignored smaller-than-chunk shards.
            for change_map_region_row in range(change_to_allocate_array.shape[0]):
                L.info('Starting horizontal row ' + str(change_map_region_row))
                for change_map_region_col in range(change_to_allocate_array.shape[1]):
                    if not change_to_allocate_array[change_map_region_row, change_map_region_col] > 0:
                        num_cells_to_allocate = 0
                    else:
                        num_cells_to_allocate = int(round(change_to_allocate_array[change_map_region_row, change_map_region_col] / ha_per_source_cell))

                    if num_cells_to_allocate > 0:
                        source_map_starting_row = change_map_region_row * aspect_ratio
                        source_map_starting_col = change_map_region_col * aspect_ratio
                        combined_adjacency_effect_chunk = overall_suitability_array[source_map_starting_row: source_map_starting_row + aspect_ratio,
                                                          source_map_starting_col: source_map_starting_col + aspect_ratio]

                        ranked_chunk, sorted_keys = hb.get_rank_array_and_keys(combined_adjacency_effect_chunk, ndv=0)

                        if num_cells_to_allocate > len(sorted_keys[0]):
                            previous_num_cells_to_allocate = num_cells_to_allocate
                            num_skipped = num_cells_to_allocate - len(sorted_keys[0])
                            num_cells_to_allocate = len(sorted_keys[0])
                            L.warning(
                                'Allocation algorithm requested to allocate more cells than were available for transition given the suitability constraints. Num requested: ' + str(
                                    previous_num_cells_to_allocate) + ', Num allocated: ' + str(len(sorted_keys[0])) + ', Num skipped ' + str(num_skipped))
                            num_cells_skipped[change_map_region_row, change_map_region_col] = num_skipped

                        sorted_keys_array = np.array(sorted_keys)

                        # Create a tuple (ready for use as a numpy key) of the top allocation_amoutn keys
                        keys_to_change = (sorted_keys_array[0][0:num_cells_to_allocate], sorted_keys_array[1][0:num_cells_to_allocate])

                        change_chunk = np.zeros(ranked_chunk.shape)
                        change_chunk[keys_to_change] = 1

                        ## TODOO this was useful but there was a 29x29 vs 30x30 error. Renable after fix.
                        # Just for visualization purposes, who what all the ranked zones look like together when mosaiced.
                        combined_rank_array[source_map_starting_row: source_map_starting_row + aspect_ratio,
                        source_map_starting_col: source_map_starting_col + aspect_ratio] = ranked_chunk

                        # TODOO BUG, there's a slight shift to the right that comes in here.
                        change_array[source_map_starting_row: source_map_starting_row + aspect_ratio,
                        source_map_starting_col: source_map_starting_col + aspect_ratio] = change_chunk

            L.info('Processing outputted results.')
            p.new_classes_int_list = [13]
            p.final_lulc_addition_value = 13
            new_lulc_array = np.where((change_array == 1), p.final_lulc_addition_value, new_lulc_array)  # NOTE, pasture will be 8 thus, crops 9

            change_array_path = os.path.join(p.cur_dir, str(class_to_allocate) + '_change_array.tif')
            p.change_array_paths.append(change_array_path)
            hb.save_array_as_geotiff(change_array, change_array_path, p.match_int_path, compress=True)

            p.num_cells_skipped_path = hb.ruri(os.path.join(p.cur_dir, str(class_to_allocate) + '_num_cells_skipped.tif'))
            hb.save_array_as_geotiff(num_cells_skipped, p.num_cells_skipped_path, change_map_path, compress=True)

            p.combined_rank_array_path = hb.ruri(os.path.join(p.cur_dir, str(class_to_allocate) + '_combined_rank_array.tif'))
            hb.save_array_as_geotiff(combined_rank_array, p.combined_rank_array_path, p.match_int_path, compress=True, data_type=7)

        hb.save_array_as_geotiff(new_lulc_array, p.projected_lulc_simplified_path, p.match_int_path, compress=True)

def convert_simplified_to_original_classes():
    global p
    lulc_class_types_odict = hb.file_to_python_object(p.lulc_class_types_path, declare_type='DD')
    p.simple_classes_to_projected_original_classes = OrderedDict()
    for original_class, csv_odict in lulc_class_types_odict.items():
        if csv_odict['output_class_id'] != '':
            p.simple_classes_to_projected_original_classes[int(csv_odict['lulc_class_type'])] = int(csv_odict['output_class_id'])

    if p.run_this and p.run_this_zone:

        lulc_original_classes_array = hb.as_array(p.base_year_current_zone_lulc_path)

        for c, path in enumerate(p.change_array_paths):
            change_array = hb.as_array(path)
            change_array_ndv = hb.get_nodata_from_uri(path)


            lulc_projected_original_classes_array = np.where((change_array > 0) & (change_array != change_array_ndv), p.simple_classes_to_projected_original_classes[p.classes_projected_to_change[c]], lulc_original_classes_array)


        p.lulc_projected_original_classes_path = os.path.join(p.cur_dir, 'lulc_projected_original_classes.tif')
        hb.save_array_as_geotiff(lulc_projected_original_classes_array, p.lulc_projected_original_classes_path, p.match_int_path)
        p.layers_to_stitch.append(p.lulc_projected_original_classes_path)

    # ALSO NOTE that we only return this once, because separate batched tasks are appending to it
    return ('layers_to_stitch', 'append_to_list', p.layers_to_stitch) # WARNING the only intended use of returns in a tasks is if its a return resource to be synced among parallel tasks.

def clean_intermediate_files():
    global p
    hb.remove_path(p.generate_batch_zones_dir)

def stitch_projections():
    global p
    if p.run_this:

        scenario_name = os.path.split(p.workspace_dir)[1]
        p.projected_lulc_stitched_path = hb.ruri(os.path.join(p.cur_dir, 'projected_lulc.tif'))
        p.projected_lulc_stitched_merged_path = hb.ruri(os.path.join(p.cur_dir, 'projected_lulc_merged.tif'))
        p.original_lulc_stitched_path = hb.ruri(os.path.join(p.cur_dir, 'original_lulc.tif'))

        do_global_stitch = True
        if p.output_base_map_path and len(p.layers_to_stitch) > 0 and do_global_stitch:
            L.info('Stamping generated lulcs with extent_shift_match_path of output_base_map_path ' + str(p.output_base_map_path))
            ndv = hb.get_datatype_from_uri(p.output_base_map_path)
            hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.projected_lulc_stitched_path, p.output_base_map_path, dstnodata=255)

            base_raster_path_band_list = [(p.projected_lulc_stitched_path, 1),
                                          (p.output_base_map_path, 1)]

            def fill_where_missing(a, b):
                # NOTE == not work here because a.any() or a.all() error. Crappy workaround is inequalities.
                return np.where((a >= 255) & (b <= 255), b, a)

            # Because SEALS doesn't run for small islands, we fill in any missing values based on the base data input lulc.

            datatype_target = 1
            nodata_target = 255
            opts = ['TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=lzw']
            hb.raster_calculator(base_raster_path_band_list, fill_where_missing, p.projected_lulc_stitched_merged_path,
                                 datatype_target, nodata_target, gtiff_creation_options=opts)

            try:
                import geoecon as ge
                ge.add_geotiff_overview_file(p.projected_lulc_stitched_merged_path)
            except:
                pass
        else:
            L.info('Stitching together all of the generated LULCs.')
            if len(p.layers_to_stitch) > 0:
                hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.projected_lulc_stitched_path, dstnodata=255)



define_zones_iterator = p.add_iterator(generate_batch_zones)

process_coarse_change_maps_task = p.add_task(process_coarse_change_maps, parent=define_zones_iterator)
create_lulc_task = p.add_task(create_lulc, parent=define_zones_iterator)
create_physical_suitability_task = p.add_task(create_physical_suitability, parent=define_zones_iterator)
create_convolution_inputs_task = p.add_task(create_convolution_inputs, parent=define_zones_iterator)
create_conversion_eligibility_task = p.add_task(create_conversion_eligibility, parent=define_zones_iterator)
create_overall_suitability_task = p.add_task(create_overall_suitability, parent=define_zones_iterator)
create_allocation_from_change_map_task = p.add_task(create_allocation_from_change_map, parent=define_zones_iterator)
convert_simplified_to_original_classes_task = p.add_task(convert_simplified_to_original_classes, parent=define_zones_iterator)

define_zones_iterator.run = 1 # Has to be run, otherwise there won't be the iterator replacements.

process_coarse_change_maps_task.run = 1
create_lulc_task.run = 1
create_physical_suitability_task.run = 1
create_convolution_inputs_task.run = 1
create_conversion_eligibility_task.run = 1
create_overall_suitability_task.run = 1
create_allocation_from_change_map_task.run = 1
convert_simplified_to_original_classes_task.run = 1


process_coarse_change_maps_task.skip_existing = 0
create_lulc_task.skip_existing = 0
create_physical_suitability_task.skip_existing = 0
create_convolution_inputs_task.skip_existing = 0
create_conversion_eligibility_task.skip_existing = 0
create_overall_suitability_task.skip_existing = 0
create_allocation_from_change_map_task.skip_existing = 0
convert_simplified_to_original_classes_task.skip_existing = 0

stitch_projections_task = p.add_task(stitch_projections)
stitch_projections_task.run = 1

clean_intermediate_files_task = p.add_task(clean_intermediate_files)
clean_intermediate_files_task.creates_dir = False
clean_intermediate_files_task.run = 0

main = ''
if __name__ == '__main__':
    from hazelbean.ui import model, inputs
    import seals_ui

    use_ui = False
    if use_ui:
        ui = seals_ui.SealsUI(p)
        ui.run()
        EXITCODE = inputs.QT_APP.exec_()  # Enter the Qt application event loop. Without this line the UI will launch and then close.
    else:

        args = {

            # 'area_of_interest_path': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/seals_admin_regions.shp',
            # 'area_of_interest_path': r"C:\OneDrive\Projects\cge\seals\base_data\seals_admin_regions.shp",
            'area_of_interest_path': r"C:\OneDrive\Projects\cge\seals\projects\project_0_2_6_luh\input\krasnodar_2_zones.shp",
            'base_year_lulc_path': 'C:/OneDrive/Projects/cge/seals/base_data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
            'batch_id': 'seals_id',
            'class_proximity_parameters_path': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/class_proximity_parameters.csv',
            'classes_to_ignore': '0',
            'coarse_change_maps_dir': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input',
            'conversion_eligibility_path': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/conversion_eligibility.csv',
            'enable_batch_mode': True,
            'intermediate_dir': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/intermediate',
            'lulc_class_types_path': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/lulc_class_types.csv',
            'output_base_map_path': 'C:/OneDrive/Projects/cge/seals/base_data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
            'pairwise_class_relationships_path': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/pairwise_class_relationships.csv',
            'physical_suitability_importance': '0.5',
            'physical_suitability_path': 'C:/OneDrive/Projects/cge/seals/base_data/physical_suitability_compressed.tif',
            'skip_existing_batch_components': False,
            'use_existing_batch': False,
            'workspace_dir': 'C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion',
            'remove_unneeded_intermediate_files': False,
        }

        coarse_names = [
            "extensification_results_total_SEALS_ha_0.20yieldscenario",
            "extensification_results_total_SEALS_ha_0.10yieldscenario",
            "extensification_results_total_SEALS_ha_0.05yieldscenario",
            "extensification_results_total_SEALS_ha_baseline",
            "extensification_results_2015-2020_SEALS_ha_0.20yieldscenario",
            "extensification_results_2015-2020_SEALS_ha_0.10yieldscenario",
            "extensification_results_2015-2020_SEALS_ha_0.05yieldscenario",
            "extensification_results_2015-2020_SEALS_ha_baseline",
        ]

        replacements = OrderedDict()
        # replacements['workspace_dir'] = ['C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/' + i for i in coarse_names]
        replacements['intermediate_dir'] = ['C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/intermediate/' + i for i in coarse_names]
        replacements['output_dir'] = ['C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/output/' + i for i in coarse_names]
        replacements['coarse_change_maps_dir'] = ['C:/OneDrive/Projects/cge/seals/projects/unilever_gluc_maize_expansion/input/' + i for i in coarse_names]

        lengths = []
        for k, v in replacements.items():
            lengths.append(len(v))

        set_lengths = list(set(lengths))
        if len(set_lengths) > 1:
            raise NameError('Wrong lengths given to API')
        for i in range(set_lengths[0]):
            for k, v in replacements.items():
                args[k] = v[i]

            p.execute(args)

