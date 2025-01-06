DATASET_LOCATION = 'pysaliency_datasets'
MIT300_FIXATIONS = 'datasets/MIT300/fixations_with_initial_central_fixation.hdf5'
# CAT2000_FIXATIONS = 'datasets/CAT2000/fixations.hdf5'
CAT2000_FIXATIONS = 'datasets/CAT2000_full_data/test_fixation_trains.hdf5'
COCO_FREEVIEW_FIXATIONS='pysaliency_datasets/COCO-Freeview/fixations_test.hdf5'
OUTPUT_LOCATION = 'output'

CAT2000_BASELINE = 'submissions/initial_CAT2000/centerbias/predictions.hdf5'
MIT300_BASELINE = 'submissions/initial/centerbias/predictions.hdf5'
COCO_FREEVIEW_BASELINE = 'submissions/initial_COCO-Freeview/centerbias/predictions.hdf5'

BASELINE_SUBMISSIONS = {
    'mit300': 'submissions/initial/centerbias',
    'cat2000': 'submissions/initial_CAT2000/centerbias',
    'coco-freeview': 'submissions/initial_COCO-Freeview/centerbias',
}