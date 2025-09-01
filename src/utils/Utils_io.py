import glob
import os, errno
import logging
import numpy as np

try:
    import matplotlib.pyplot as plt
    import json
except Exception as e:
    print('Import of matplotlib or json failed with: {} \n try to install them with pip install...')

# define some helper classes and a console and file logger
class ConsoleAndFileLogger:
    def __init__(self, logfile_name='Log', log_lvl=logging.INFO, path='./logs/'):
        """
        Create your own logger
        log debug messages into a logfile
        log info messages into the console
        log error messages into a dedicated *_error logfile
        :param logfile_name:
        """

        # Define the general formatting schema
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger()

        # define a general logging level,
        # each handler has its own logging level
        # the console handler ist selectable by log_lvl
        logger.setLevel(logging.DEBUG)

        log_f = os.path.join(path, logfile_name + '.log')
        ensure_dir(os.path.dirname(os.path.abspath(log_f)))

        # delete previous handlers and overwrite with given setup
        logger.handlers = []
        if not logger.handlers:

            # Define debug logfile handler
            hdlr = logging.FileHandler(log_f)
            hdlr.setFormatter(formatter)
            hdlr.setLevel(logging.DEBUG)

            # Define info console handler
            hdlr_console = logging.StreamHandler()
            hdlr_console.setFormatter(formatter)
            hdlr_console.setLevel(log_lvl)

            # write error messages in a dedicated logfile
            log_f_error = os.path.join(path, logfile_name + '_errors.log')
            ensure_dir(os.path.dirname(os.path.abspath(log_f_error)))
            hdlr_error = logging.FileHandler(log_f_error)
            hdlr_error.setFormatter(formatter)
            hdlr_error.setLevel(logging.ERROR)

            # Add all handlers to our logger instance
            logger.addHandler(hdlr)
            logger.addHandler(hdlr_console)
            logger.addHandler(hdlr_error)

        cwd = os.getcwd()
        logging.info('{} {} {}'.format('--' * 10, 'Start', '--' * 10))
        logging.info('Working directory: {}.'.format(cwd))
        logging.info('Log file: {}'.format(log_f))
        logging.info('Log level for console: {}'.format(logging.getLevelName(log_lvl)))

def ensure_dir(file_path):
    """
    Make sure a directory exists or create it
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        logging.debug('Creating directory {}'.format(file_path))

        try:# necessary for parallel workers
            os.makedirs(file_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_plot(fig, path, filename='', override=False, tight=True):
    """
    Saves an matplotlib figure to the given path + filename
    If the figure exists, ad a number at the end and increase it
    as long as there is already an image with this name
    :param fig:
    :param path:
    :param filename:
    :param override:
    :param tight:
    :return:
    """
    logging.debug('Trying to save to {0}'.format(path))
    ensure_dir(path)
    if tight:
        plt.tight_layout()

    i = 0
    if override:
        newname = '{}.png'.format(filename)
        fig.savefig(os.path.join(path, newname))
    else:
        while True:
            i += 1
            newname = '{}{:d}.png'.format(filename + '_', i)
            if os.path.exists(os.path.join(path, newname)):
                continue
            fig.savefig(os.path.join(path, newname))
            break
    logging.debug('Image saved: {}'.format(os.path.join(path, newname)))
    # free memory, close fig
    plt.close(fig)


def init_config(config, save=True):
    """
    Extract all config params (CAPITAL letters) from global or local namespace
    save a serializable version to disk
    make sure all config paths exist

    :param config: dictionary, e.g.> locals() or globals() or any manual defined dict
    :param save:
    :return: config (dict) with all training/evaluation params
    """
    
    allowed_datatypes = [bool, int, str, float, list, dict]

    # make sure config path and experiment name are set
    exp = config.get('EXPERIMENT', 'UNDEFINED')
    exp = config.get('EXP_PATH', os.path.join('tmp/', exp))
    config['EXP_PATH'] = exp
    config['CONFIG_PATH'] = config.get('CONFIG_PATH', os.path.join(exp, 'config'))
    config['TENSORBOARD_PATH'] = config.get('TENSORBOARD_PATH', os.path.join(exp, 'tensorboard_logs'))
    config['MODEL_PATH'] = config.get('MODEL_PATH', os.path.join(exp, 'models'))
    

    # make sure all paths exists
    ensure_dir(config['EXP_PATH'])
    ensure_dir(config['TENSORBOARD_PATH'])
    ensure_dir(config['MODEL_PATH'])
    ensure_dir(config['CONFIG_PATH'])

    # Define a config for param injection and save it for usage during evaluation, save all upper key,value pairs from global namespace
    config = dict(((key, value) for key, value in config.items()
                   if key.isupper() and key not in ['HTML', 'K']))

    if save:
        # convert functions to string representations
        try:
            write_config = dict(
                [(key, value.__name__) if callable(value) else (key, value) for key, value in config.items()])
        except:
            write_config = dict(
                [(key, getattr(value, 'name', 'unknownfunction')) if callable(value) else (key, value) for key, value in config.items()])

        # save only simple data types
        write_config = dict(((key, value) for key, value in write_config.items()
                             if type(value) in allowed_datatypes))

        # save to disk
        with open(os.path.join(write_config['CONFIG_PATH'], 'config.json'), 'w') as fp:
            json.dump(write_config, fp)

    return config


def init_json(json_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    json_file = os.path.join(file_path, 'dataset.json')
    with open(json_file, 'w') as json_file_path:
        json.dump(json_data, json_file_path, indent=4)

    return json_data


def get_json(search_pattern, file_path):
    search_path = os.path.join(file_path, search_pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        search_path = os.path.join(os.path.dirname(file_path), '**', search_pattern)
        files = sorted(glob.glob(search_path))

    print(f"Config file: {files}")
    return files


def get_post_processing(json_file):
    with open(json_file) as json_file:
        json_data = json.load(json_file)
    ret = json_data["post_processing"]
    if ret["use_segmentation"]:
        if not ret["mask_channels"]:
            ret["mask_channels"] = [
                label_id for label, label_id in json_data["labels"].items() if label != "background"
            ]
    else:
        ret["mask_channels"] = []
    return ret



def write_random_example_4d_files_to_disk(PRETRAINED_SEG, config, example_path, moved, number_of_examples, segmentation,
                                          vects, x_val_lax, norm_thresh=55, connected_component_filter=None, mask_channels=None):

    if number_of_examples == None:
        number_of_examples = vects.shape[0] - 1  # export all patients
    else:
        number_of_examples = 1
    np.random.seed(42)
    examples = np.random.choice(np.array(range(vects.shape[0])), size=number_of_examples, replace=False)
    logging.info('Saving example patients with direction as nrrd')

    # order of moved axis is wrong, so rearrange them:
    moved = np.transpose(moved, (0, 1, 4, 2, 3))
    focus_size = round(moved.shape[-1] / 96)  # Setting size of focus point in depending on the size of the image

    write_4d_files_to_disk(examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                           vects, x_val_lax, norm_thresh=norm_thresh, connected_component_filter=connected_component_filter, mask_channels=mask_channels)


def write_4d_files_to_disk(examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                           vects, x_val_lax, norm_thresh=55, connected_component_filter=None, mask_channels=None):
    import SimpleITK as sitk
    import os
    from src.models.predict_phase_reg_model import interpret_deformable

    for example in examples:
        dir_1d_mean, directions, norm_1d_mean, norm_nda, ct, _ = interpret_deformable(vects_nda=vects[example],
                                                                                      masks=segmentation[
                                                                                          example] if PRETRAINED_SEG else None,
                                                                                      mask_channels=mask_channels
                                                                                      if PRETRAINED_SEG else None,
                                                                                      ct_calculation=[1, 2, 3],
                                                                                      norm_percentile=norm_thresh,
                                                                                      component_padding=
                                                                                      connected_component_filter)

        if np.ma.is_masked(directions):
            directions[directions.mask] = -10
            # directions = directions.data * ~directions.mask
        if np.ma.is_masked(norm_nda):
            norm_nda = norm_nda.data * ~norm_nda.mask

        zeros = np.zeros_like(moved[example])
        zeros[:, :,
        int(ct[0] - focus_size):int(ct[0] + focus_size),
        int(ct[1] - focus_size):int(ct[1] + focus_size)] = 1

        # Testen ob es auch direkt mit GetImageFromArray klappt, ohne for schleifen iteration
        sitk_images = [sitk.GetImageFromArray(vol.astype('float32')) for vol in moved[example]]

        sitk_vects = [sitk.GetImageFromArray(vol.astype('float32'), isVector=True) for vol in
                      np.transpose(vects[example], (3, 1, 2, 0))]
        sitk_dir = [sitk.GetImageFromArray(vol.astype('float32')) for vol in rearrange_axis_of_ndarray(directions)]
        sitk_norm = [sitk.GetImageFromArray(vol.astype('float32')) for vol in rearrange_axis_of_ndarray(norm_nda)]
        sitk_foc = [sitk.GetImageFromArray(vol.astype(np.uint8)) for vol in zeros]

        # Define spacing for saving the images
        spacing = config.get('SPACING', (2.5, 2.5))
        spacing = list(reversed(spacing)) + [1.0, 1.0]

        elem = x_val_lax[example]
        file_type = '.nrrd' if '.nrrd' in elem else '.nii.gz'

        # Save image, vector, direction, norm and focus point each as nrrd/NIFTI
        export_img_f_name = os.path.join(example_path, os.path.basename(elem))
        save_sitk(sitk_images, spacing, export_img_f_name)

        export_vec_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_vec.nrrd'))
        save_sitk(sitk_vects, spacing, export_vec_f_name)

        export_dir_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_dir.nrrd'))
        save_sitk(sitk_dir, spacing, export_dir_f_name)

        export_norm_f_name = os.path.join(example_path,
                                          os.path.basename(elem).replace(file_type, '_norm.nrrd'))
        save_sitk(sitk_norm, spacing, export_norm_f_name)

        export_foc_f_name = os.path.join(example_path,
                                         os.path.basename(elem).replace(file_type, '_foc.nrrd'))
        save_sitk(sitk_foc, spacing, export_foc_f_name)

        if PRETRAINED_SEG:
            from src.models.predict_phase_reg_model import seg_based_direction
            sitk_mask = [sitk.GetImageFromArray(np.flipud(vol.astype(np.uint8))) for vol in np.transpose(segmentation[example], (0, 3, 1, 2))]
            new_mask_clean = sitk.JoinSeries(sitk_mask)
            new_mask_clean.SetSpacing(spacing)
            export_mask_f_name = os.path.join(example_path,
                                              os.path.basename(elem).replace(file_type, '_mask.nrrd'))
            sitk.WriteImage(new_mask_clean, export_mask_f_name)
            seg_based_direction(vects[example], moved[example], segmentation[example], x_val_lax[example],
                                focus_size, example_path, config, file_type)

    return


def save_sitk(sitk_img, spacing, export_f_name):
    """
    Save a sitk image to disk.
    """
    import SimpleITK as sitk
    new_img_clean = sitk.JoinSeries(sitk_img)
    new_img_clean.SetSpacing(spacing)
    sitk.WriteImage(new_img_clean, export_f_name)
    return new_img_clean


def rearrange_axis_of_ndarray(array, order=(1, 0, 2, 3), additional_row=True):
    """
    This method rearranges the axis of an ndarray.
    :param array: ndarray to be rearranged
    :param order: order of axis to be rearranged to
    :param additional_row: boolean to add a row to the array
    :return: ndarray rearranged
    """
    if additional_row:
        array = array[np.newaxis, :, :, :]
    array = np.transpose(array, order)
    return array

