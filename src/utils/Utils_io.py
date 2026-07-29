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


def init_json(json_data, file_path=None):
    """
    Initialize a json file with dataset information.

    json_data: dict or str:
        Either:
        - a dictionary containing JSON data, or
        - a path to an existing JSON file.
    """
    print(file_path)
    if isinstance(json_data, (str, os.PathLike)):
        if file_path is None:
            return json_data
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        json_file = os.path.join(file_path, 'dataset.json')
        with open(json_file, 'w') as json_file_path:
            json.dump(json_data, json_file_path, indent=4)

        return json_data

    if isinstance(json_data, dict):
            if file_path is not None:
                os.makedirs(file_path, exist_ok=True)
                json_file = os.path.join(file_path, 'dataset.json')
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=4)

            return json_data

def get_json(search_pattern, file_path):
    search_path = os.path.join(file_path, search_pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        search_path = os.path.join(os.path.dirname(file_path), '**', search_pattern)
        files = sorted(glob.glob(search_path))

    print(f"Config file: {files}")
    return files


def get_post_processing(json_data):
    if type(json_data) == str:
        with open(json_data) as json_file:
            json_data = json.load(json_file)
    ret = json_data["post_processing"]
    if ret["use_segmentation"]:
        if not ret.get("mask_channels"):
            ret["mask_channels"] = [
                label_id for label, label_id in json_data["labels"].items() if label != "background"
            ]
    else:
        ret["mask_channels"] = []
    return ret



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



# Add z in the method as parameter and gtind!
def plot_direction_instance(dir_1d_mean, directions, exp_path, mid, norm_1d_mean, norm_nda, patient, save, upper):
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D
    from src.models.KerasLayers import minmax_lambda
    # with sb.plotting_context("paper"):
    plt.rcParams['font.size'] = '16'
    timesteps = directions.shape[0]
    size_per_timestep = (25 / timesteps) * 2  # use a relative hight, to scope with the square mid-cavity plots
    figsize = (25, size_per_timestep)
    fig = plt.figure(figsize=figsize)
    # DIR 2D+t
    dir_2d_t = directions.copy()  # SKM Combine: here it is for SAX .copy()[:,z]
    if np.ma.is_masked(directions): dir_2d_t = dir_2d_t.data * ~directions.mask
    div_cmap = 'bwr'
    fig = show_2D_or_3D(dir_2d_t, allow_slicing=False, cmap=div_cmap, fig=fig, interpolation=None, vmin=-1, vmax=1)
    ax_ = fig.get_axes()[0]
    _ = ax_.set_yticks([])
    _ = ax_.set_xticks([])
    ax = fig.get_axes()[1]
    _ = ax.set_ylabel(r'$\alpha$ ' + '\n2d+t')  # \nmid
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    cax = fig.add_axes([0.45, 0.84, 0.1, 0.03])
    cb = fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
    cb.ax.tick_params(color="black", labelsize=10, labelcolor='black')
    rows = 2
    pos = 2
    ax = fig.add_subplot(rows, 1, pos)
    # DIR 2D T x Z
    # SKM Combine: SAX has this definitions.... necessary?
    # directions_tz = np.ma.mean(directions,axis=(2, 3))
    # _ = ax.imshow(directions_tz.T, aspect='auto', cmap=div_cmap, vmin=directions_tz.min(), vmax=directions_tz.max(),
    #               origin='lower', interpolation='none')
    # _ = ax.set_xticks(gtind, minor=False)
    ax2 = ax.twinx()
    # DIR 1D
    # SKM Combine: SAX has this definitions.... necessary?
    # current_z = directions.shape[1] // 2
    # dir_1d_mean_apex = np.ma.mean(directions_tz[:, :current_z], axis=1)
    # dir_1d_mean_base = np.ma.mean(directions_tz[:, current_z:], axis=1)
    _ = ax2.plot(dir_1d_mean, c='black', label=r'$\alpha_{ap_t}$')
    # SKM Combine: SAX has this definitions.... necessary?, makes plot per base and apex
    # _ = ax2.plot(dir_1d_mean_apex, c='green', label=r'$\alpha_{ap_t}$')
    # _ = ax2.plot(dir_1d_mean_base, c='black', label=r'$\alpha_{ba_t}$')
    _ = ax.set_yticks([])
    _ = ax.set_ylabel(r'$\alpha$' + '\nz+t')  # \nap:ba
    ax2.label_outer()
    _ = ax2.tick_params(axis="y", pad=-40)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.axhline(0, color='red', linestyle='--', label='Zero Line')
    ax2.axhline(0, color='grey', linestyle='--', label='Zero Line')
    if save: fig.savefig(os.path.join(exp_path, '{}_alpha.svg'.format(patient)))
    norm_cmap = 'hot'
    # NORM 2D + t
    fig = plt.figure(figsize=figsize)
    norm_2d_t = norm_nda  # SKM Combine: Here also norm_nda[:, z]
    norm_2d_t = minmax_lambda([norm_2d_t, mid, upper])
    fig = show_2D_or_3D(norm_2d_t, allow_slicing=False, cmap=norm_cmap, interpolation='none',
                        fig=fig)
    ax = fig.get_axes()[0]
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    ax = fig.get_axes()[1]
    _ = ax.set_ylabel(r'$|\vec{v}|$' + '\n2d+t')  # \nmid
    _ = ax.set_yticks([])
    _ = ax.set_xticks([])
    cax = fig.add_axes([0.45, 0.84, 0.1, 0.03])
    cb = fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
    # cb.set_ticks([])
    cb.ax.tick_params(color="white", labelsize=10, labelcolor='white')
    rows = 2
    pos = 2
    ax = fig.add_subplot(rows, 1, pos)
    # SKM Combine: Also this is listed for Sax
    # # NORM 2D TxZ
    # norm2d = minmax_lambda([norm_nda.mean(axis=(2, 3)), mid, upper])
    # _ = ax.imshow(norm2d.T, aspect='auto', origin='lower', cmap=norm_cmap, interpolation='none')
    # _ = ax.set_xticks(gtind, minor=False)
    # _ = ax.set_yticks([])
    # _ = ax.set_ylabel(r'$|\vec{v}|$' + '\nz+t')  # \nap:ba
    ax2 = ax.twinx()
    _ = ax2.plot(minmax_lambda([norm_1d_mean, mid, upper]), c='black', label=r'$|\vec{v}_{t}|$')
    # SKM Combine: SAX line:
    # ax2.text(len(norm_1d_mean) / 2, norm_1d_mean[len(norm_1d_mean) // 2] - 0.2, r'$|\vec{v}_{t}|$', color='black')
    _ = ax2.tick_params(axis="y", pad=-20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if save:
        try:
            fig.savefig(os.path.join(exp_path, '{}_norm.svg'.format(patient)))
        except Exception as e:
            print(e)
            fig.savefig(os.path.join(exp_path, '{}_norm.png'.format(patient)))
    return fig
