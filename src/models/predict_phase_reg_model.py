# predict cardiac phases for a cv experiment
import logging
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Union, Literal

from scipy.ndimage import gaussian_filter1d


DEBUG = False

class CMRPhaseDetector:
    def __init__(self, model_config, data_info_path, data_root, exp_path, c2l=True):
        """
        :param model_config: path to config file
        :type model_config: str
        :param data_root: path to data root (cmr data + masks (optional)
        """
        import json, logging, os
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        import tensorflow as tf
        tf.get_logger().setLevel('FATAL')
        from logging import info
        from src.utils.Utils_io import ConsoleAndFileLogger, ensure_dir
        from ProjectRoot import change_wd_to_project_root
        change_wd_to_project_root()
        from src.data.Postprocess import get_predicted_as_segmentation
        from src.utils.Tensorflow_helper import choose_gpu_by_id

        # load the experiment config and json
        if type(model_config) == type(''):
            with open(model_config, encoding='utf-8') as data_file:
                self.config = json.loads(data_file.read())
        else:
            self.config = model_config

        globals().update(self.config)

        if data_info_path is not None and type(data_info_path) == type(""):
            with open(data_info_path, encoding='utf-8') as data_file:
                self.dataset_json = json.loads(data_file.read())
        elif data_info_path is not None:
            self.dataset_json = data_info_path
        else:
            self.dataset_json = {}

        self.Masker = CMRViewMasker(self.dataset_json)

        # Read in used view to define future processing of 3D+t (SAX) or 2D+t (LAX, 4CH, 3CH, 2CH)
        # Done? SKM combine: Add flag/information if it is lax or sax to decide here
        self.view = self.dataset_json['view']
        self.CMR3D = self.view.lower() == "sax"


        # ------------------------------------------define GPU id/s to use
        self.GPU_IDS = self.config.get('GPU_IDS', '0,1')
        self.GPUS = choose_gpu_by_id(self.GPU_IDS)
        print(self.GPUS)
        print(tf.config.list_physical_devices('GPU'))

        self.EXPERIMENT = self.config.get('EXPERIMENT', 'UNDEFINED')
        ConsoleAndFileLogger(self.EXPERIMENT, logging.INFO)
        info('Loaded config for experiment: {}'.format(self.EXPERIMENT))
        self.PRETRAINED_SEG = self.config.get('PRETRAINED_SEG', False)

        # Load CMR sequences
        # cluster to local data mapping
        if c2l:
            self.config['DATA_PATH_'+self.view] = os.path.join(data_root, self.view.lower())
            self.config['MODEL_PATH'] = os.path.join(exp_path, *self.config['MODEL_PATH'].split('/')[-2:])
            if not self.config.get('INFERENCE', False):
                self.config['DF_FOLDS'] = os.path.join(data_root, 'df_kfold.csv')
                self.config['DF_META'] = os.path.join(data_root, 'phases.csv')
                self.config['EXP_PATH'] = exp_path  # replace the relative path with the SDS path

            if exp_path is not None and self.PRETRAINED_SEG:
                self.config['SEGMENTATION_MODEL'] = os.path.join(exp_path .replace('phase_regression', ''),
                                                            *self.config['SEGMENTATION_MODEL'].split('/')[-2:])
                self.config['SEGMENTATION_WEIGHTS'] = os.path.join(exp_path .replace('phase_regression', ''),
                                                              *self.config['SEGMENTATION_WEIGHTS'].split('/')[-4:])

        self.suffix = self.dataset_json.get("suffix", None)
        if self.suffix is None:
            self.FILE_ENDING = "nii.gz"
        else:
            self.FILE_ENDING = self.suffix["file_ending"]


    def predict(self, number_of_examples=None):
        """
        Predict on the held-out validation split

        """
        import numpy as np
        from src.data.Dataset import get_trainings_files
        from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
        from src.models.PhaseRegModels import PhaseRegressionModel

        from src.utils.Utils_io import ensure_dir
        from src.data.Postprocess import get_predicted_as_segmentation

        x_train_cmr, y_train_cmr, x_val_cmr, y_val_cmr = get_trainings_files(data_path=self.config['DATA_PATH_'+self.view],
                                                                             suffix=self.suffix,
                                                                             ftype=self.FILE_ENDING,
                                                                             path_to_folds_df=self.config['DF_FOLDS'],
                                                                             fold=self.config['FOLD'])

        logging.info(f'{self.view} train CMR: {len(x_train_cmr)}, {self.view}train masks: {len(y_train_cmr)}')
        logging.info(f'{self.view} val CMR: {len(x_val_cmr)}, {self.view} val masks: {len(y_val_cmr)}')

        chunk_size = 10
        x_train_cmrs = [x_train_cmr[i:i + chunk_size] for i in range(0, len(x_train_cmr), chunk_size)]
        y_train_cmrs = [y_train_cmr[i:i + chunk_size] for i in range(0, len(y_train_cmr), chunk_size)]
        x_val_cmrs = [x_val_cmr[i:i + chunk_size] for i in range(0, len(x_val_cmr), chunk_size)]
        y_val_cmrs = [y_val_cmr[i:i + chunk_size] for i in range(0, len(y_val_cmr), chunk_size)]

        logging.info('Split into chunks of:')
        logging.info(f'{self.view} train CMR: {len(x_train_cmrs)}, {self.view} train masks: {len(y_train_cmrs)}')
        logging.info(f'{self.view} val CMR: {len(x_val_cmrs)}, {self.view} val masks: {len(y_val_cmrs)}')

        # turn off all augmentation operations while inference
        # create another config for the validation data
        # we want the prediction to run with batchsize of 1
        # otherwise we might inference only on the even number of val files
        # the mirrored strategy needs to get a single gpu instance named, otherwise batchsize=1 does not work
        val_config = self.config.copy()
        val_config['SHUFFLE'] = False
        val_config['AUGMENT'] = False
        val_config['AUGMENT_PHASES'] = False
        val_config['AUGMENT_TEMP'] = False
        val_config['BATCHSIZE'] = 1
        val_config['HIST_MATCHING'] = False
        val_config['GPUS'] = ['/gpu:0']
        val_config['CMR3D'] = self.CMR3D
        model = PhaseRegressionModel(val_config).get_model()
        logging.info('Trying to load the model weights')
        logging.info('work dir: {}'.format(os.getcwd()))
        logging.info('model weights dir: {}'.format(os.path.join(val_config['MODEL_PATH'], 'model.h5')))
        model.load_weights(os.path.join(val_config['MODEL_PATH'], 'model.h5'))
        logging.info('loaded model weights as h5 file')

        # predict on the validation generator
        # this should avoid memory leaks for huge inference datasets
        pred_path = os.path.join(val_config['EXP_PATH'], 'pred')
        moved_path = os.path.join(val_config['EXP_PATH'], 'moved')
        example_path = os.path.join(val_config['EXP_PATH'], 'example')

        ensure_dir(pred_path)
        ensure_dir(moved_path)
        ensure_dir(example_path)

        for junk, (x_train_cmr_, y_train_cmr_, x_val_cmr_, y_val_cmr_) in enumerate(zip(x_train_cmrs, y_train_cmrs, x_val_cmrs, y_val_cmrs)):
            preds_, moved_, vects_, gts_, segmentations_ = [], [], [], [], []
            logging.info('***********  processing junk: {} of {}'.format(junk, len(x_val_cmrs)))
            validation_generator = PhaseRegressionGenerator_v2(x_val_cmr_, y_val_cmr_, config=val_config, in_memory=False, dataset_json=self.dataset_json)

            for i, (x, y) in enumerate(validation_generator):
                results = model.predict_on_batch(x)
                if self.PRETRAINED_SEG:
                    preds, moved, vects, seg = results
                    segmentations_.append(get_predicted_as_segmentation(seg[0], return_as='label', start_c=1, threshold=0.5,
                                                                        connected_component=True).astype(np.uint8))
                else:
                    preds, moved, vects = results
                    if self.Masker.PRECOMPUTED_MASKS:
                        seg = y[3]
                        segmentations_.append(seg)

                preds_.append(preds.astype('float16'))
                moved_.append(moved.astype('float16'))
                vects_.append(vects.astype('float16'))
                gts_.append(y[0])

            fold = "{:04d}".format((100 * val_config['FOLD']) + junk) if val_config['FOLD'] > 0 else "{:04d}".format(junk)

            pred_filename = os.path.join(pred_path, 'gtpred_fold{}.npy'.format(fold))
            moved_filename = os.path.join(moved_path, 'moved_f{}.npy'.format(fold))
            vects_filename = os.path.join(moved_path, 'vects_f{}.npy'.format(fold))

            preds = np.concatenate(preds_, axis=0)
            moved = np.concatenate(moved_, axis=0)
            vects = np.concatenate(vects_, axis=0)
            gts = np.concatenate(gts_, axis=0)

            np.save(pred_filename, np.stack([gts, preds], axis=0))
            np.save(moved_filename, moved)
            np.save(vects_filename, vects)

            if self.PRETRAINED_SEG or self.Masker.PRECOMPUTED_MASKS:
                segmentation_filename = os.path.join(pred_path, 'segmentation_f{}'.format(fold))
                segmentation = np.stack(segmentations_, axis=0)
                np.save(segmentation_filename, segmentation)
            else:
                segmentation = None


            if number_of_examples > 0:
                self.write_random_example_4d_files_to_disk(self.PRETRAINED_SEG or self.Masker.PRECOMPUTED_MASKS, self.config, example_path, moved,
                                                           number_of_examples,
                                                           segmentation,
                                                           vects, x_val_cmr_,
                                                           norm_thresh=self.Masker.norm_threshold,
                                                           connected_component_filter=self.Masker.connected_component_filter,
                                                           mask_channels=self.Masker.mask_channels)

        del validation_generator
        del model

        # create a list of patients based on the filenames
        patients_filename = os.path.join(pred_path, 'patients.txt')
        with open(patients_filename, "a+") as f:
            _ = [f.writelines(str(val_config['FOLD']) + '_' + os.path.basename(elem) + '\n') for elem in x_val_cmr]
        logging.info('saved as: \n{}\n{} \n example patients processed!'.format(pred_filename, patients_filename))


    def predict_phase_from_deformable(self, create_figures=True, dir_axis=0, roll_by_gt=True,
                                  normalise_dir=False, normalise_norm=False, return_files=False, mask_channels=None,
                                  ct_calculation='septum', save_dir_as_nrrd=False, max_junks=None,
                                  norm_thresh=50, connected_component_filter=None, exp_mode=None, diff_thresh=1.0):
        """
        Predict the temporal occurence for five cardiac phases from a cmr-phase-regression experiment folder
        Expects to find all files written from a CV-experiment, e.g.> train_regression_model.py
        Args:
            create_figures (bool): Creates plots with direction and norm curves and violin + scatterplot for phase detection
            dir_axis (int): out of [0,1], 0 = z,y,x motion, 1 = y,x motion, z- is negative during systole, y,x positive
            roll_by_gt (bool): use the gt labels or the pred labels to align the direction cohort plots
            normalise_dir (bool): normalise the direction cohort plots
            normalise_norm (bool): normalise the norm cohort plots
            return_files (bool): Return prediction results as a dataframe
            mask_channels (list): list of channel names for masks
            ct_calculation (str): Can be None, if self-supervised, a list of int for mask channels or 'septum'
            save_dir_as_nrrd: save the prediction results as nrrd
            max_junks (int): max number of junks to use
            norm_thresh (int): 0 < norm_thresh < 100, overrides the dataset json's post_processing.norm_threshold
            connected_component_filter (int): padding size around the largest connected component, or None to disable
            exp_mode (str): focus-point mode: None/'mse' balanced center, 'vol' volume center, 'septum'/'lv' anatomical
            diff_thresh (float): minimal direction change (max-min) for a voxel to be included in the self-supervised mask

        Returns:

        """
        from time import time
        t0 = time()
        if DEBUG: print('start: {:0.3f} s'.format(t0))
        import numpy as np
        import pandas as pd
        import os
        import logging
        from src.data.Dataset import load_phase_reg_exp
        from src.utils.Metrics import meandiff
        from src.data.Postprocess import align_resample_multi
        from src.visualization.Visualize import plot_dir_norm, plot_dir_norm_split_by, plot_pfd_per_phase_as_violin, \
            plot_scatter

        aligned_length = 40
        vols_alignedscaled = None
        vols_rv_alignedscaled = None

        # load all files of this experiment
        nda_vects, gt, pred, gt_len, mov, masks, patients = load_phase_reg_exp(self.config['EXP_PATH'], junk=max_junks)
        logging.info(f'vects_nda: {nda_vects.shape}')

        if mask_channels is not None:
            self.Masker.mask_channels = mask_channels

        if self.Masker.mask_channels is None and masks is not None:
            self.Masker.mask_channels = np.delete(np.unique(masks), 0, axis=0)
            logging.info(f"Set mask channels to {self.Masker.mask_channels}")
        elif masks is not None and len(self.Masker.mask_channels) > 1:
            logging.info(f"Set mask channels to {self.Masker.mask_channels}")
        else:
            logging.info("Found no masks, will use self-supervised masking.")

        # SKM Combine: change here for 2D and 3 D difference (spacing, and calc_vol_along_t function)
        if masks is not None:
            from src.data.Dataset import calc_vol_along_t
            # calc the lv blood pool volume from the predicted mask 1=RV,2=MYO,3=LV
            vols_lv = np.array([calc_vol_along_t(msk, label=3, spacing=[2.5, 2.5, 2.5]) for msk in masks])
            vols_rv = np.array([calc_vol_along_t(msk, label=1, spacing=[2.5, 2.5, 2.5]) for msk in masks])

        if self.Masker.use_segmentation:
            # SKM Combine change here also code to fit libary of names the dataset.json
            logging.info(
                f'Using mask(s): {", ".join(np.array(["None", "left ventricle", "left ventricle myocardium", "rv myocardium"])[self.Masker.mask_channels])} for calculation')
            if ct_calculation is None:
                ct_calculation = [1]
        t1 = time()
        print('files loaded {:0.3f} s, continue with deformable2direction2phase'.format(t1 - t0))

        # predict phase per patient and write result as df into experiment folder
        pred_u = np.zeros_like(gt)
        upred_ind = []
        gt_ind = []
        cycle_len = []
        dir_1ds = []
        norms_1ds = []
        directions = []
        norms = []
        msks = []
        cts = []
        executor = ThreadPoolExecutor(max_workers=12)
        instances = nda_vects.shape[0]
        dir_axis_list = [dir_axis] * instances
        indicies = list(range(instances))
        f_names = [None] * instances
        if not return_files: mov = [None] * instances  # save memory, we only need to return the cmr for jupyter plots
        cts = [ct_calculation] * instances
        assert len(patients) == instances, f'please check your list of patients ({len(patients)} != {instances})'
        if save_dir_as_nrrd:
            f_names = [os.path.join(self.config['EXP_PATH'], p) for p in patients]

        if DEBUG: print(pred_u.shape)

        norm_thresh_list = [norm_thresh] * instances
        connected_component_filter_list = [connected_component_filter] * instances
        exp_mode_list = [exp_mode] * instances
        diff_thresh_list = [diff_thresh] * instances

        if self.Masker.use_segmentation:
            # array has to be repeated for iterator used in multithreading
            masks_multi = masks
            mask_channels_multi = [self.Masker.mask_channels] * instances
        else:
            masks_multi = [None] * instances
            mask_channels_multi = [None] * instances

        # positional order must match interpret_deformable_async's signature exactly, since
        # executor.map(func, *iterables) zips these lists positionally
        params = [nda_vects, gt_len, gt, dir_axis_list, indicies, f_names, cts, masks_multi, mask_channels_multi,
                 norm_thresh_list, connected_component_filter_list, exp_mode_list, diff_thresh_list]

        for result in executor.map(self.interpret_deformable_async, *params):
            cardiac_cycle_length, dir_1d_mean, ind, indices, norm_1d_mean, weight, i, ct, mask = result

            cycle_len.append(cardiac_cycle_length)
            upred_ind.append(indices)
            gt_ind.append(ind)
            dir_1ds.append(dir_1d_mean)
            norms_1ds.append(norm_1d_mean)
            msks.append(mask)
            # directions.append(direction_nda)
            # norms.append(norm_nda)
            indices = np.array(indices)
            onehot = np.zeros((indices.size, cardiac_cycle_length))
            onehot[np.arange(indices.size), indices] = weight
            pred_u[i][0:cardiac_cycle_length] = onehot.T
            cts[i] = ct
            t_temp = time()
            if DEBUG: print('prediction took: {:0.3f} s'.format(t_temp - t1))
        upred_ind = np.stack(upred_ind, axis=0)
        gt_ind = np.stack(gt_ind, axis=0)
        cycle_len = np.stack(cycle_len, axis=0)
        dir_1ds = np.stack(dir_1ds, axis=0)
        norms_1ds = np.stack(norms_1ds, axis=0)

        # re-create a compatible shape for the metric fn
        gt_ = np.stack([gt, gt_len], axis=1)
        pred_ = np.stack([pred_u, np.zeros_like(pred_u)], axis=1)

        # create some dataframes for further processing
        phases = ['ED', 'MS', 'ES', 'PF', 'MD']
        res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
        pfd_df = pd.DataFrame(res.numpy(), columns=phases)
        pfd_df['patient'] = patients
        pfd_df.to_csv(os.path.join(self.config['EXP_PATH'], 'cfd.csv'))
        # save the predicted phases as csv
        pred_df = pd.DataFrame(upred_ind, columns=phases)
        pred_df['patient'] = patients
        pred_df.to_csv(os.path.join(self.config['EXP_PATH'], 'pred_phases.csv'))

        gt_df = pd.DataFrame(gt_ind, columns=phases)
        gt_df['patient'] = patients
        gt_df.to_csv(os.path.join(self.config['EXP_PATH'], 'gt_phases.csv'), index=False)
        t2 = time()
        if DEBUG: print('prediction complete: {:0.3f} s'.format(t2 - t1))

        # create some plots
        t3 = time()
        if not roll_by_gt: gt = pred_u
        dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled = align_resample_multi(dirs=dir_1ds,
                                                                                             norms=norms_1ds,
                                                                                             gt=gt,
                                                                                             gt_len=gt_len,
                                                                                             target_t=aligned_length,
                                                                                             normalise_dir=normalise_dir,
                                                                                             normalise_norm=normalise_norm,
                                                                                             rescale=True)
        if create_figures:
            _, _ = plot_dir_norm(dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled, self.config['EXP_PATH'],
                                 fname='alpha_per_patient')

        if return_files: dirs_aligned, norms_aligned, gt_ind_aligned = align_resample_multi(dirs=dir_1ds,
                                                                                            norms=norms_1ds,
                                                                                            gt=gt,
                                                                                            gt_len=gt_len,
                                                                                            target_t=aligned_length,
                                                                                            normalise_dir=normalise_dir,
                                                                                            normalise_norm=normalise_norm,
                                                                                            rescale=False)

        if create_figures:
            _ = plot_pfd_per_phase_as_violin(df_pfd=pfd_df, exp_path=self.config['EXP_PATH'])
            fig = plot_scatter(exp_path=self.config['EXP_PATH'], gt_df=gt_df, phases=phases, pred_df=pred_df)

        return_args = [pred_df, gt_df, pfd_df, res, cycle_len]

        if return_files and 'dirs_aligned' in locals():
            if type(masks) is not type([]) and type(masks) is not type(None) and len(masks) > 0:
                msks = masks
            return_args += (nda_vects, msks, gt, pred, gt_len, mov, patients, dir_1ds,
                            norms_1ds, gt_ind_alignedscaled, dirs_alignedscaled, norms_alignedscaled, gt_ind_alignedscaled,
                            vols_alignedscaled, vols_rv_alignedscaled, cts)
        if DEBUG: print('load intermediate files took: {}'.format(time() - t3))
        return return_args


    def get_directions(self, ct, dim_, length, vects_nda, diff_thresh=1.2, masked=False, dir_axis=0, as_angle=False,
                       sigma=0.8, norm_thresh=None, connected_component_filter=None):
        """
        Create a focus matrix with the shape specified in dim_.
        In this matrix for each voxel we will have a focus vector pointing towards the center ct.
        For each vector v in the deformable vects_nda calculate the cosine angle (direction) between this deformation vector
        and the corresponding focus vector
        that goes
        Args:
            ct ():
            dim_ ():
            length ():
            vects_nda ():
            diff_thresh ():

        Returns:

        Parameters
        ----------
        masked defines if vectors are already filtered by a use_segmentation mask

        """
        import numpy as np
        import tensorflow as tf
        import scipy.ndimage
        from src.models.KerasLayers import get_idxs_tf, get_focus_tf, flow2direction_lambda
        idx = get_idxs_tf(dim_)
        c = get_focus_tf(ct[0:len(dim_)], dim_)
        centers = c - idx
        centers_tensor = centers[tf.newaxis, ...]

        # direction relative to the focus point C_n
        if as_angle:
            directions = CMRPhaseDetector.signed_angle_np(vects_nda[..., 1:], centers_tensor[..., 1:].numpy())  # inplane polar angle
        else:
            directions = flow2direction_lambda([vects_nda[..., dir_axis:], centers_tensor[..., dir_axis:]])[
                ..., 0].numpy()  # remove the last extra channel
            if np.min(directions) < -1:
                print("cosine smaller than -1 detected: " + str(np.min(directions)))
            if np.max(directions) > 1:
                print("cosine large than 1 detected: " + str(np.max(directions)))

        directions_cut = directions[:length]
        dir_rest = directions[length:]
        dir_rest = scipy.ndimage.gaussian_filter(dir_rest, sigma=sigma,
                                                 mode='wrap')  # no border wrapping for the rest sequence

        # smooth the direction field - especially at the cycle end
        directions_cut = scipy.ndimage.gaussian_filter(directions_cut, sigma=sigma, mode='wrap')
        directions[:length] = directions_cut
        directions[length:] = dir_rest
        dir_mask = None
        norm_thresh = self.Masker.norm_threshold if norm_thresh is None else norm_thresh

        if not masked and norm_thresh != 0:  # additional filtering required
            # create mask by another constrain: we include only voxels with a direction change (max - min) greater than
            # smooth the direction field - especially at the cycle end
            # directions = scipy.ndimage.gaussian_filter(directions, sigma=0.8, mode='wrap')
            # np.ma.min(directions_cut, axis=0)
            min_dir_ = np.min(directions_cut, axis=0)
            max_dir_ = np.max(directions_cut, axis=0)
            diff = max_dir_ - min_dir_
            # diff_thresh = np.percentile(diff,diff_thresh)
            if DEBUG: print('diff thresh', diff_thresh)
            dir_mask = diff >= diff_thresh

            # binary opening removes single voxels and close crushed vessels
            # SKM Combine:  SAX/LAX difference structure = np.ones((1, 3, 3))
            structure = np.ones((3, 3))
            dir_mask = scipy.ndimage.binary_opening(dir_mask, structure=structure, iterations=1)

        elif norm_thresh == 0:
            dir_mask = np.zeros_like(directions_cut)
            dir_mask = dir_mask == 0

        connected_component_filter = self.Masker.connected_component_filter if connected_component_filter is None \
            else connected_component_filter
        if connected_component_filter is not None and connected_component_filter is not False:
            # SKM Combine: Check if connected component filter works for 2D and 3D
            # # (even though we didn't use it in the experiments, so remove if not working)
            dir_mask = CMRViewMasker.filter_for_connected_components(dir_mask, pad_size=connected_component_filter)

        return dir_mask, directions

    def interpret_deformable_async(self, nda_vect, gt_len, gt, dir_axis,  idx, filename=None,
                                   ct_calculation: Union[Literal['septum'], list, int, None] = 'septum',
                                   masks=None, mask_channels=None, norm_thresh=None,
                                   connected_component_filter=None, experiment_mode=None, diff_thresh=1.2):
        import numpy as np
        weight = 1
        cardiac_cycle_length = int(gt_len[:, 0].sum())
        ind = np.argmax(gt[:cardiac_cycle_length], axis=0)

        if cardiac_cycle_length > 40:
            print(f"PROBLEM! Skip patient {idx}\nThe cardiac cycle lenght is: {cardiac_cycle_length}")

        dir_1d_mean, direction_nda, norm_1d_mean, norm_nda, ct, masks = self.interpret_deformable(
            vects_nda=nda_vect,
            dir_axis=dir_axis,
            length=cardiac_cycle_length,
            filename=filename,
            masks=masks,
            mask_channels=mask_channels,
            ct_calculation=ct_calculation,
            norm_thresh=norm_thresh,
            connected_component_filter=connected_component_filter,
            experiment=experiment_mode,
            diff_thresh=diff_thresh)

        from src.utils.detect_phases_from_dir import detect_phases
        indices = detect_phases(dir_1d_mean=dir_1d_mean[:cardiac_cycle_length])

        return cardiac_cycle_length, dir_1d_mean, ind, indices, norm_1d_mean, weight, idx, ct, masks

    def interpret_deformable(self, vects_nda, masks=None, mask_channels=None, dir_axis=0, length=None, filename=None,
                             z=None, diff_thresh=1.2,  sigma=0.8, as_angle=False,
                             ct_calculation: Union[Literal['septum'], list, int, None] = 'septum',
                             norm_thresh=None, connected_component_filter=None, experiment=None):
        import numpy as np
        from scipy import ndimage
        from src.visualization.save import write_sitk
        if length is None:
            length = vects_nda.shape[0]
        if vects_nda.dtype is not np.float32: vects_nda = vects_nda.astype("float32")
        if z is None:
            z = vects_nda.shape[1] // 2
        if experiment is None:
            experiment = self.Masker.focus_point
        if isinstance(experiment, str):
            experiment = experiment.lower()
        # vects_nda: vectors towards registered voxels (M->F), (40, 128, 128, 2)
        # mask: binary mask from use_segmentation, repeated to 3 channels repeated to match vects_nda, (40, 128, 128, 2)
        # norm_nda: vector_length, (40, 128, 128)
        # dim_: dimension of on image (128, 128)
        # ct: center of mass calculated from mask
        # directions: direction relative to vector pointing toward ct
        # dir_1d_mean: mean of direction vectors
        # norm_1d_mean: mean of vector lengths

        dim_ = vects_nda.shape[1:-1]
        # calc the norm (supervised and self-supervised is similar)
        norm_mask, norm_nda = self.Masker.get_norm(dir_axis, vects_nda, norm_threshold=norm_thresh)
        norm_nda = norm_nda.astype(np.float16)
        all_masks = []

        if masks is not None and mask_channels is not None and len(mask_channels) > 0:
            # supervised mask and center derivation
            if DEBUG:
                _ = np.array(["right ventricle", "myocardium", "left ventricle"])
                print(f'Using mask(s): {", ".join(_[mask_channels])} for calculation')

            # SKM Combine: What impact does this line have:
            # vects_nda_ma = vects_nda
            # vects_nda_ma = ndimage.filters.gaussian_filter(vects_nda_ma, sigma=1.0)
            mask = self.Masker.get_as_single_mask(masks, channels=mask_channels).astype(bool)
            # SKM Combine: if vol add option for 3D or 2D + Add this differntiation below too
            if ct_calculation == 'VOL':
                ct = [dim_[0] / 2, dim_[1] / 2]
            elif ct_calculation == 'MSE':
                ct = CMRPhaseDetector.get_balanced_center(dim_, norm_mask)
            else:
                ct = self.get_focus_point(masks, print_ct=DEBUG, calculation=ct_calculation, z=z)
                # SKM Combine: Add option to differentiate between LAX and SAX to set z
            # SKM Combine: change here vects_nda to vects_nda_ma, to try out
            _, directions = self.get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda, as_angle=as_angle,
                                           diff_thresh=diff_thresh, masked=True,
                                           connected_component_filter=connected_component_filter)
        else:
            # 1st center definition,
            # Volume center & norm_msk COM
            vects_nda_ma = vects_nda * np.broadcast_to(norm_mask[None, ..., None], shape=vects_nda.shape)
            # Select focus point: None/'mse' balances the volume center towards the norm-mask COM,
            # 'vol' takes the plain volume center, 'septum'/'lv' use an anatomical focus point
            if experiment == 'vol':
                ct = np.array(dim_) // 2
            elif experiment in ('septum', 'lv'):
                ct = self.get_focus_point(masks, print_ct=DEBUG, calculation=ct_calculation, z=z)
            else:
                ct = CMRPhaseDetector.get_balanced_center(dim_, norm_mask)
            mask, directions = self.get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda_ma,
                                              as_angle=as_angle,
                                              sigma=sigma, diff_thresh=diff_thresh,
                                              connected_component_filter=connected_component_filter)
            if mask is None:
                mask = norm_mask
            all_masks.append(mask)
            if filename is not None: write_sitk(directions, filename=filename, suffix='dir1')

            # 2nd center definition: refine the center via COM of the direction mask - only for the
            # balanced-center (None/'mse') mode; 'vol'/'septum'/'lv' keep the fixed 1st-step center
            if experiment is None or experiment == 'mse':
                ct = np.array([*ndimage.center_of_mass(mask)[0:2]])
                # SKM Combine: add here also 3D vs 2D option for focus point
                # ct = np.array([int(z), *ndimage.center_of_mass(mask)[1:]])
                mask, directions = self.get_directions(ct=ct, dim_=dim_, length=length, vects_nda=vects_nda_ma,
                                                  as_angle=as_angle,
                                                  sigma=sigma, diff_thresh=diff_thresh,
                                                  connected_component_filter=connected_component_filter)
                # SKM Combine: Check impact of binary opening
                # structure = np.ones((1, 3, 3))
                # mask = scipy.ndimage.binary_opening(mask, structure=structure, iterations=1)
                if mask is None:
                    mask = norm_mask

            all_masks.append(mask)

        # calc direction, based on the labels mask or the self-supervised mask
        if filename is not None: write_sitk(directions, filename=filename, suffix='dir2')
        # mask direction with a supervised or self-supervised mask
        # TODO: validate if scaling the direction values with the norm provides further value for disease classification
        # mask norm and directions with a supervised or self-supervised mask
        from src.data.Postprocess import minmax_lambda
        # SKM Combine: Check impact of line below, as it was not included in SAX:
        # directions = directions * minmax_lambda([norm_nda, 1, 2])
        # SKM Combine: Differentiation for ax: SAX (1, 2, 3) and LAX (1, 2)
        ax = (1, 2)
        directions_ma, dir_1d_mean = self.Masker.get_masked_array(directions, mask, axis=ax)
        norm_ma, norm_1d_mean = self.Masker.get_masked_array(norm_nda, mask, axis=ax)

        if filename is not None:
            directions_masked = directions_ma.data * ~directions_ma.mask
            write_sitk(directions_masked, filename=filename, suffix='dir_masked')
        return dir_1d_mean, directions_ma.astype(np.float16), norm_1d_mean, norm_ma, ct, all_masks

    def seg_based_direction(self, vect, moved, segmentation, x_val_sax, focus_size, example_path, config,
                            targetfile_type='nii'):
        import SimpleITK as sitk
        import os
        _, directions_seg, _, norm_nda_seg, ct, _ = self.interpret_deformable(vects_nda=vect, masks=segmentation,
                                                                         mask_channels=[2, 3], ct_calculation=[1],
                                                                         dir_axis=0)
        if np.ma.is_masked(directions_seg):
            # directions_seg = directions_seg.data * ~directions_seg.mask
            # directions_seg[directions_seg.mask] = -10 # this works well with the jet transparent color map
            dir = directions_seg.data  # * ~directions_seg.mask
            dir[directions_seg.mask] = -1.
            directions_seg = dir
        if np.ma.is_masked(norm_nda_seg):
            # norm_nda_seg = norm_nda_seg.data * ~norm_nda_seg.mask
            n = norm_nda_seg.data  # * ~norm_nda_seg.mask
            n[norm_nda_seg.mask] = -1.
            norm_nda_seg = n
        # create a bucket-based mask from direction movement
        dir_seg_bucket = directions_seg.copy()
        dir_seg_bucket[(directions_seg < -0.5) & (directions_seg > -1)] = 1  # strong contraction [-1,-.5]
        dir_seg_bucket[(directions_seg < 0) & (directions_seg >= -0.5)] = 2  # moderate contraction [-.5, 0]
        dir_seg_bucket[(directions_seg >= 0) & (directions_seg < 0.5)] = 3  # moderate relaxation [0, .5]
        dir_seg_bucket[(directions_seg >= 0.5) & (directions_seg <= 1)] = 4  # strong relaxation [.5, 1]
        dir_seg_bucket[directions_seg == -1] = 0

        ############################################
        zeros = np.zeros_like(moved)
        zeros[
            :, :, int(ct[0] - focus_size):int(ct[0] + focus_size), int(ct[1] - focus_size):int(ct[1] + focus_size)] = 1

        sitk_dir_seg = [sitk.GetImageFromArray(vol.astype('float32')) for vol in
                        np.transpose(directions_seg[..., None], (0, 3, 1, 2))]
        sitk_norm_seg = [sitk.GetImageFromArray(vol.astype('float32')) for vol in
                         np.transpose(norm_nda_seg[..., None], (0, 3, 1, 2))]
        sitk_foc_seg = [sitk.GetImageFromArray(vol.astype(np.uint8)) for vol in np.transpose(zeros, (0, 2, 3, 1))]

        new_dir_clean_seg = sitk.JoinSeries(sitk_dir_seg)
        new_norm_clean_seg = sitk.JoinSeries(sitk_norm_seg)
        new_foc_clean_seg = sitk.JoinSeries(sitk_foc_seg)

        spacing = config.get('SPACING', (1.0, 1.0))
        spacing = list(reversed(spacing)) + [1.0, 1.0]
        new_dir_clean_seg.SetSpacing(spacing)
        new_norm_clean_seg.SetSpacing(spacing)
        new_foc_clean_seg.SetSpacing(spacing)

        elem = x_val_sax
        file_type = '.nrrd' if '.nrrd' in elem else '.nii.gz'

        export_dir_seg_f_name = os.path.join(example_path,
                                             os.path.basename(elem).replace(file_type,
                                                                            '_dir_seg{}'.format(targetfile_type)))
        export_norm_f_seg_name = os.path.join(example_path,
                                              os.path.basename(elem).replace(file_type,
                                                                             '_norm_seg{}'.format(targetfile_type)))
        export_foc_seg_f_name = os.path.join(example_path,
                                             os.path.basename(elem).replace(file_type,
                                                                            '_foc_seg{}'.format(targetfile_type)))

        sitk.WriteImage(new_dir_clean_seg, export_dir_seg_f_name)
        sitk.WriteImage(new_norm_clean_seg, export_norm_f_seg_name)
        sitk.WriteImage(new_foc_clean_seg, export_foc_seg_f_name)

    def get_rv_lv_dir(self, vects_nda, masks=None, length=-1, plot=True, z=None, dir_axis=0, gtind=None, exp_path=None,
                      patient='temp',
                      save=False):
        from src.utils.detect_phases_from_dir import detect_phases

        lv_args = self.interpret_deformable(dir_axis=dir_axis, masks=masks, length=length, vects_nda=vects_nda,
                                       mask_channels=[2], ct_calculation=[1, 2], as_angle=False)
        lv_dir_1d_mean, lv_directions, lv_norm_1d_mean, lv_norm_nda, lv_ct, lv_mask = lv_args
        lv_ind = detect_phases(dir_1d_mean=lv_dir_1d_mean[:length])

        rv_args = self.interpret_deformable(dir_axis=dir_axis, masks=masks, length=length, vects_nda=vects_nda,
                                       mask_channels=[3], ct_calculation=[3], as_angle=False)
        rv_dir_1d_mean, rv_directions, rv_norm_1d_mean, rv_norm_nda, rv_ct, rv_mask = rv_args
        rv_ind = detect_phases(dir_1d_mean=rv_dir_1d_mean[:length])

        if plot:
            from src.visualization.Visualize import plot_two_direction_instance
            fig = plot_two_direction_instance(lv_dir_1d_mean, rv_dir_1d_mean, lv_directions, rv_directions)
            return fig, [lv_ind, rv_ind]


    def get_focus_point(self, mask2d: np.ndarray, calculation: Union[Literal['septum'], List, int, None] = 'septum',
                        z: int = None, print_ct=False, whole_mask=True):
        from scipy import ndimage

        if calculation == 'septum':
            from src.data.Preprocess import get_ip_from_2dmask

            def _ips_from_slice(slice_2d):
                return get_ip_from_2dmask(np.squeeze(slice_2d).astype(np.uint8))

            if self.CMR3D:
                # one timestep, full z-stack: average the insertion points found over all valid z-slices
                mask3d = mask2d[0]
                first_ips, second_ips = [], []
                for z_slice in mask3d:
                    first, second = _ips_from_slice(z_slice)
                    if first is not None and second is not None:
                        first_ips.append(first)
                        second_ips.append(second)
                focus = None
                if len(first_ips) > 0:
                    fip = np.array(first_ips).mean(axis=0)
                    sip = np.array(second_ips).mean(axis=0)
                    center = np.mean([fip, sip], axis=0)[::-1]  # (x,y) -> (y,x)
                    focus = np.array([int(z), *center])
            else:
                # loop over time, use the first timestep where both insertion points are found
                focus = None
                for t_slice in mask2d:
                    first, second = _ips_from_slice(t_slice)
                    if first is not None and second is not None:
                        focus = np.mean([first, second], axis=0)[::-1]  # (x,y) -> (y,x)
                        break

            if focus is None:
                logging.error(
                    "It was not possible to find intersection points to use the mid of septum as focus point.\n"
                    "Will use whole mask center point instead.")
                return self.get_focus_point(mask2d, calculation=[1, 2, 3], z=z, print_ct=print_ct,
                                            whole_mask=whole_mask)
            if print_ct: print('Using mid of septum as focus point')
            return focus

        if type(calculation) == int:
            calculation = [calculation]
        mask_for_focus = self.Masker.get_as_single_mask(mask2d, channels=calculation,
                                            whole_mask=whole_mask)  # masks of first ts
        if len(mask_for_focus.shape) > 3:
            mask_for_focus = mask_for_focus[0]

        if self.CMR3D:
            focus = np.array([int(z), *ndimage.center_of_mass(mask_for_focus)[1:]])
        else:
            focus = np.array([*ndimage.center_of_mass(mask_for_focus)])

        if print_ct: print(
            f'Using mask(s): {", ".join(np.array(["right ventricle", "myocardium", "left ventricle", "rv outline"])[calculation])} for ct')
        return focus

    @staticmethod
    def get_balanced_center(dim_, norm_msk):
        import scipy
        import numpy as np
        ct_norm = scipy.ndimage.center_of_mass(
            norm_msk)  # x,y =  np.mean(np.where(norm_mask)) compareable results, and usable in a differentiable model with tf
        ct = np.array(dim_) // 2
        ct = (ct + ct_norm) // 2
        return ct

    @staticmethod
    def signed_angle_np(p1, p2):
        p2 = np.broadcast_to(p2, shape=p1.shape)
        assert p1.shape == p2.shape
        shape_ = p1.shape

        size = np.prod(shape_[:-1])
        p1 = p1.reshape((size, shape_[-1]))
        p2 = p2.reshape((size, shape_[-1]))
        ang1 = np.arctan2(p1[..., 1], p1[..., 0])
        ang2 = np.arctan2(p2[..., 1], p2[..., 0])
        rel_angle = np.rad2deg(np.abs(ang1) - np.abs(ang2))
        # for anti clockwise degrees 0 - 360
        # np.rad2deg((ang1 - ang2) % (2 * np.pi))
        rel_angle = rel_angle.reshape(shape_[:-1])

        print('angles (-180:180)', rel_angle.min(), rel_angle.max(), rel_angle.mean())
        bord = 90
        neg_mask = rel_angle < 0
        # rel angle is n the range -180:180, by this relaxing angles have a higher impact on the mean
        # here we create a range -90:90, with abs(angle)>90 = 90 - angle mod 90
        rel_angle = np.where(abs(rel_angle) > bord, abs(rel_angle) - bord, abs(rel_angle))
        rel_angle = np.where(neg_mask, rel_angle * -1, rel_angle)
        # assert np.all((rel_angle < -bord) | (rel_angle > bord))
        print('angles (-90:90)', rel_angle.min(), rel_angle.max(), rel_angle.mean())
        return rel_angle


    def write_random_example_4d_files_to_disk(self, PRETRAINED_SEG, config, example_path, moved, number_of_examples,
                                              segmentation,
                                              vects, x_val_lax, norm_thresh=55, connected_component_filter=None,
                                              mask_channels=None):

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

        self.write_4d_files_to_disk(examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                               vects, x_val_lax, norm_thresh=norm_thresh,
                               connected_component_filter=connected_component_filter, mask_channels=mask_channels)

    def write_4d_files_to_disk(self, examples, focus_size, PRETRAINED_SEG, config, example_path, moved, segmentation,
                               vects, x_val_lax, norm_thresh=55, connected_component_filter=None, mask_channels=None):
        import SimpleITK as sitk
        import os
        from src.utils.Utils_io import save_sitk, rearrange_axis_of_ndarray

        for example in examples:
            dir_1d_mean, directions, norm_1d_mean, norm_nda, ct, _ = self.interpret_deformable(vects_nda=vects[example],
                                                                                          masks=segmentation[
                                                                                              example] if PRETRAINED_SEG else None,
                                                                                          mask_channels=mask_channels
                                                                                          if PRETRAINED_SEG else None,
                                                                                          ct_calculation=[1, 2, 3])

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
                sitk_mask = [sitk.GetImageFromArray(np.flipud(vol.astype(np.uint8))) for vol in
                             np.transpose(segmentation[example], (0,  1, 2, 3))]
                new_mask_clean = sitk.JoinSeries(sitk_mask)
                new_mask_clean.SetSpacing(spacing)
                export_mask_f_name = os.path.join(example_path,
                                                  os.path.basename(elem).replace(file_type, '_mask.nrrd'))
                sitk.WriteImage(new_mask_clean, export_mask_f_name)
                self.seg_based_direction(vects[example], moved[example], segmentation[example], x_val_lax[example],
                                    focus_size, example_path, config, file_type)

        return


class CMRViewMasker:
    """ Reusable class for masking views based on segmentation or rules."""
    def __init__(self, data_json: dict):
        from src.utils.Utils_io import get_post_processing

        self.dataset_json = data_json
        self.view = data_json.get('view', 'sax')
        self.CMR3D = self.view.lower() == 'sax'

        # Settings for masking
        self.post_processing = get_post_processing(data_json)
        self.use_segmentation = self.post_processing.get("use_segmentation", False)

        background_label = ["background", "bg"]
        mask_labels = {"background": 0,
                  "RV": 1,
                  "MLV": 2,
                  "LVC": 3
        }
        self.mask_labels = self.dataset_json.get("labels", mask_labels)
        self.mask_channels = None

        self.norm_threshold = self.post_processing.get("norm_threshold", 40)

        self.connected_component_filter = self.post_processing.get("cc_filter", None)
        if self.connected_component_filter is None: self.connected_component_filter = False

        self.focus_point = self.post_processing.get("focus_point", None)

        if self.use_segmentation is None or self.use_segmentation is False:
            self.PRECOMPUTED_MASKS = False
            val_config_nnunet = False
        else:
            self.PRECOMPUTED_MASKS = self.dataset_json.get("seg_model_path", False)
            val_config_nnunet = self.PRECOMPUTED_MASKS
            self.mask_channels = self.post_processing.get("mask_channels", None)
            if self.mask_channels is None:
                self.mask_channels = [self.mask_labels[l] for l in self.mask_labels.keys() if l not in background_label]



    def get_norm(self, dir_axis,  vects_nda, norm_threshold = None):
        import numpy as np
        from src.models.KerasLayers import minmax_lambda
        from src.data.Preprocess import clip_quantile

        if norm_threshold is None:
            norm_threshold = self.norm_threshold

        # norm of the vector
        norm_nda = np.linalg.norm(vects_nda[..., dir_axis:], axis=-1)
        norm_nda = clip_quantile(norm_nda, 0.99)
        # norm_nda = minmax_lambda([norm_nda, mid, upper])
        norm_msk = norm_nda.copy()
        norm_msk = np.median(norm_msk[:-1],
                             axis=0)  # exclude the cyclic (last) registration step for the mask generation
        threshold = np.percentile(norm_msk, norm_threshold)
        if threshold > 0.3:
            threshold = 0.3
        norm_msk = norm_msk > threshold
        # norm_msk = norm_msk > norm_percentile
        # for norm msk improvements the following did not work well:
        # connected component filtering before COM
        # Gauss smoothing or any other conv operation such as closing etc.
        # usually there are occlusions that stop these methods to work for each patient
        return norm_msk, norm_nda



    @staticmethod
    def filter_for_connected_components(data, pad_size=10):
        """
        This method returns a mask of the largest connected component and all connected components around the largest
        connected component. Including surround components, is used with a rectangular mask around largest connected
        component. Padding around largest component uses the min and max row and column and add/subtract the pad_size
        from it.  Not ideal right now, but fast!
        :param data: the input data
        :param pad_size: the size of the padding around the largest connected component

        :returns: a masked array, including all connected components inside the radius of pad_size around the largest connected component
        """
        largest_component, labeled_ma = CMRViewMasker.get_largest_connected_components(data)

        true_indices = np.argwhere(largest_component)

        row_min = int(np.maximum(true_indices[:, 0].min() - pad_size / 2, 0))
        row_max = int(np.minimum(true_indices[:, 0].max() + pad_size / 2, largest_component.shape[0]))
        col_min = int(np.maximum(true_indices[:, 1].min() - pad_size / 2, 0))
        col_max = int(np.minimum(true_indices[:, 1].max() + pad_size / 2, largest_component.shape[1]))

        pad_mask = np.zeros_like(largest_component, dtype=bool)
        pad_mask[row_min:row_max, col_min:col_max] = True

        label_binary_ma = labeled_ma != 0
        padded_array = np.logical_or(label_binary_ma, pad_mask)

        ret_ma, _ = CMRViewMasker.get_largest_connected_components(padded_array, padded_array)

        ret_ma = ret_ma * data

        return ret_ma


    @staticmethod
    def get_largest_connected_components(data, data_bin_ma=None):
        """
        This method extracts the largest connected component and a labelled mask of all connected components.
        :param data: the input data an array with various connected components
        :return: the largest connected component and a labelled mask of all connected components
        """
        from scipy import ndimage
        if data_bin_ma is None:
            data_bin_ma = np.abs(data) > 0

        labeled_ma, num_labels = ndimage.label(data_bin_ma)
        region_sizes = ndimage.sum(data_bin_ma, labeled_ma, range(num_labels + 1))

        largest_component_label = np.argmax(region_sizes)
        largest_component = (labeled_ma == largest_component_label) * data

        return largest_component, labeled_ma


    @staticmethod
    def get_rv_outline_as_mask_3D_sax(rv_mask, myo_mask, faked_rv_myo, timestep, include_septum=False):
        import cv2 as cv
        from scipy.spatial.distance import cdist
        for z in range(rv_mask.shape[1]):
            rv_slice = rv_mask[timestep, z]
            rv_contours, _ = cv.findContours(rv_slice.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(rv_contours) == 0:
                continue
            rv_contours = rv_contours[0][:, 0]
            if not include_septum:
                myo_slice = myo_mask[timestep, z]
                myo_contours, _ = cv.findContours(myo_slice.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if len(myo_contours) == 0:
                    continue
                myo_contours = myo_contours[0][:, 0]

                # exclude the septum using only rv contour values which are NOT next to the myo
                rv_contours = rv_contours[np.min(cdist(rv_contours, myo_contours), axis=1) > 1]
            new_slice = np.zeros(faked_rv_myo.shape[2:])
            new_slice[rv_contours[:, 1], rv_contours[:, 0]] = 1

            # dilate rv outline to increase thickness (but only to inside)
            kernel = np.ones((3, 3), np.uint8)
            new_slice = cv.dilate(new_slice, kernel=kernel, iterations=1)
            new_slice = np.logical_and(new_slice, rv_mask[timestep, z])
            faked_rv_myo[timestep, z] = new_slice

        return faked_rv_myo


    @staticmethod
    def get_rv_outline_as_mask_2D_lax(rv_mask, myo_mask, faked_rv_myo, include_septum=False):
        import cv2 as cv
        from scipy.spatial.distance import cdist

        for timestep in range(rv_mask.shape[0]):
            rv_timestep = rv_mask[timestep]
            rv_contours, _ = cv.findContours(rv_timestep.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(rv_contours) == 0:
                continue
            rv_contours = rv_contours[0][:, 0]
            if not include_septum:
                myo_timestep = myo_mask[timestep]  # if 3D [timestep, z]
                myo_contours, _ = cv.findContours(myo_timestep.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if len(myo_contours) == 0:
                    continue
                myo_contours = myo_contours[0][:, 0]
                rv_contours = rv_contours[np.min(cdist(rv_contours, myo_contours), axis=1) > 1]

            new_slice = np.zeros(faked_rv_myo.shape[1:])  # SAX: .shape[2:]
            new_slice[rv_contours[:, 1], rv_contours[:, 0]] = 1

            # dilate rv outline to increase thickness (but only to one side)
            # define thickness
            dilate_thickness = new_slice.shape[0] // 32
            kernel = np.ones((dilate_thickness, dilate_thickness), np.uint8)

            new_slice = cv.dilate(new_slice, kernel=kernel, iterations=1)
            # Add if for LAX (as only done in lax)
            new_slice = CMRViewMasker.remove_masking_in_atrium_direction(new_slice, myo_contours, rv_timestep)

            inverse_rv_mask = np.logical_not(rv_mask[timestep])
            faked_rv_myo[timestep] = np.logical_and(new_slice, inverse_rv_mask)
        return faked_rv_myo

    def get_rv_outline_as_mask(self, masks, include_septum=False):
        """
        This method creates a mask of the right ventricle myocardium,
        if a mask of the right ventricular cavity exclusive the myocardium is given.

        :param masks: array of masks (2D+t)
        :param include_septum: determines if the septum should be included in the rv mask or not

        :return: mask of simulated right ventricle myocardium
        :rtype: ndarray
        """
        if len(masks.shape) == 4:
            masks = masks[:, :, :, 0]
        rv_mask = masks == 3
        myo_mask = masks == 2
        faked_rv_myo = np.zeros_like(masks)
        ## SKM Combine: Add option for 3D volumes vs. 2D
        for  timestep in range(rv_mask.shape[0]):
            if self.CMR3D:
                faked_rv_myo = CMRViewMasker.get_rv_outline_as_mask_3D_sax(rv_mask, myo_mask, faked_rv_myo, timestep)
            else:
                faked_rv_myo = CMRViewMasker.get_rv_outline_as_mask_2D_lax(rv_mask, myo_mask, faked_rv_myo)
        return faked_rv_myo


    @staticmethod
    def remove_masking_in_atrium_direction(rv_contours_slice, myo_contours, rv_mask):
        """
        Removes the surplus of the rv myocardium mask in the aria of the tricuspid valve (in direction of the atrium).
        """
        import cv2 as cv
        from scipy import ndimage
        myo_mask = np.zeros_like(rv_mask)
        myo_mask[myo_contours[:, 1], myo_contours[:, 0]] = 1

        # create a intersection line between the left ventricle myocardium (+ septum) and the right ventricle cavity
        intersection_line = np.logical_and(myo_mask, rv_contours_slice)
        indices = np.where(intersection_line)

        if indices[0].size == 0 or indices[1].size == 0:
            more_dilated_rv = cv.dilate(rv_contours_slice, kernel=(10, 10), iterations=1)
            intersection_line = np.logical_and(myo_mask, more_dilated_rv)
            indices = np.where(intersection_line)
            if indices[0].size == 0 or indices[1].size == 0:
                more_dilated_rv = cv.dilate(more_dilated_rv, kernel=(5, 5), iterations=1)
                intersection_line = np.logical_and(myo_mask, more_dilated_rv)
                indices = np.where(intersection_line)

        if indices[0].size == 0 or indices[1].size == 0:
            indices = np.where(rv_contours_slice == 1)
            if indices[0].size == 0 or indices[1].size == 0:
                logging.error(f"No intersection line found.")
                return rv_contours_slice

        first_column = indices[1].min()
        first_row_index = indices[0][indices[1] == first_column][0]
        rvip_a = np.array([first_row_index, first_column])

        last_column = indices[1].max()
        last_row_index = indices[0][indices[1] == last_column][0]
        rvip_b = np.array([last_row_index, last_column])

        distance_a_b = np.linalg.norm(rvip_b - rvip_a)
        radius_size = int(distance_a_b * 0.5)

        center_of_rv = ndimage.center_of_mass(rv_mask)

        distance_a = np.linalg.norm(center_of_rv - rvip_a)
        distance_b = np.linalg.norm(center_of_rv - rvip_b)
        if distance_a > distance_b:
            rvip_atrium = rvip_b
        else:
            rvip_atrium = rvip_a

        atrial_surplus_mask = np.zeros_like(rv_mask)
        atrial_surplus_mask[rvip_atrium[0], rvip_atrium[1]] = 1
        atrial_surplus_mask = (atrial_surplus_mask > 0).astype(np.uint8)
        # Draw a filled circle around atrial intersection point
        cv.circle(atrial_surplus_mask, (rvip_atrium[1], rvip_atrium[0]), radius_size, (1), -1)

        atrial_surplus_mask = np.logical_not(atrial_surplus_mask)
        rv_contours_slice = np.logical_and(rv_contours_slice, atrial_surplus_mask)

        return rv_contours_slice


    def get_as_single_mask(self, segmentation, channels, whole_mask=True) -> np.ndarray:
        '''

        Parameters
        ----------
        masks array of probabilities for each mask
        channels that should be used for masking: 3 rv outline, 0 rv, 1 myo, 2 lv
        different_labels used for mitk where labels used for the different masks eg. 1, 2, 3
        threshold

        Returns
        -------

        '''

        # uses connected component
        mask = segmentation
        channels = np.array(channels)
        rv_outline = 3 in channels or 4 in channels
        include_rv_septum = 4 in channels
        if whole_mask and len(channels) >= 3:
            channels = channels
        else:
            channels = channels[(channels > 0) & (channels < 3)]
        _ = np.zeros_like(mask)
        for channel in channels:
            _ = np.logical_or(_, mask == (channel))
        mask = _
        if rv_outline:
            fake_rv_myo = self.get_rv_outline_as_mask(segmentation, include_septum=include_rv_septum)
            mask = np.logical_or(mask, fake_rv_myo[..., None])  # SKM Combine: for SAX only: ... , fake_rv_myo)
        mask = mask
        return mask


    def get_combined_masking_norm(self, vects_nda, mask, dir_axis=0):
        """

        """

        vects_nda_ma = vects_nda * np.broadcast_to(mask, shape=vects_nda.shape)
        heart_coverage_percentage = np.sum([vects_nda_ma != 0]) / vects_nda_ma.size
        new_norm_percentile = (100 - heart_coverage_percentage * 100) + heart_coverage_percentage * self.norm_threshold
        norm_mask, norm_nda = self.get_norm(dir_axis, new_norm_percentile, vects_nda_ma)

        return vects_nda_ma, norm_mask, norm_nda


    @staticmethod
    def get_masked_array(array, mask, axis=(1, 2), aggregation_func=np.ma.mean):
        """
        Returns a masked array and a one dimensional array representing the in-plane aggregation of the array.

        :param array: array to be masked
        :type array: np.ndarray
        :param mask: mask to apply
        :type mask: np.ndarray
        :param axis: axis along which to apply the mask
        :param aggregation_func: aggregation function to apply to the array (np.ma.mean or np.ma.median)

        :returns:   masked array (array_ma np.array) one dimensional array representing the in-plane aggregation of the array (array_1d np.array).
        """
        mask = np.squeeze(mask)
        if mask.shape != array.shape:
            mask = np.broadcast_to(mask, array.shape)
        array_ma = np.ma.masked_array(array, mask=np.broadcast_to(~mask, shape=array.shape))
        array_1d = aggregation_func(array_ma, axis=axis)
        return array_ma, array_1d




if __name__ == "__main__":
    import argparse, os, sys
    from src.utils.Utils_io import get_json
    from src.utils.Utils_io import get_post_processing

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually the exp root parameters should yield to a config, which encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/sds/sd20i001/sven/code/exp/miccai_baseline')
    parser.add_argument('-data', action='store', default='')
    parser.add_argument('-work_dir', action='store', default='/mnt/ssd/git/cmr-phase-detection')
    parser.add_argument('-c2l', action='store_true', default=False)

    results = parser.parse_args()
    os.chdir(results.work_dir)
    sys.path.append(os.getcwd())
    print('given parameters: {}'.format(results))

    # get all cfgs and dataset json (we expect to find 4 as we usually train a 4-fold cv)
    # call the predict_fn for each cfg
    cfg_files = get_json('config/config.json', results.exp_root)
    dataset_files = get_json('config/dataset.json', results.exp_root)

    patients_txt_file = os.path.join(results.exp_root, 'pred', 'patients.txt')

    if os.path.exists(patients_txt_file):
        import os

        # removing previous inference files using the os.remove() method
        os.remove(patients_txt_file)

    for cfg, data_json in zip(cfg_files, dataset_files):
        cmr_phase_predictor = CMRPhaseDetector(model_config=cfg, data_info_path=data_json, data_root=results.data, exp_path=results.exp_root)

        try:
            cmr_phase_predictor.predict(number_of_examples=1)
            pass
        except Exception as e:
            print(e)

        try:
            logging.info('Predict cardiac keyframes and save them as figures')
            cmr_phase_predictor.predict_phase_from_deformable()
        except Exception as e:
            logging.error('{} predict phase failed with: {}'.format(results.exp_root, e))

    exit()
