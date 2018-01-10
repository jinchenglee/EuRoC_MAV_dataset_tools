% ------------------------------------------------------------------------------
% Function : minimal load dataset test script
% Project  : IJRR MAV Datasets
% Author   : www.asl.ethz.ch
% Version  : V01  28AUG2015 Initial version.
% Comment  :
% Status   : under review
% ------------------------------------------------------------------------------

addpath('/work/git_repo/dataset_tools/matlab/quaternion');

% set dataset folder
datasetPath = ...
  '/work/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy';

disp(' ');
disp([' > dataset_load_test [', datasetPath, ']']);

assert(exist(datasetPath, 'dir') > 0, ...
  ' > Dataset folder does not exist, Please set datasetPath.');

% load dataset
dataset = dataset_load(datasetPath);

% plot dataset
dataset_plot(dataset);
