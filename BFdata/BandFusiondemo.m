addpath('./libsvm/');
addpath('./drtoolbox/');
load ..\BFdata\Indian_pines_corrected.mat;
% load the ground truth and the hyperspectral image
%%% image fusion
img2=average_fusion(indian_pines_corrected,29);