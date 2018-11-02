% Created by Tilke Judd, Feb 2012
% Modified by Zoya Bylinskii, April 2014
% Modified by Matthias Kümmerer, September 2018

% Outline to test new models against the existing benchmark

function [scores,tp,fp] = TestNewModels(origFolder,resultDirName,nameOfModel,metrics,metricNames,ext,probabilisticModel)

%% PARAMETERS

% probabilisticModel indicates whether the model is submitted as classic saliency maps
% or as probabilistic model of fixation density prediction where we compute
% metric specific saliency maps according to Kümmerer et al, ECCV 2018
% the metric specific saliency maps are computed with external python code
% and are expected in subdirectories of origFolder
if nargin < 7 || isempty(probabilisticModel)
    probabilisticModel = 0;
end

% origFolder = fullfile('../../SALIENCYMAPStotest',nameOfModel);
if nargin < 6 || isempty(ext)
    ext = 'jpg';
end
tp = nan; fp = nan; % returns tp and fp values for all images for the last AUC metric

metrics_web = [];
if nargin < 4 || isempty(metrics) || isempty(metricNames)
    metrics = {'InfoGain','S','ROC','ROC_borji','sROC_borji','CC','NSS','KL','EMD'};
    metricNames = {'InfoGain','Similarity','AUC (Judd)',...
                             'AUC (Borji)','shuffled AUC','Cross-correlation',...
                             'Normalized Scanpath Saliency','KL','Earth Mover Distance'};
    % subdirectories with metric specific saliency maps (Kümmerer et al, ECCV 2018)
    metricSpecificSaliencyMapDirs = {'IG', 'SIM', 'AUC', 'AUC', 'sAUC', 'CC', 'NSS', 'KLdiv', 'CC'};
    metrics_web = {'ROC','S','EMD','ROC_borji','sROC_borji','CC','NSS','KL','InfoGain'};
end
  

%% SETUP
addpath(genpath('/data/graphics/PAMIbenchmark-master/PAMIbenchmark/Scripts/code_forMetrics'));
addpath(genpath('/data/graphics/PAMIbenchmark-master/PAMIbenchmark/Scripts/FastEMD/'));
if nargin < 2 || isempty(resultDirName)
    resultDirName = fullfile(origFolder,['Results_',date]);
end
if ~exist(resultDirName,'dir')
    mkdir(resultDirName);
end

FIXATIONDB = '/data/graphics/PAMIbenchmark-master/PAMIbenchmark/SavedData/FixationDBCleaned.mat';
FIXATIONMAPS = '/data/graphics/PAMIbenchmark-master/PAMIbenchmark/FIXATIONMAPS';
STIMULI = '/data/graphics/PAMIbenchmark-master/PAMIbenchmark/ALLSTIMULI';
HASH = '/data/graphics/PAMIbenchmark-master/PAMIbenchmark/SavedData/hash.mat'; 

load(FIXATIONDB);

%% write results to file
%fid = fopen(fullfile(origFolder,resultDirName,'results.txt'),'w');
curfile = fullfile(resultDirName,'results.txt');
if exist(curfile,'file')
    fid = fopen(curfile,'a');
else
    fid = fopen(curfile,'w');
end

%% Sanity check
if ~probabilisticModel
    temp = dir(fullfile(origFolder,['*.',ext]));
    if length(temp)<300
        fprintf('Did not find 300 %s files in %s. Terminating.',ext,origFolder)
        fprintf(fid,'Did not find 300 %s files in %s. Terminating.',ext,origFolder);
        return
    end
else
    for ii = 1:length(metricSpecificSaliencyMapDirs)
        metricDir = fullfile(origFolder, metricSpecificSaliencyMapdirs(ii));
        temp = dir(fullfile(metricDir, ['*.',ext]));
        if length(temp)<300
            fprintf('Did not find 300 %s files in %s. Terminating.',ext,metricDir)
            fprintf(fid,'Did not find 300 %s files in %s. Terminating.',ext,metricDir);
            return
        end
    end
end

%% To see how the basic provided model performs on benchmark
% Receive: 300 saliency images created from the images online
% URL is here:
% http://people.csail.mit.edu/tjudd/SaliencyBenchmark/BenchmarkIMAGES.zip

fprintf(fid,'Thanks for submitting your saliency maps to the benchmark. We have run your model on the saliency benchmark and the scores are:\n\n');

%% Run code on multiple cores for speed-up
try
    matlabpool open
end

%% Get performances on benchmark dataset under all the metrics

if exist(fullfile(resultDirName,'Results.mat'))
    load(fullfile(resultDirName,'Results.mat'));
else
    scores = struct();
end

for ii = 1:length(metrics)
    tic
    metric = metrics{ii};
    metricName = metricNames{ii};
    fprintf('%s metric: ',metricName);
    if ~probabilisticModel
        saliencyMapDir = origFolder;
    else
        saliencyMapDir = fullfile(origFolder, metricSpecificSaliencyMapDirs(ii));
    end
    fprintf(' reading saliency maps from %s', saliencyMapDir);
    [scores.(metric),tp,fp] = scoreModel_parallel(saliencyMapDir,metric,ext,FixationDBCleaned,FIXATIONMAPS,STIMULI,HASH);
    save(fullfile(resultDirName,'Results.mat'),'scores'); % save after each metric, just in case
    meanscore = mean(scores.(metric));
    fprintf(' %2.4f ',meanscore);
    fprintf('(time elapsed: %2.2f s)\n',toc)
    % write results to file
    fprintf(fid,'%s metric: %2.4f\n',metricName,meanscore);
end


%% finish up

fprintf(fid,'\nTo see how this compares to other models see the other scores on the saliency benchmark site (http://saliency.mit.edu) and our paper for reference.\n\n');

%fprintf(fid,'We often find that models are improved by a small amount center bias and we believe yours would as well.  To optimize these parameters, you can test and refine the performance of your model on the ICCV 2009 dataset (from http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html) which does include access to human fixations.\n\n');
fprintf(fid,'We can include this model and score on our benchmark website.  In this case, please inform us about what you would like the model named and if you have any links to a website or code.\n\n');
fprintf(fid,'As you continue to work on the paper about your model, you can mention these scores on this benchmark as a way of comparing against many other models.  Please cite:\n');
fprintf(fid,'Z. Bylinskii, T. Judd, A. Borji, L. Itti, F. Durand, A. Oliva, and A. Torralba. MIT Saliency Benchmark. Available at: http://saliency.mit.edu\n')
fprintf(fid,'Z. Bylinskii, T. Judd, A. Oliva, A. Torralba, F. Durand. What do different evaluation metrics tell us about saliency models? arXiv:1604.03605, 2016 (https://arxiv.org/abs/1604.03605)\n')
fprintf(fid,'T. Judd, F. Durand, and A. Torralba. A Benchmark of Computational Models of Saliency to Predict Human Fixations. MIT technical report, 2012 (http://hdl.handle.net/1721.1/68590)\n')
fprintf('Nothing left to do.')
%try
%    matlabpool close
%end
fclose(fid);
fprintf('DONE EVERYTHING! Time to send results...\n')
%%
 if nargin > 2 && ~isempty(nameOfModel)
     %%
     curfile = fullfile(origFolder,['i4.',ext]);
     im = imresize(imread(curfile),[200,200]);
     imwrite(im,['../../MAPsamples/',nameOfModel,'.jpg']);
     modelName = nameOfModel;
     save(fullfile(origFolder,'modelName.mat'),'modelName');
 end

% Write website block
fid_web = fopen(fullfile(resultDirName,'results_web.txt'),'w');
fprintf(fid_web,'<tr>\n\t<td>%s</td>\n\t<td>%s</td>\n\t<td>%s</td>\n\t', char(origFolder), 'published', 'code');

%<tr>
%        <td>DeepFix</td>
%        <td>Srinivas S S Kruthiventi, Kumar Ayush, R. Venkatesh Babu <br><a href="http://arxiv.org/abs/1510.02927">DeepFix: A Fully Convolutional Neural Network for predicting Human Eye Fixations [arXiv 2015]</a></td>
%        <td></td>
%        <td> 0.87</td> <!--AUC Judd-->
%        <td>0.67</td> <!--Sim-->
%        <td>2.04</td> <!--EMD-->
%        <td>0.80</td> <!--AUC Borji-->
%        <td>0.71</td> <!--sAUC-->
%        <td>0.78</td> <!--CC-->
%        <td>2.26</td> <!--NSS-->
%        <td>2.26</td> <!--KL-->
%        <td>first tested: 10/02/2015 <br>last tested: 10/02/2015<br><font color="DarkGray"> maps from authors</font></td>
%        <td><img src="MAPsamples/Srinivas_Kruthiventi_MIT300.jpg"></td>
%    </tr>

if ~isempty(metrics_web)
    for jj = 1:length(metrics_web)
        metric_w = metrics_web{jj};
        %[scores.(metric_w),tp,fp] = scoreModel_parallel(origFolder,metric,ext,FixationDBCleaned,FIXATIONMAPS,STIMULI,HASH);
        meanscore_w = mean(scores.(metric_w));
        fprintf(fid_web,'<td>%.2f</td>\n\t',meanscore_w);
    end
    fprintf(fid_web,'<td>first tested: %s<br>last tested: %s<br><font color="DarkGray"> maps from authors</font></td>\n\t', 'FIRST TESTED', date);
    fprintf(fid_web, '<td><img src="%s"></td>\n</tr>', 'IMAGE')
    fclose(fid_web)
end
