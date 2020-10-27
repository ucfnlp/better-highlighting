function [traindocs, context] = read_text_train(dataset, type, th)

addpath(genpath(pwd));

%% parameters
% C = 2^-3;
if dataset == 0
    re_exp = '^d\d+';
    data_name = 'DUC';
elseif dataset == 1
    re_exp = '^D\d+\-[A-B]';
    data_name = 'TAC';
elseif dataset == 2
    re_exp = '\d+';
    data_name = 'cnn';
end

isSegOn = false;    % true to update similarity scores with threshold
bert_directory = 'BERT_features';
is_sigmoid = 0;     % true for applying sigmoid to sim, imp outputs

% voca_name = '_voca50k_300d';
% q = q_val; %-2:2; %3:6; %-1:1; %-3:3;   % -3:12

%% configurations
base = '../../data/';
traindir = [base [data_name, '/', type, '/train']];

%% context
disp('Loading context...');
context.wordmap = read_wordmap([base [data_name, '/', type, '/shared/dict']]);
context.n = length(context.wordmap);

%% load train docs
disp('Loading training documents...');
context.idf = read_idf(fullfile(traindir, 'idf'), context);
traindocs = read_docs(traindir, context, re_exp);

%% load CNN similarity / save most similar sentences (top-10)
disp('Loading train CNN features ...')
base_dir_CNN_weight = fullfile(traindir, bert_directory);
traindocs = load_pair_single(base_dir_CNN_weight, traindocs, is_sigmoid, isSegOn, th);

end
