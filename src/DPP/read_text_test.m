function [testdocs, context] = read_text_test(dataset, type, th)

addpath(genpath(pwd));

%% parameters
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

%% configurations
base = '../../data/';
testdir  = [base [data_name, '/', type, '/test']];

%% context
disp('Loading context...');
context.wordmap = read_wordmap([base [data_name, '/', type, '/shared/dict']]);
context.n = length(context.wordmap);

%% load test docs
disp('Loading test documents...');
context.idf = read_idf(fullfile(testdir, 'idf'), context);
testdocs = read_docs(testdir, context, re_exp);

%% load CNN similarity / save most similar sentences (top-10)
disp('Loading test CNN features ...')
base_dir_CNN_weight = fullfile(testdir, bert_directory);
testdocs = load_pair_single(base_dir_CNN_weight, testdocs, is_sigmoid, isSegOn, th);

end
