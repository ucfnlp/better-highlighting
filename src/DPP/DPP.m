function [rouge_scores, tt] = DPP(dataset, traindocs, testdocs, train_context, test_context, type, alpha, q)
%% parameters
if dataset == 0    
    data_name = 'DUC';
    compRouge = @get_syssum_rouge_DUC;
elseif dataset == 1    
    data_name = 'TAC';
    compRouge = @get_syssum_rouge_TAC;
elseif dataset == 2    
    data_name = 'cnn';
    compRouge = @get_syssum_rouge_CNN;
end

base = '../../data/';
testdir  = [base [data_name, '/', type, '/test']];
rouge_filename = sprintf('run_pyrouge_%s.py', data_name);
rouge_py_file = fullfile('../../data/pyrouge', rouge_filename);
min_words = 10;     % min # words per selected sentence
use_CNN_sim = 3;    % 0: no CNN(tfidf), 1: CNN, 2: max(CNN,tfidf), 3: avg(CNN,tfidf)
w_file = 'weights/imp_cls_w.mat';
is_concat_feat = 1;
pos_sorting = 1;

feat_len = [1, 21, 5, 6, 21, 1, 0,0,0,0,0]; % no global bin for similarity & lexrank
which = zeros(11,1);
which(1:6) = 1;

if is_concat_feat
    disp('use base+BERT_imp concatenated features')
else
    disp('use base feature only')
end

%% assign features
if is_concat_feat
    disp('Assign train features ...')
    [traindocs, ~, ~] = assign_features(traindocs, ...
        [base [data_name, '/', type, '/shared/prp']], which, use_CNN_sim, alpha);
    
    disp('Assign test features ...')
    [testdocs, ~, ~] = assign_features(testdocs, ...
        [base [data_name, '/', type, '/shared/prp']], which, use_CNN_sim, alpha);
end

% don't load gold/refs, just to make sure we don't cheat
traindocs = assign_refs(traindocs, train_context);
testdocs = assign_refs(testdocs, test_context);

%% train dpp
W_weights = load(w_file);
W_weights = W_weights.W_Extract;
assert (length(W_weights) == 769);
W_weights(end-10:end)
W_weights = W_weights(1:end-1)';
theta0 = double(W_weights);

disp('Train dpp ...')
if is_concat_feat    
    feat_tot_len = length(theta0) + sum(feat_len(logical(which)));
else
    feat_tot_len = sum(feat_len(logical(which)));
end
tt = zeros( feat_tot_len, length(q) );
for c=1:length(q)
    C = 2^q(c) / length(traindocs);
    theta = traindpp_MLE(traindocs, C, theta0, is_concat_feat, alpha);
    tt(:,c) = theta;
end

%% test dpp
rouge_scores = zeros(length(q), 12);    % F1,P,R scores of R-1,2,L,SU4
avg_sents = zeros(length(q), 1);        % # of average sentences
for c=1:length(q)
    testdocs = testdpp(testdocs, tt(:,c), min_words, is_concat_feat, alpha);
    [rouge_scores(c,:), avg_sents(c)] = compRouge(testdocs, testdir, rouge_py_file, pos_sorting);
end


return
