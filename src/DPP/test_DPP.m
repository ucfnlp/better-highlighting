%% test dpp for one C parameter
% assume 'tt' is loaded from saved file
% clear variables

load('res_TAC.mat', 'res', 'tt', 'alpha_list', 'q_list', 'th_list', 'dataset', 'type_list');

%%
if exist('dataset','var') ~= 1
    dataset = 0;
end

%% parameters
min_words = 10;
use_CNN_sim = 3;
pos_sorting = 1;
is_concat_feat = 1;
which = zeros(11,1);
which(1:6) = 1;

% ===================== %
type_id = 1;
alpha = 0.7;
q = 2;
% ===================== %

al_id = 0;
ai = 1;
for al = alpha_list
    if alpha == al
        al_id  = ai;
        break;
    end
    ai  = ai  + 1;
end
q_id = 0;
qi = 1;
for ql = q_list
    if q == ql
        q_id  = qi;
        break;
    end
    qi  = qi  + 1;
end

fprintf('alpha_id:%d, q_id:%d, %s\n', al_id, q_id, type_list{type_id});
tt_ = tt{type_id, al_id}(:, q_id);

type = type_list{type_id}; % % tree, xlnet, sent
th_cut = th_list;

%%
if dataset == 0    
    data_path_name = 'DUC';
    compRouge = @get_syssum_rouge_DUC;
elseif dataset == 1    
    data_path_name = 'TAC';
    compRouge = @get_syssum_rouge_TAC;
elseif dataset == 2    
    data_path_name = 'cnn';
    data_name = 'CNN';
    compRouge = @get_syssum_rouge_CNN;
end

base = '../../data/';
testdir  = [base [data_path_name, '/', type, '/test']];

rouge_filename = sprintf('run_pyrouge_%s.py', data_path_name);
rouge_py_file = fullfile('../../data/pyrouge', rouge_filename);

%%
[testdocs, context] = read_text_test(dataset, type, th_cut);
disp('test docs are loaded.');

%%
[testdocs, ~, ~] = assign_features(testdocs, ...
        [base [data_path_name, '/', type, '/shared/prp']], which, use_CNN_sim, alpha);
disp('Assigned test features ...')
%% test DPP
testdocs = testdpp(testdocs, tt_, min_words, is_concat_feat, alpha);
[rouge_scores, avg_sents] = compRouge(testdocs, testdir, rouge_py_file, pos_sorting);

