function main_DPP(dataset, type_list, alpha_list, q_list, th_list)
%%
addpath(genpath(pwd));

%%
if dataset == 0    
    data_name = 'DUC';
elseif dataset == 1    
    data_name = 'TAC';
elseif dataset == 2
    data_name = 'CNN';
end

%%
res = cell(length(type_list), numel(alpha_list));
tt = cell(length(type_list), numel(alpha_list));

stg = tic;
t_id = 1;
for type=type_list    
    for th = th_list
        [traindocs, train_context] = read_text_train(dataset, type{1}, th);
        disp('train docs are loaded.');
        [testdocs, test_context] = read_text_test(dataset, type{1}, th);
        disp('test docs are loaded.');
        
        st = tic;
        a_id = 1;
        for a=alpha_list            
            fprintf('processing [%s - %s - a:%.02f - th:%.01f] ...\n', data_name, type{1}, a, th);
            [res{t_id, a_id}, tt{t_id, a_id}] = DPP(dataset, traindocs, testdocs, train_context, test_context, type{1}, a, q_list);        
            a_id = a_id + 1;
        end
        toc(st);
        t_id = t_id + 1;
    end
end
toc(stg);

save(sprintf('res_%s.mat', data_name), 'res', 'tt', 'type_list', 'alpha_list', 'q_list', 'th_list', 'dataset');
end
