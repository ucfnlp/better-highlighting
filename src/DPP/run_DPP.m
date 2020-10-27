clear variables
type_list = {'sent', 'tree', 'xlnet'};
alpha_list = [0.7, 0.8, 0.9];
q_list = -1:3;
th_list = 0.2;

dataset = 1; % 0: DUC, 1: TAC, 2: CNN

main_DPP(dataset, type_list, alpha_list, q_list, th_list);
